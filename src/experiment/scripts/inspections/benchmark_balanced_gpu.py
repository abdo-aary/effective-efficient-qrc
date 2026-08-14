"""Benchmark exact balanced-reservoir feature extraction on one NVIDIA GPU."""

from __future__ import annotations

import argparse
import json
from time import perf_counter
from typing import Any

import numpy as np

from src.backends.nvidia import NvidiaBackend
from src.backends.nvidia._engine import make_runner, pack_program
from src.core.capabilities import Precision
from src.core.factories import (
    gaussian_jl_projection,
    ring_topology,
    sample_balanced_reservoirs,
)
from src.core.program import (
    ObservableSet,
    ProjectionSpec,
    QuaRKProgram,
    ResetChannelSpec,
)
from src.core.requests import ExecutionSpec
from src.estimators.exact import ExactFeatureEstimator
from src.experiment.numerical import execute_nvidia_rate_sweep_sharded


def _program(
    *,
    n: int,
    reservoirs: int,
    window_length: int,
    tau_plus: float,
    seed: int,
) -> QuaRKProgram:
    topology = ring_topology(n)
    return QuaRKProgram(
        projection=ProjectionSpec(
            gaussian_jl_projection(
                input_dim=1,
                encoded_width=n,
                rng=np.random.default_rng(seed + 1),
            )
        ),
        topology=topology,
        reservoirs=sample_balanced_reservoirs(
            num_reservoirs=reservoirs,
            topology=topology,
            lambda_plus=np.exp(-1.0 / tau_plus),
            gamma=0.5,
            mixer_rng=np.random.default_rng(seed + 2),
            memory_rng=np.random.default_rng(seed + 3),
        ),
        reset_channel=ResetChannelSpec(),
        observables=ObservableSet.cycle_complete(
            num_qubits=n,
            edges=topology.edges,
        ),
        window_length=window_length,
        angle_map="tanh",
        angle_scale=np.pi,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--gpu-ids", type=int, nargs="+")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--reservoirs", type=int, default=64)
    parser.add_argument("--window-length", type=int, default=512)
    parser.add_argument("--windows", type=int, default=1)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[64])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--profile-phases", action="store_true")
    parser.add_argument("--tau-plus-values", type=float, nargs="+")
    return parser


def _event_seconds(cp: Any, operation: Any) -> float:
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    operation()
    end.record()
    end.synchronize()
    return float(cp.cuda.get_elapsed_time(start, end) / 1_000.0)


def _profile_phases(
    *,
    program: QuaRKProgram,
    windows: np.ndarray,
    gpu_id: int,
    chunk_size: int,
) -> dict[str, float]:
    import cupy as cp

    flat, layout = pack_program(program, windows)
    runner = make_runner(
        program,
        state_dtype=Precision.COMPLEX128.value,
        chunk_size=chunk_size,
        gpu_id=gpu_id,
        output_backend="cupy",
        output_kind="expectation",
    )
    masks = [
        runner._masks_from_pauli_label(label) for label in program.observables.labels
    ]
    captured: dict[str, Any] = {}
    with cp.cuda.Device(gpu_id):
        rows = cp.asarray(flat[:chunk_size], dtype=cp.float64)
        plus = cp.asarray(
            runner._plus_state(program.num_qubits), dtype=cp.complex128
        )

        def evolve() -> None:
            captured["states"], captured["weights"] = (
                runner._run_chunk_cupy_ensemble_device(
                    rows=rows,
                    layout=layout,
                    angle_positioning_name=program.angle_map,
                    plus_state=plus,
                )
            )

        evolution_seconds = _event_seconds(cp, evolve)

        def extract() -> None:
            captured["features"] = runner._expectations_from_ensemble_cupy(
                states=captured["states"],
                weights=captured["weights"],
                pauli_masks=masks,
                n=program.num_qubits,
            )

        extraction_seconds = _event_seconds(cp, extract)
        pool_bytes = float(cp.get_default_memory_pool().total_bytes())
    return {
        "profile_rows": float(min(chunk_size, flat.shape[0])),
        "evolution_seconds": evolution_seconds,
        "extraction_seconds": extraction_seconds,
        "extraction_fraction": extraction_seconds
        / (evolution_seconds + extraction_seconds),
        "memory_pool_bytes": pool_bytes,
    }


def main() -> None:
    args = _parser().parse_args()
    if min(args.n, args.reservoirs, args.window_length, args.windows, args.trials) < 1:
        raise ValueError("All benchmark dimensions and trial counts must be positive.")

    tau_plus_values = tuple(args.tau_plus_values or (32.0,))
    program = _program(
        n=args.n,
        reservoirs=args.reservoirs,
        window_length=args.window_length,
        tau_plus=max(tau_plus_values),
        seed=args.seed,
    )
    windows = np.random.default_rng(args.seed + 4).uniform(
        -1.0,
        1.0,
        size=(args.windows, args.window_length, 1),
    )
    estimator = ExactFeatureEstimator(precision=Precision.COMPLEX128)
    lambda_0 = float(np.exp(-1.0))
    lambda_anchor = float(np.exp(-1.0 / max(tau_plus_values)))
    memory_uniforms = np.log(program.reservoirs.reset_rates / lambda_0) / np.log(
        lambda_anchor / lambda_0
    )
    lambda_plus_values = np.exp(
        -1.0 / np.asarray(tau_plus_values, dtype=np.float64)
    )
    reset_rate_matrix = lambda_0 * np.power(
        lambda_plus_values[:, None] / lambda_0,
        memory_uniforms[None, :],
    )

    if args.gpu_ids:
        if len(tau_plus_values) == 1:
            raise ValueError("Multi-GPU benchmarking requires --tau-plus-values.")
        import cupy as cp

        backends = []
        for gpu_id in args.gpu_ids:
            properties = cp.cuda.runtime.getDeviceProperties(gpu_id)
            total_memory = int(properties["totalGlobalMem"])
            chunk_size = 512 if total_memory >= 60 * 2**30 else 256
            backends.append(NvidiaBackend(gpu_id=gpu_id, chunk_size=chunk_size))
        started = perf_counter()
        _, details = execute_nvidia_rate_sweep_sharded(
            backends=backends,
            program=program,
            windows=windows,
            reset_rate_matrix=reset_rate_matrix,
            execution=ExecutionSpec(retain_device_array=False),
        )
        elapsed = perf_counter() - started
        print(
            json.dumps(
                {
                    "gpu_ids": args.gpu_ids,
                    "chunk_sizes": [backend.chunk_size for backend in backends],
                    "n": args.n,
                    "R": args.reservoirs,
                    "w": args.window_length,
                    "N": args.windows,
                    "K": program.observables.size,
                    "rate_count": len(tau_plus_values),
                    "dtype": estimator.precision.value,
                    "elapsed_seconds": elapsed,
                    "windows_per_wall_second": args.windows / elapsed,
                    "reservoir_windows_per_wall_second": (
                        args.windows * args.reservoirs / elapsed
                    ),
                    "shards": details["shards"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    for chunk_size in args.chunk_sizes:
        backend = NvidiaBackend(gpu_id=args.gpu_id, chunk_size=chunk_size)
        for trial in range(args.trials):
            started = perf_counter()
            execution = ExecutionSpec(
                chunk_size=chunk_size,
                retain_device_array=True,
            )
            if len(tau_plus_values) == 1:
                result = estimator.estimate(program, windows, backend, execution)
            else:
                result = backend.execute_exact_rate_sweep(
                    program,
                    windows,
                    reset_rate_matrix,
                    execution,
                )
            elapsed = perf_counter() - started
            details = result.execution_metadata.details
            gpu_seconds = float(details["timings"]["gpu_execution_seconds"])
            print(
                json.dumps(
                    {
                        "trial": trial,
                        "gpu_id": args.gpu_id,
                        "n": args.n,
                        "R": args.reservoirs,
                        "w": args.window_length,
                        "N": args.windows,
                        "K": program.observables.size,
                        "rate_count": len(tau_plus_values),
                        "chunk_size": chunk_size,
                        "dtype": estimator.precision.value,
                        "elapsed_seconds": elapsed,
                        "gpu_seconds": gpu_seconds,
                        "windows_per_gpu_second": args.windows / gpu_seconds,
                        "reservoir_windows_per_gpu_second": (
                            args.windows * args.reservoirs / gpu_seconds
                        ),
                        "gpu_peak_increment_bytes": details["resources"][
                            "gpu_peak_increment_bytes"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if args.profile_phases:
            print(
                json.dumps(
                    {
                        "chunk_size": chunk_size,
                        "gpu_id": args.gpu_id,
                        **_profile_phases(
                            program=program,
                            windows=windows,
                            gpu_id=args.gpu_id,
                            chunk_size=chunk_size,
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
