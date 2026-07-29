"""Local timing helper for the paper-equivalent reservoir-channel runner.

This is intentionally not a CI test: it prints wall-clock timings and numerical
differences so we can compare the optimized batched engine against the reference
engine and, optionally, Aer density-matrix simulation.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy as np

from src.backends.aer.circuits import CircuitFactory
from src.core.legacy_config import RingQRConfig
from src.core.seeds import SeedBundle
from src.backends.qiskit_utils import angle_positioning_tanh, generate_k_local_paulis
from src.backends.aer.legacy_runner import ExactAerCircuitsRunner, ExactExpectationResults, ExactResults
from src.features.legacy_retrievers import ExactFeatureMapsRetriever
from src.backends.nvidia.legacy_runner import (
    ExactReservoirChannelRunner,
    TruncatedReservoirChannelRunner,
)


def _make_pubs(*, N: int, w: int, d: int, n: int, R: int, seed: int):
    cfg = RingQRConfig(input_dim=d, num_qubits=n, seed=seed)
    X = np.random.default_rng(seed).uniform(-0.8, 0.8, size=(N, w, d))
    pubs = CircuitFactory.create_pubs_dataset_reservoirs_IsingRingSWAP(
        qrc_cfg=cfg,
        angle_positioning=angle_positioning_tanh,
        X=X,
        num_reservoirs=R,
        lam_0=0.1,
        seed=seed,
        eps=1e-8,
    )
    return cfg, pubs


def _result_shape(result: ExactResults | ExactExpectationResults) -> tuple[int, ...]:
    if isinstance(result, ExactExpectationResults):
        return tuple(result.expectations.shape)
    return tuple(result.states.shape)


def _time(label: str, fn: Callable[[], ExactResults | ExactExpectationResults]) -> tuple[ExactResults | ExactExpectationResults, float]:
    t0 = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t0
    print(f"{label:>18s}: {dt:8.4f}s  output={_result_shape(result)}", flush=True)
    return result, dt


def _compare_pair(name: str, a: ExactResults, b: ExactResults) -> None:
    max_abs = float(np.max(np.abs(a.states - b.states)))
    print(f"{name:>18s}: max_abs_diff={max_abs:.3e}", flush=True)


def _run_case(
    *,
    N: int,
    w: int,
    d: int,
    n: int,
    R: int,
    seed: int,
    chunk_size: int,
    include_reference: bool,
    include_aer: bool,
    aer_device: str,
    direct_only: bool,
    max_history: int | None,
) -> None:
    print(
        f"\nCase: N={N}, R={R}, w={w}, d={d}, n={n}, chunk_size={chunk_size}, max_history={max_history}",
        flush=True,
    )
    cfg, pubs = _make_pubs(N=N, w=w, d=d, n=n, R=R, seed=seed)
    observables = generate_k_local_paulis(locality=2, num_qubits=n)
    direct_runner_cls = (
        TruncatedReservoirChannelRunner
        if max_history is not None
        else ExactReservoirChannelRunner
    )

    if direct_only:
        _time(
            "cupy-direct",
            lambda: direct_runner_cls(
                cfg,
                engine="cupy",
                chunk_size=chunk_size,
                gpu_id=0,
                output_backend="cupy",
                output_kind="expectation",
                max_history=max_history,
            ).run_pubs(
                pubs=pubs,
                angle_positioning_name="tanh",
                observables=observables,
            ),
        )
        return

    reference = None
    if include_reference:
        reference, _ = _time(
            "reference",
            lambda: ExactReservoirChannelRunner(cfg, engine="reference", chunk_size=chunk_size).run_pubs(
                pubs=pubs,
                angle_positioning_name="tanh",
            ),
        )
    batched, _ = _time(
        "batched",
        lambda: ExactReservoirChannelRunner(cfg, engine="batched", chunk_size=chunk_size).run_pubs(
            pubs=pubs,
            angle_positioning_name="tanh",
        ),
    )
    if reference is not None:
        _compare_pair("batched-reference", batched, reference)

    try:
        cupy, _ = _time(
            "cupy",
            lambda: ExactReservoirChannelRunner(cfg, engine="cupy", chunk_size=chunk_size, gpu_id=0).run_pubs(
                pubs=pubs,
                angle_positioning_name="tanh",
            ),
        )
    except ImportError as exc:
        print(f"{'cupy':>18s}: skipped ({exc})", flush=True)
    else:
        _compare_pair("cupy-batched", cupy, batched)
        direct, _ = _time(
            "cupy-direct",
            lambda: direct_runner_cls(
                cfg,
                engine="cupy",
                chunk_size=chunk_size,
                gpu_id=0,
                output_backend="cupy",
                output_kind="expectation",
                max_history=max_history,
            ).run_pubs(
                pubs=pubs,
                angle_positioning_name="tanh",
                observables=observables,
            ),
        )
        direct_phi = ExactFeatureMapsRetriever(cfg, observables, backend="cupy").get_feature_maps(direct)
        density_phi = ExactFeatureMapsRetriever(cfg, observables, backend="cupy").get_feature_maps(cupy)
        max_abs = float(np.max(np.abs(direct_phi.get() - density_phi.get())))
        print(f"{'direct-cupy':>18s}: feature_max_abs_diff={max_abs:.3e}", flush=True)

    if include_aer:
        aer, _ = _time(
            f"aer-{aer_device.lower()}",
            lambda: ExactAerCircuitsRunner(cfg).run_pubs(
                pubs=pubs,
                device=aer_device,
                seed_simulator=0,
                optimization_level=0,
                max_parallel_threads=0,
                max_parallel_experiments=8,
                max_parallel_shots=1,
            ),
        )
        _compare_pair("batched-aer", batched, aer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-N", type=int, default=16)
    parser.add_argument("--large-N", type=int, default=128)
    parser.add_argument("--w", type=int, default=25)
    parser.add_argument("--d", "--input-dim", dest="d", type=int, default=3)
    parser.add_argument("--n", "--num-qubits", dest="n", type=int, default=5)
    parser.add_argument("--R", "--num-reservoirs", dest="R", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--skip-aer", action="store_true")
    parser.add_argument("--aer-device", default="GPU", choices=("CPU", "GPU"))
    parser.add_argument("--direct-only", action="store_true")
    parser.add_argument("--max-history", type=int, default=None)
    args = parser.parse_args()
    replicate_sequences = SeedBundle.from_root(args.seed).sequence("replicate").spawn(2)
    replicate_seeds = [
        int(sequence.generate_state(1, dtype=np.uint32)[0])
        for sequence in replicate_sequences
    ]

    _run_case(
        N=args.small_N,
        w=args.w,
        d=args.d,
        n=args.n,
        R=args.R,
        seed=replicate_seeds[0],
        chunk_size=args.chunk_size,
        include_reference=not args.skip_reference,
        include_aer=not args.skip_aer,
        aer_device=args.aer_device,
        direct_only=args.direct_only,
        max_history=args.max_history,
    )
    if args.large_N > 0:
        _run_case(
            N=args.large_N,
            w=args.w,
            d=args.d,
            n=args.n,
            R=args.R,
            seed=replicate_seeds[1],
            chunk_size=args.chunk_size,
            include_reference=not args.skip_reference,
            include_aer=False,
            aer_device=args.aer_device,
            direct_only=args.direct_only,
            max_history=args.max_history,
        )


if __name__ == "__main__":
    main()
