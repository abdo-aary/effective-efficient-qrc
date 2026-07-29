"""Direct GPU sampling of local-Pauli classical shadows."""

from __future__ import annotations

import numpy as np

from src.backends.nvidia._engine import make_runner, pack_program
from src.core.program import QuaRKProgram
from src.core.requests import ExecutionSpec
from src.estimators.csmom import CSMoMFeatureEstimator, ShadowSnapshots


def _apply_variable_measurement_rotations(cp, states, bases, gate_cache) -> None:
    inv_sqrt_two = 1.0 / np.sqrt(2.0)
    for q in range(int(bases.shape[1])):
        i0 = gate_cache.single_i0[q]
        i1 = gate_cache.single_i1[q]
        for basis in (0, 1):
            selected = cp.nonzero(bases[:, q] == basis)[0]
            if int(selected.size) == 0:
                continue
            a0 = states[selected[:, None], i0[None, :]].copy()
            a1 = states[selected[:, None], i1[None, :]].copy()
            if basis == 0:  # H
                states[selected[:, None], i0[None, :]] = (a0 + a1) * inv_sqrt_two
                states[selected[:, None], i1[None, :]] = (a0 - a1) * inv_sqrt_two
            else:  # H Sdg
                states[selected[:, None], i0[None, :]] = (a0 - 1j * a1) * inv_sqrt_two
                states[selected[:, None], i1[None, :]] = (a0 + 1j * a1) * inv_sqrt_two


def sample_csmom_gpu(
    program: QuaRKProgram,
    windows: np.ndarray,
    estimator: CSMoMFeatureEstimator,
    execution: ExecutionSpec,
    *,
    state_dtype: str,
    gpu_id: int | None,
    default_chunk_size: int,
) -> ShadowSnapshots:
    try:
        import cupy as cp
    except ImportError as exc:
        from src.core.exceptions import BackendDependencyError

        raise BackendDependencyError(
            "NVIDIA CSMoM execution requires CuPy built for the local CUDA runtime."
        ) from exc

    flat, numeric_layout = pack_program(program, windows)
    # The optimized runner's layout is structural; duck typing is intentional.
    runner = make_runner(
        program,
        state_dtype=state_dtype,
        chunk_size=execution.chunk_size or default_chunk_size,
        gpu_id=gpu_id,
        output_backend="cupy",
        output_kind="expectation",
    )
    plus = runner._plus_state(program.num_qubits).astype(state_dtype, copy=False)
    B = flat.shape[0]
    S = estimator.snapshots
    n = program.num_qubits
    basis_rng = execution.seeds.generator("shadow_bases")
    trajectory_rng = execution.seeds.generator("reset_trajectories")
    outcome_rng = execution.seeds.generator("measurement_outcomes")
    bases = basis_rng.integers(0, 3, size=(B, S, n), dtype=np.uint8)
    outcomes = np.empty((B, S, n), dtype=np.int8)

    step = int(execution.chunk_size or default_chunk_size)
    device = cp.cuda.Device(gpu_id) if gpu_id is not None else None
    context = device if device is not None else cp.cuda.Device()
    with context:
        plus_gpu = cp.asarray(plus, dtype=cp.dtype(state_dtype))
        for offset in range(0, B, step):
            end = min(offset + step, B)
            count = end - offset
            rows_gpu = cp.asarray(flat[offset:end], dtype=cp.float64)
            states, weights = runner._run_chunk_cupy_ensemble_device(
                rows=rows_gpu,
                layout=numeric_layout,
                angle_positioning_name=program.angle_map,
                plus_state=plus_gpu,
            )

            branch_uniforms = cp.asarray(
                trajectory_rng.random((count, S)), dtype=cp.float64
            )
            cumulative_weights = cp.cumsum(weights, axis=1)
            branch_indices = cp.sum(
                branch_uniforms[:, :, None] > cumulative_weights[:, None, :],
                axis=2,
            )
            branch_indices = cp.minimum(branch_indices, weights.shape[1] - 1)
            batch_indices = cp.arange(count, dtype=cp.int64)[:, None]
            sampled_states = states[batch_indices, branch_indices].reshape(
                count * S, 1 << n
            )
            sampled_bases = cp.asarray(bases[offset:end].reshape(count * S, n))
            _apply_variable_measurement_rotations(
                cp,
                sampled_states,
                sampled_bases,
                runner._get_cupy_gate_cache(),
            )

            probabilities = cp.abs(sampled_states) ** 2
            probabilities /= cp.sum(probabilities, axis=1, keepdims=True)
            cumulative = cp.cumsum(probabilities, axis=1)
            measurement_uniforms = cp.asarray(
                outcome_rng.random(count * S), dtype=cp.float64
            )
            indices = cp.sum(
                measurement_uniforms[:, None] > cumulative,
                axis=1,
            )
            indices = cp.minimum(indices, (1 << n) - 1)
            signs = cp.empty((count * S, n), dtype=cp.int8)
            for q in range(n):
                signs[:, q] = 1 - 2 * ((indices >> q) & 1).astype(cp.int8)
            outcomes[offset:end] = cp.asnumpy(signs.reshape(count, S, n))
        cp.cuda.get_current_stream().synchronize()

    N = int(windows.shape[0])
    R = program.num_reservoirs
    return ShadowSnapshots(
        bases=bases.reshape(N, R, S, n),
        outcomes=outcomes.reshape(N, R, S, n),
    )
