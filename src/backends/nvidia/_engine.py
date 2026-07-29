"""Numerical bridge to the existing optimized CuPy reservoir kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.backends._legacy import legacy_config
from src.core.program import QuaRKProgram


@dataclass(frozen=True)
class NumericProgramLayout:
    z_cols: np.ndarray
    j_cols: np.ndarray
    hx_cols: np.ndarray
    hz_cols: np.ndarray
    lam_col: int

    @property
    def w(self) -> int:
        return int(self.z_cols.shape[0])


def pack_program(
    program: QuaRKProgram,
    windows: np.ndarray,
) -> tuple[np.ndarray, NumericProgramLayout]:
    projected = np.asarray(windows, dtype=float) @ program.projection.matrix
    N = int(projected.shape[0])
    R = program.num_reservoirs
    n = program.num_qubits
    w = program.window_length
    E = len(program.topology.edges)

    z_cols = np.arange(w * n, dtype=int).reshape(w, n)
    start = w * n
    j_cols = np.arange(start, start + E, dtype=int)
    start += E
    hx_cols = np.arange(start, start + n, dtype=int)
    start += n
    hz_cols = np.arange(start, start + n, dtype=int)
    start += n
    lam_col = start

    injection = projected.reshape(N, w * n)
    reservoir = np.concatenate(
        [
            program.reservoirs.zz,
            program.reservoirs.x_fields,
            program.reservoirs.z_fields,
            program.reservoirs.reset_rates[:, None],
        ],
        axis=1,
    )
    values = np.concatenate(
        [
            np.broadcast_to(injection[:, None, :], (N, R, injection.shape[1])),
            np.broadcast_to(reservoir[None, :, :], (N, R, reservoir.shape[1])),
        ],
        axis=2,
    )
    return values.reshape(N * R, values.shape[-1]), NumericProgramLayout(
        z_cols=z_cols,
        j_cols=j_cols,
        hx_cols=hx_cols,
        hz_cols=hz_cols,
        lam_col=lam_col,
    )


def make_runner(
    program: QuaRKProgram,
    *,
    state_dtype: str,
    chunk_size: int,
    gpu_id: int | None,
    output_backend: str,
    output_kind: str,
):
    # Kept behind this boundary while the already-tested gate kernels are
    # The engine keeps the proven numerical recurrence behind the typed adapter.
    from src.backends.nvidia.legacy_runner import ExactReservoirChannelRunner

    return ExactReservoirChannelRunner(
        legacy_config(program),
        state_dtype=state_dtype,
        chunk_size=chunk_size,
        engine="cupy",
        gpu_id=gpu_id,
        output_backend=output_backend,
        output_kind=output_kind,
        weight_atol=0.0,
        max_history=None,
        angle_scale=program.angle_scale,
    )
