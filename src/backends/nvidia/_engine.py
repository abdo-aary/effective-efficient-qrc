"""Numerical bridge to the existing optimized CuPy reservoir kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.backends._legacy import legacy_config
from src.core.factories import cycle_matchings
from src.core.program import BalancedReservoirParameters, QuaRKProgram


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


@dataclass(frozen=True)
class BalancedNumericProgramLayout:
    z_cols: np.ndarray
    local_axis_cols: np.ndarray
    local_angle_cols: np.ndarray
    edge_axis_left_cols: np.ndarray
    edge_axis_right_cols: np.ndarray
    edge_angle_cols: np.ndarray
    matching_order_cols: np.ndarray
    lam_col: int
    matchings: tuple[tuple[int, ...], ...]

    @property
    def w(self) -> int:
        return int(self.z_cols.shape[0])


def pack_program(
    program: QuaRKProgram,
    windows: np.ndarray,
) -> tuple[np.ndarray, NumericProgramLayout]:
    if isinstance(program.reservoirs, BalancedReservoirParameters):
        return _pack_balanced_program(program, windows)
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


def _pack_balanced_program(
    program: QuaRKProgram,
    windows: np.ndarray,
) -> tuple[np.ndarray, BalancedNumericProgramLayout]:
    parameters = program.reservoirs
    if not isinstance(parameters, BalancedReservoirParameters):
        raise TypeError("Expected balanced reservoir parameters.")
    projected = np.asarray(windows, dtype=float) @ program.projection.matrix
    N, w, n = projected.shape
    R = program.num_reservoirs
    E = len(program.topology.edges)
    J = parameters.matching_orders.shape[1]
    cursor = 0

    z_cols = np.arange(cursor, cursor + w * n, dtype=int).reshape(w, n)
    cursor += w * n
    local_axis_cols = np.arange(cursor, cursor + n * 3, dtype=int).reshape(n, 3)
    cursor += n * 3
    local_angle_cols = np.arange(cursor, cursor + n, dtype=int)
    cursor += n
    edge_axis_left_cols = np.arange(cursor, cursor + E * 3, dtype=int).reshape(E, 3)
    cursor += E * 3
    edge_axis_right_cols = np.arange(cursor, cursor + E * 3, dtype=int).reshape(E, 3)
    cursor += E * 3
    edge_angle_cols = np.arange(cursor, cursor + E, dtype=int)
    cursor += E
    matching_order_cols = np.arange(cursor, cursor + J, dtype=int)
    cursor += J
    lam_col = cursor

    injection = projected.reshape(N, w * n)
    reservoir = np.concatenate(
        [
            parameters.local_axes.reshape(R, n * 3),
            parameters.local_angles,
            parameters.edge_axes_left.reshape(R, E * 3),
            parameters.edge_axes_right.reshape(R, E * 3),
            parameters.edge_angles,
            parameters.matching_orders,
            parameters.reset_rates[:, None],
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
    return values.reshape(N * R, values.shape[-1]), BalancedNumericProgramLayout(
        z_cols=z_cols,
        local_axis_cols=local_axis_cols,
        local_angle_cols=local_angle_cols,
        edge_axis_left_cols=edge_axis_left_cols,
        edge_axis_right_cols=edge_axis_right_cols,
        edge_angle_cols=edge_angle_cols,
        matching_order_cols=matching_order_cols,
        lam_col=lam_col,
        matchings=cycle_matchings(program.topology),
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
    if isinstance(program.reservoirs, BalancedReservoirParameters):
        from src.backends.nvidia.balanced_runner import BalancedReservoirChannelRunner

        runner_type = BalancedReservoirChannelRunner
    else:
        from src.backends.nvidia.legacy_runner import ExactReservoirChannelRunner

        runner_type = ExactReservoirChannelRunner

    return runner_type(
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
