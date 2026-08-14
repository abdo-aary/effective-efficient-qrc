"""Untruncated CuPy weighted-ensemble runner for balanced random mixers."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.backends.nvidia.legacy_runner import ExactReservoirChannelRunner


class BalancedReservoirChannelRunner(ExactReservoirChannelRunner):
    """Apply arbitrary-axis local and edge-product gates exactly on CuPy."""

    def _run_chunk_cupy_ensemble_device(
        self,
        *,
        rows: Any,
        layout: Any,
        angle_positioning_name: str,
        plus_state: Any,
    ) -> tuple[Any, Any]:
        cp = self._cupy or self._import_cupy()
        B = int(rows.shape[0])
        n = int(self.qrc_cfg.num_qubits)
        dim = 1 << n
        dtype = cp.dtype(self.state_dtype.name)
        history_cap = int(layout.w + 1)
        if self.max_history is not None:
            history_cap = min(history_cap, int(self.max_history))

        states = cp.zeros((B, history_cap, dim), dtype=dtype)
        weights = cp.zeros((B, history_cap), dtype=cp.float64)
        states[:, 0, :] = plus_state[None, :]
        weights[:, 0] = 1.0

        lam = rows[:, layout.lam_col]
        local_axes = rows[:, layout.local_axis_cols]
        local_angles = rows[:, layout.local_angle_cols]
        edge_axes_left = rows[:, layout.edge_axis_left_cols]
        edge_axes_right = rows[:, layout.edge_axis_right_cols]
        edge_angles = rows[:, layout.edge_angle_cols]
        matching_orders = rows[:, layout.matching_order_cols].astype(cp.int64)

        active_count = 1
        overwrite_slot = 0
        for step in range(layout.w):
            active_states = states[:, :active_count, :]
            theta = self._angle_positioning_cupy(
                rows[:, layout.z_cols[step]], angle_positioning_name
            )
            for qubit in range(n):
                self._apply_ry_cupy(active_states, qubit, theta[:, qubit])
            self._apply_balanced_mixer_cupy(
                active_states,
                local_axes=local_axes,
                local_angles=local_angles,
                edge_axes_left=edge_axes_left,
                edge_axes_right=edge_axes_right,
                edge_angles=edge_angles,
                matching_orders=matching_orders,
                matchings=layout.matchings,
            )

            weights[:, :active_count] *= lam[:, None]
            if active_count < history_cap:
                reset_slot = active_count
                active_count += 1
            else:
                reset_slot = overwrite_slot
                overwrite_slot = (overwrite_slot + 1) % history_cap
            states[:, reset_slot, :] = plus_state[None, :]
            weights[:, reset_slot] = 1.0 - lam

        return states[:, :active_count, :], weights[:, :active_count]

    def _apply_balanced_mixer_cupy(
        self,
        states: Any,
        *,
        local_axes: Any,
        local_angles: Any,
        edge_axes_left: Any,
        edge_axes_right: Any,
        edge_angles: Any,
        matching_orders: Any,
        matchings: tuple[tuple[int, ...], ...],
    ) -> None:
        cp = self._cupy or self._import_cupy()
        n = int(self.qrc_cfg.num_qubits)
        for qubit in range(n):
            self._apply_axis_rotation_cupy(
                states,
                qubit,
                local_axes[:, qubit, :],
                local_angles[:, qubit],
            )

        for layer_position in range(len(matchings)):
            selected = matching_orders[:, layer_position]
            for matching_index, edge_indices in enumerate(matchings):
                active = selected == matching_index
                for edge_index in edge_indices:
                    theta = cp.where(active, edge_angles[:, edge_index], 0.0)
                    left, right = self.qrc_cfg.topology.edges[edge_index]
                    self._apply_axis_product_rotation_cupy(
                        states,
                        left,
                        right,
                        edge_axes_left[:, edge_index, :],
                        edge_axes_right[:, edge_index, :],
                        theta,
                    )

    def _apply_axis_rotation_cupy(
        self,
        states: Any,
        qubit: int,
        axes: Any,
        theta: Any,
    ) -> None:
        cp = self._cupy or self._import_cupy()
        cache = self._get_cupy_gate_cache()
        i0 = cache.single_i0[int(qubit)]
        i1 = cache.single_i1[int(qubit)]
        a0 = states[:, :, i0].copy()
        a1 = states[:, :, i1].copy()
        c = cp.cos(theta / 2.0)[:, None, None]
        s = cp.sin(theta / 2.0)[:, None, None]
        ax = axes[:, 0, None, None]
        ay = axes[:, 1, None, None]
        az = axes[:, 2, None, None]
        states[:, :, i0] = (c - 1j * s * az) * a0 + (-s * ay - 1j * s * ax) * a1
        states[:, :, i1] = (s * ay - 1j * s * ax) * a0 + (c + 1j * s * az) * a1

    def _apply_axis_product_rotation_cupy(
        self,
        states: Any,
        left: int,
        right: int,
        left_axes: Any,
        right_axes: Any,
        theta: Any,
    ) -> None:
        cp = self._cupy or self._import_cupy()
        indices = self._edge_indices(int(left), int(right), int(states.shape[-1]))
        index_arrays = tuple(cp.asarray(index, dtype=cp.int64) for index in indices)
        amplitudes = cp.stack(
            [states[:, :, index] for index in index_arrays],
            axis=-1,
        )
        left_matrix = self._axis_matrices_cupy(left_axes)
        right_matrix = self._axis_matrices_cupy(right_axes)
        generator = cp.einsum(
            "bij,bkl->bikjl", left_matrix, right_matrix, optimize=True
        ).reshape((-1, 4, 4))
        identity = cp.eye(4, dtype=states.dtype)[None, :, :]
        c = cp.cos(theta / 2.0)[:, None, None]
        s = cp.sin(theta / 2.0)[:, None, None]
        unitary = c * identity - 1j * s * generator
        transformed = cp.einsum(
            "bij,bhqj->bhqi", unitary, amplitudes, optimize=True
        )
        for slot, index in enumerate(index_arrays):
            states[:, :, index] = transformed[..., slot]

    def _axis_matrices_cupy(self, axes: Any) -> Any:
        cp = self._cupy or self._import_cupy()
        matrices = cp.empty((axes.shape[0], 2, 2), dtype=cp.dtype(self.state_dtype.name))
        matrices[:, 0, 0] = axes[:, 2]
        matrices[:, 0, 1] = axes[:, 0] - 1j * axes[:, 1]
        matrices[:, 1, 0] = axes[:, 0] + 1j * axes[:, 1]
        matrices[:, 1, 1] = -axes[:, 2]
        return matrices

    @staticmethod
    def _edge_indices(left: int, right: int, dim: int) -> tuple[np.ndarray, ...]:
        if left == right:
            raise ValueError("An edge must join distinct qubits.")
        basis = np.arange(dim, dtype=np.int64)
        mask = (1 << left) | (1 << right)
        base = basis[(basis & mask) == 0]
        return (
            base,
            base | (1 << right),
            base | (1 << left),
            base | mask,
        )
