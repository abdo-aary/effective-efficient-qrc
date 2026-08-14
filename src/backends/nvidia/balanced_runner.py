"""Untruncated CuPy weighted-ensemble runner for balanced random mixers."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.backends.nvidia.legacy_runner import ExactReservoirChannelRunner


class BalancedReservoirChannelRunner(ExactReservoirChannelRunner):
    """Apply arbitrary-axis local and edge-product gates exactly on CuPy."""

    def run_multi_rate_expectations(
        self,
        *,
        flat: np.ndarray,
        layout: Any,
        angle_positioning_name: str,
        plus_state: np.ndarray,
        chunk_size: int,
        pauli_masks: list[tuple[int, int, int]],
        reset_rates: np.ndarray,
    ) -> Any:
        """Evolve each pure history once and evaluate every reset-rate weighting."""

        cp = self._cupy or self._import_cupy()
        rates = np.asarray(reset_rates, dtype=np.float64)
        if rates.ndim != 2 or rates.shape[0] != flat.shape[0]:
            raise ValueError("reset_rates must have shape (flat_rows, rate_count).")
        if rates.shape[1] < 1 or np.any((rates <= 0.0) | (rates >= 1.0)):
            raise ValueError("Every reset rate must lie strictly in (0, 1).")
        if self.max_history is not None:
            raise ValueError("Multi-rate exact execution forbids history truncation.")

        device = cp.cuda.Device(self.gpu_id) if self.gpu_id is not None else cp.cuda.Device()
        with device:
            out = cp.empty(
                (flat.shape[0], rates.shape[1], len(pauli_masks)),
                dtype=cp.float64,
            )
            plus_gpu = cp.asarray(plus_state, dtype=cp.dtype(self.state_dtype.name))
            for offset in range(0, flat.shape[0], int(chunk_size)):
                end = min(offset + int(chunk_size), flat.shape[0])
                rows_gpu = cp.asarray(flat[offset:end], dtype=cp.float64)
                states, _ = self._run_chunk_cupy_ensemble_device(
                    rows=rows_gpu,
                    layout=layout,
                    angle_positioning_name=angle_positioning_name,
                    plus_state=plus_gpu,
                )
                out[offset:end] = self._multi_rate_expectations_from_histories(
                    states=states,
                    reset_rates=cp.asarray(rates[offset:end], dtype=cp.float64),
                    pauli_masks=pauli_masks,
                    n=int(self.qrc_cfg.num_qubits),
                )
            cp.cuda.get_current_stream().synchronize()
            if self.output_backend == "cupy":
                return out
            return cp.asnumpy(out)

    def _multi_rate_expectations_from_histories(
        self,
        *,
        states: Any,
        reset_rates: Any,
        pauli_masks: list[tuple[int, int, int]],
        n: int,
    ) -> Any:
        cp = self._cupy or self._import_cupy()
        B, history_count, dim = (int(value) for value in states.shape)
        if dim != 1 << int(n):
            raise ValueError("State dimension does not match num_qubits.")

        rates = reset_rates[:, :, None]
        reset_exponents = cp.arange(
            history_count - 2,
            -1,
            -1,
            dtype=cp.float64,
        )
        initial = rates ** float(history_count - 1)
        reset = (1.0 - rates) * rates ** reset_exponents[None, None, :]
        weights = cp.concatenate((initial, reset), axis=2)
        rows = cp.arange(dim, dtype=cp.uint32)
        out = cp.empty((B, reset_rates.shape[1], len(pauli_masks)), dtype=cp.float64)

        for k, (xmask, zmask, ny) in enumerate(pauli_masks):
            cols = rows ^ cp.uint32(xmask)
            phase = self._pauli_phase_vector_cupy(zmask=zmask, ny=ny, dim=dim)
            exp_by_history = cp.sum(
                states
                * cp.conj(states[:, :, cols])
                * phase[None, None, :],
                axis=2,
            ).real
            out[:, :, k] = cp.sum(
                weights * exp_by_history[:, None, :],
                axis=2,
            )
        return out

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
        local_unitaries = self._local_unitaries_cupy(local_axes, local_angles)
        edge_unitaries = self._edge_unitaries_cupy(
            edge_axes_left,
            edge_axes_right,
            edge_angles,
        )

        active_count = 1
        overwrite_slot = 0
        for step in range(layout.w):
            active_states = states[:, :active_count, :]
            theta = self._angle_positioning_cupy(
                rows[:, layout.z_cols[step]], angle_positioning_name
            )
            self._apply_injection_and_local_cupy(
                active_states,
                theta=theta,
                local_unitaries=local_unitaries,
            )
            self._apply_precomputed_edge_layers_cupy(
                active_states,
                edge_unitaries=edge_unitaries,
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

    def _local_unitaries_cupy(self, axes: Any, angles: Any) -> Any:
        cp = self._cupy or self._import_cupy()
        B, n, _ = axes.shape
        generators = self._axis_matrices_cupy(axes.reshape(B * n, 3)).reshape(
            B, n, 2, 2
        )
        identity = cp.eye(2, dtype=generators.dtype)[None, None, :, :]
        c = cp.cos(angles / 2.0)[:, :, None, None]
        s = cp.sin(angles / 2.0)[:, :, None, None]
        return c * identity - 1j * s * generators

    def _edge_unitaries_cupy(
        self,
        left_axes: Any,
        right_axes: Any,
        angles: Any,
    ) -> Any:
        cp = self._cupy or self._import_cupy()
        B, edge_count, _ = left_axes.shape
        left = self._axis_matrices_cupy(
            left_axes.reshape(B * edge_count, 3)
        ).reshape(B, edge_count, 2, 2)
        right = self._axis_matrices_cupy(
            right_axes.reshape(B * edge_count, 3)
        ).reshape(B, edge_count, 2, 2)
        generators = cp.einsum(
            "beij,bekl->beikjl",
            left,
            right,
            optimize=True,
        ).reshape(B, edge_count, 4, 4)
        identity = cp.eye(4, dtype=generators.dtype)[None, None, :, :]
        c = cp.cos(angles / 2.0)[:, :, None, None]
        s = cp.sin(angles / 2.0)[:, :, None, None]
        return c * identity - 1j * s * generators

    def _apply_injection_and_local_cupy(
        self,
        states: Any,
        *,
        theta: Any,
        local_unitaries: Any,
    ) -> None:
        cp = self._cupy or self._import_cupy()
        cache = self._get_cupy_gate_cache()
        for qubit in range(int(self.qrc_cfg.num_qubits)):
            i0 = cache.single_i0[qubit]
            i1 = cache.single_i1[qubit]
            a0 = states[:, :, i0].copy()
            a1 = states[:, :, i1].copy()
            c = cp.cos(theta[:, qubit] / 2.0)
            s = cp.sin(theta[:, qubit] / 2.0)
            mixer = local_unitaries[:, qubit]
            u00 = mixer[:, 0, 0] * c + mixer[:, 0, 1] * s
            u01 = -mixer[:, 0, 0] * s + mixer[:, 0, 1] * c
            u10 = mixer[:, 1, 0] * c + mixer[:, 1, 1] * s
            u11 = -mixer[:, 1, 0] * s + mixer[:, 1, 1] * c
            states[:, :, i0] = u00[:, None, None] * a0 + u01[:, None, None] * a1
            states[:, :, i1] = u10[:, None, None] * a0 + u11[:, None, None] * a1

    def _apply_precomputed_edge_layers_cupy(
        self,
        states: Any,
        *,
        edge_unitaries: Any,
        matching_orders: Any,
        matchings: tuple[tuple[int, ...], ...],
    ) -> None:
        for layer_position in range(len(matchings)):
            selected = matching_orders[:, layer_position]
            for matching_index, edge_indices in enumerate(matchings):
                active_rows = (selected == matching_index).nonzero()[0]
                if int(active_rows.size) == 0:
                    continue
                for edge_index in edge_indices:
                    left, right = self.qrc_cfg.topology.edges[edge_index]
                    self._apply_precomputed_edge_cupy(
                        states,
                        left=int(left),
                        right=int(right),
                        unitary=edge_unitaries[:, edge_index],
                        active_rows=active_rows,
                    )

    def _apply_precomputed_edge_cupy(
        self,
        states: Any,
        *,
        left: int,
        right: int,
        unitary: Any,
        active_rows: Any,
    ) -> None:
        cp = self._cupy or self._import_cupy()
        indices = self._edge_indices(left, right, int(states.shape[-1]))
        basis_indices = tuple(cp.asarray(index, dtype=cp.int64) for index in indices)
        history = cp.arange(states.shape[1], dtype=cp.int64)[None, :, None]
        rows = active_rows[:, None, None]
        amplitudes = cp.stack(
            [
                states[rows, history, index[None, None, :]]
                for index in basis_indices
            ],
            axis=-1,
        )
        transformed = cp.einsum(
            "aij,ahqj->ahqi",
            unitary[active_rows],
            amplitudes,
            optimize=True,
        )
        for slot, index in enumerate(basis_indices):
            states[rows, history, index[None, None, :]] = transformed[..., slot]

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
