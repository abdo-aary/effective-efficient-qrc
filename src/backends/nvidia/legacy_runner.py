r"""Optimized NumPy/CuPy reduced reset-channel kernels.

The paper circuit implements the contraction channel through a SWAP dilation on
``2n + 1`` qubits, then saves the reduced reservoir density matrix. This runner
implements the mathematically equivalent reduced channel directly on the
``n``-qubit reservoir:

``rho <- U_t rho U_t^\dagger``
``rho <- lambda * rho + (1 - lambda) * |+><+|^{\otimes n}``

It returns the same :class:`ExactResults` shape as ``ExactAerCircuitsRunner`` so
the existing exact and c-shadow feature retrievers can be reused unchanged.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator, SparsePauliOp

from src.core.legacy_config import BaseQRConfig
from src.backends.aer.legacy_runner import BaseCircuitsRunner, ExactExpectationResults, ExactResults, PUB


_DEFAULT_ANGLE_SCALE = np.pi * (1.0 - 1e-6)


@dataclass(frozen=True)
class _SwapTemplateLayout:
    z_cols: np.ndarray
    j_cols: np.ndarray
    hx_cols: np.ndarray
    hz_cols: np.ndarray
    lam_col: int

    @property
    def w(self) -> int:
        return int(self.z_cols.shape[0])


@dataclass(frozen=True)
class _GateIndexCache:
    single_i0: tuple[np.ndarray, ...]
    single_i1: tuple[np.ndarray, ...]
    rzz_same: tuple[np.ndarray, ...]


class ExactReservoirChannelRunner(BaseCircuitsRunner):
    """Paper-equivalent reservoir-channel runner without Aer density matrices."""

    requires_angle_positioning_name = True
    requires_observables = True
    _SUPPORTED_ENGINES = frozenset({"batched", "reference", "cupy"})
    _SUPPORTED_OUTPUT_KINDS = frozenset({"density_matrix", "expectation"})

    def __init__(
        self,
        qrc_cfg: BaseQRConfig,
        *,
        state_dtype: str = "complex128",
        chunk_size: int = 2048,
        angle_scale: float = _DEFAULT_ANGLE_SCALE,
        weight_atol: float = 0.0,
        engine: str = "batched",
        gpu_id: int | None = None,
        output_backend: str = "numpy",
        output_kind: str = "density_matrix",
        max_history: int | None = None,
        _allow_approximate: bool = False,
    ) -> None:
        self.qrc_cfg = qrc_cfg
        self.state_dtype = np.dtype(state_dtype)
        if self.state_dtype not in (np.dtype("complex64"), np.dtype("complex128")):
            raise ValueError(f"state_dtype must be complex64 or complex128, got {state_dtype!r}.")
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
        self.angle_scale = float(angle_scale)
        self.weight_atol = float(weight_atol)
        self.engine = str(engine)
        if self.engine not in self._SUPPORTED_ENGINES:
            raise ValueError(f"engine must be one of {sorted(self._SUPPORTED_ENGINES)}, got {engine!r}.")
        if self.engine == "cupy" and self.weight_atol > 0.0:
            raise ValueError("engine='cupy' does not support weight_atol pruning; use weight_atol=0.0.")
        self.gpu_id = None if gpu_id is None else int(gpu_id)
        self.output_backend = str(output_backend)
        if self.output_backend not in {"numpy", "cupy"}:
            raise ValueError(f"output_backend must be 'numpy' or 'cupy', got {output_backend!r}.")
        if self.output_backend == "cupy" and self.engine != "cupy":
            raise ValueError("output_backend='cupy' requires engine='cupy'.")
        self.output_kind = str(output_kind)
        if self.output_kind not in self._SUPPORTED_OUTPUT_KINDS:
            raise ValueError(
                f"output_kind must be one of {sorted(self._SUPPORTED_OUTPUT_KINDS)}, got {output_kind!r}."
            )
        if self.output_kind == "expectation" and self.engine != "cupy":
            raise ValueError("output_kind='expectation' is supported only with engine='cupy' in v1.")
        self.max_history = None if max_history is None else int(max_history)
        if self.max_history is not None:
            if self.max_history < 1:
                raise ValueError(f"max_history must be positive when provided, got {max_history}.")
            if self.engine != "cupy" or self.output_kind != "expectation":
                raise ValueError(
                    "max_history truncation is supported only with engine='cupy' and output_kind='expectation'."
                )
        if not _allow_approximate and (
            self.weight_atol > 0.0 or self.max_history is not None
        ):
            raise ValueError(
                "ExactReservoirChannelRunner forbids branch pruning and history truncation. "
                "Use TruncatedReservoirChannelRunner for explicitly approximate execution."
            )
        self._cupy: Any | None = None
        self._cupy_gate_cache: _GateIndexCache | None = None
        if self.engine == "cupy":
            self._cupy = self._import_cupy()
        self._gate_cache = self._build_gate_index_cache(int(self.qrc_cfg.num_qubits))

    def run_pubs(
        self,
        pubs: list[PUB],
        *,
        angle_positioning_name: str,
        observables: Sequence[Operator | SparsePauliOp] | None = None,
        chunk_size: int | None = None,
        **_: Any,
    ) -> ExactResults | ExactExpectationResults:
        if len(pubs) != 1:
            raise ValueError("ExactReservoirChannelRunner supports only single template PUB mode.")
        qc, vals = pubs[0]
        vals = np.asarray(vals, dtype=float)
        if vals.ndim != 3:
            raise ValueError(f"Expected template vals shape (N,R,P), got {vals.shape}.")

        N, R, P = vals.shape
        layout = self._layout_from_metadata(qc)
        if len(qc.metadata["param_order"]) != P:
            raise ValueError(f"vals has {P} columns but metadata['param_order'] has {len(qc.metadata['param_order'])}.")

        n = int(self.qrc_cfg.num_qubits)
        dim = 1 << n
        flat = vals.reshape(N * R, P)
        step = int(chunk_size or self.chunk_size)

        if self.output_kind == "expectation":
            if observables is None:
                raise ValueError("output_kind='expectation' requires observables.")
            pauli_masks, labels = self._pauli_masks_from_observables(observables, n=n)
            expectations = self._run_flat_cupy_expectation_output(
                flat=flat,
                layout=layout,
                angle_positioning_name=angle_positioning_name,
                plus_state=self._plus_state(n).astype(self.state_dtype, copy=False),
                chunk_size=step,
                pauli_masks=pauli_masks,
            )
            return ExactExpectationResults(
                expectations=expectations.reshape(N, R, len(labels)),
                qrc_cfg=self.qrc_cfg,
                observable_labels=labels,
            )

        if self.engine == "cupy" and self.output_backend == "cupy":
            states_gpu = self._run_flat_cupy_output(
                flat=flat,
                layout=layout,
                angle_positioning_name=angle_positioning_name,
                plus_state=self._plus_state(n).astype(self.state_dtype, copy=False),
                chunk_size=step,
            )
            return ExactResults(states=states_gpu.reshape(N, R, dim, dim), qrc_cfg=self.qrc_cfg)

        states_flat = np.empty((N * R, dim, dim), dtype=self.state_dtype)

        plus = self._plus_state(n).astype(self.state_dtype, copy=False)
        for offset in range(0, flat.shape[0], step):
            end = min(offset + step, flat.shape[0])
            chunk = flat[offset:end]
            if self.engine == "cupy":
                states_flat[offset:end] = self._run_chunk_cupy(
                    rows=chunk,
                    layout=layout,
                    angle_positioning_name=angle_positioning_name,
                    plus_state=plus,
                )
            elif self.engine == "batched" and self.weight_atol <= 0.0:
                states_flat[offset:end] = self._run_chunk_batched(
                    rows=chunk,
                    layout=layout,
                    angle_positioning_name=angle_positioning_name,
                    plus_state=plus,
                )
            else:
                states_flat[offset:end] = self._run_chunk_reference(
                    rows=chunk,
                    layout=layout,
                    angle_positioning_name=angle_positioning_name,
                    plus_state=plus,
                )

        return ExactResults(states=states_flat.reshape(N, R, dim, dim), qrc_cfg=self.qrc_cfg)

    @staticmethod
    def _import_cupy() -> Any:
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "ExactReservoirChannelRunner(engine='cupy') requires CuPy. "
                "Install a CUDA 12 wheel with `conda run -n shaqr python -m pip install cupy-cuda12x` "
                "or install CuPy from conda-forge, then retry."
            ) from exc
        return cp

    def _layout_from_metadata(self, qc: QuantumCircuit) -> _SwapTemplateLayout:
        md = qc.metadata or {}
        required = ("param_order", "z_steps", "J", "h_x", "h_z", "lam")
        missing = [key for key in required if key not in md]
        if missing:
            raise ValueError(f"SWAP template metadata missing required keys: {missing}.")

        param_order = list(md["param_order"])
        col_by_param: dict[Parameter, int] = {p: idx for idx, p in enumerate(param_order)}

        n = int(self.qrc_cfg.num_qubits)
        z_steps = list(md["z_steps"])
        if not z_steps:
            raise ValueError("metadata['z_steps'] must be non-empty.")

        z_cols = np.empty((len(z_steps), n), dtype=int)
        for t, z_t in enumerate(z_steps):
            z_params = list(z_t)
            if len(z_params) != n:
                raise ValueError(f"z_steps[{t}] has length {len(z_params)} but expected n={n}.")
            z_cols[t] = [self._col_for_param(col_by_param, p, f"z_steps[{t}]") for p in z_params]

        j_cols = np.asarray([self._col_for_param(col_by_param, p, "J") for p in list(md["J"])], dtype=int)
        hx_cols = np.asarray([self._col_for_param(col_by_param, p, "h_x") for p in list(md["h_x"])], dtype=int)
        hz_cols = np.asarray([self._col_for_param(col_by_param, p, "h_z") for p in list(md["h_z"])], dtype=int)
        lam_col = self._col_for_param(col_by_param, md["lam"], "lam")

        if hx_cols.size != n or hz_cols.size != n:
            raise ValueError(f"Expected {n} h_x/h_z params, got {hx_cols.size}/{hz_cols.size}.")
        if j_cols.size != len(self.qrc_cfg.topology.edges):
            raise ValueError(
                f"Expected {len(self.qrc_cfg.topology.edges)} J params, got {j_cols.size}."
            )

        return _SwapTemplateLayout(z_cols=z_cols, j_cols=j_cols, hx_cols=hx_cols, hz_cols=hz_cols, lam_col=lam_col)

    @staticmethod
    def _col_for_param(col_by_param: dict[Parameter, int], param: Parameter, group: str) -> int:
        try:
            return int(col_by_param[param])
        except KeyError as exc:
            raise ValueError(
                f"Parameter {param!r} from metadata[{group!r}] "
                "is not in metadata['param_order']."
            ) from exc

    def _run_single_row(
        self,
        *,
        row: np.ndarray,
        layout: _SwapTemplateLayout,
        angle_positioning_name: str,
        plus_state: np.ndarray,
    ) -> np.ndarray:
        n = int(self.qrc_cfg.num_qubits)
        dim = 1 << n
        states = plus_state.reshape(1, dim).copy()
        weights = np.ones((1,), dtype=np.float64)

        lam = float(row[layout.lam_col])
        j_vals = row[layout.j_cols]
        hx_vals = row[layout.hx_cols]
        hz_vals = row[layout.hz_cols]

        for t in range(layout.w):
            theta = self._angle_positioning(row[layout.z_cols[t]], angle_positioning_name)
            self._apply_reservoir_unitary(states, theta=theta, j_vals=j_vals, hx_vals=hx_vals, hz_vals=hz_vals)

            if self.weight_atol > 0.0:
                keep = weights > self.weight_atol
                states = states[keep]
                weights = weights[keep]

            weights = lam * weights
            states = np.concatenate([states, plus_state.reshape(1, dim)], axis=0)
            weights = np.concatenate([weights, np.asarray([1.0 - lam], dtype=np.float64)])

        return self._density_from_ensemble(states, weights)

    def _run_chunk_reference(
        self,
        *,
        rows: np.ndarray,
        layout: _SwapTemplateLayout,
        angle_positioning_name: str,
        plus_state: np.ndarray,
    ) -> np.ndarray:
        n = int(self.qrc_cfg.num_qubits)
        dim = 1 << n
        out = np.empty((rows.shape[0], dim, dim), dtype=self.state_dtype)
        for idx, row in enumerate(rows):
            out[idx] = self._run_single_row(
                row=row,
                layout=layout,
                angle_positioning_name=angle_positioning_name,
                plus_state=plus_state,
            )
        return out

    def _run_chunk_batched(
        self,
        *,
        rows: np.ndarray,
        layout: _SwapTemplateLayout,
        angle_positioning_name: str,
        plus_state: np.ndarray,
    ) -> np.ndarray:
        B = int(rows.shape[0])
        n = int(self.qrc_cfg.num_qubits)
        dim = 1 << n

        states = np.zeros((B, layout.w + 1, dim), dtype=self.state_dtype)
        weights = np.zeros((B, layout.w + 1), dtype=np.float64)
        states[:, 0, :] = plus_state[None, :]
        weights[:, 0] = 1.0

        lam = rows[:, layout.lam_col].astype(np.float64, copy=False)
        j_vals = rows[:, layout.j_cols]
        hx_vals = rows[:, layout.hx_cols]
        hz_vals = rows[:, layout.hz_cols]

        active_count = 1
        for t in range(layout.w):
            theta = self._angle_positioning(rows[:, layout.z_cols[t]], angle_positioning_name)
            active_states = states[:, :active_count, :]
            self._apply_reservoir_unitary_batched(
                active_states,
                theta=theta,
                j_vals=j_vals,
                hx_vals=hx_vals,
                hz_vals=hz_vals,
            )

            weights[:, :active_count] *= lam[:, None]
            states[:, active_count, :] = plus_state[None, :]
            weights[:, active_count] = 1.0 - lam
            active_count += 1

        return self._density_from_ensemble_batched(states[:, :active_count, :], weights[:, :active_count])

    def _run_chunk_cupy(
        self,
        *,
        rows: np.ndarray,
        layout: _SwapTemplateLayout,
        angle_positioning_name: str,
        plus_state: np.ndarray,
    ) -> np.ndarray:
        cp = self._cupy or self._import_cupy()
        device = cp.cuda.Device(self.gpu_id) if self.gpu_id is not None else nullcontext()
        with device:
            rows_gpu = cp.asarray(rows, dtype=cp.float64)
            plus_gpu = cp.asarray(plus_state, dtype=cp.dtype(self.state_dtype.name))
            rho_gpu = self._run_chunk_cupy_device(
                rows=rows_gpu,
                layout=layout,
                angle_positioning_name=angle_positioning_name,
                plus_state=plus_gpu,
            )
            cp.cuda.get_current_stream().synchronize()
            return cp.asnumpy(rho_gpu).astype(self.state_dtype, copy=False)

    def _run_flat_cupy_output(
        self,
        *,
        flat: np.ndarray,
        layout: _SwapTemplateLayout,
        angle_positioning_name: str,
        plus_state: np.ndarray,
        chunk_size: int,
    ) -> Any:
        cp = self._cupy or self._import_cupy()
        device = cp.cuda.Device(self.gpu_id) if self.gpu_id is not None else nullcontext()
        with device:
            dtype = cp.dtype(self.state_dtype.name)
            n = int(self.qrc_cfg.num_qubits)
            dim = 1 << n
            out = cp.empty((flat.shape[0], dim, dim), dtype=dtype)
            plus_gpu = cp.asarray(plus_state, dtype=dtype)
            for offset in range(0, flat.shape[0], int(chunk_size)):
                end = min(offset + int(chunk_size), flat.shape[0])
                rows_gpu = cp.asarray(flat[offset:end], dtype=cp.float64)
                out[offset:end] = self._run_chunk_cupy_device(
                    rows=rows_gpu,
                    layout=layout,
                    angle_positioning_name=angle_positioning_name,
                    plus_state=plus_gpu,
                )
            cp.cuda.get_current_stream().synchronize()
            return out

    def _run_chunk_cupy_device(
        self,
        *,
        rows: Any,
        layout: _SwapTemplateLayout,
        angle_positioning_name: str,
        plus_state: Any,
    ) -> Any:
        states, weights = self._run_chunk_cupy_ensemble_device(
            rows=rows,
            layout=layout,
            angle_positioning_name=angle_positioning_name,
            plus_state=plus_state,
        )
        return self._density_from_ensemble_cupy(states, weights)

    def _run_chunk_cupy_ensemble_device(
        self,
        *,
        rows: Any,
        layout: _SwapTemplateLayout,
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
        j_vals = rows[:, layout.j_cols]
        hx_vals = rows[:, layout.hx_cols]
        hz_vals = rows[:, layout.hz_cols]

        active_count = 1
        overwrite_slot = 0
        for t in range(layout.w):
            theta = self._angle_positioning_cupy(rows[:, layout.z_cols[t]], angle_positioning_name)
            active_states = states[:, :active_count, :]
            self._apply_reservoir_unitary_cupy(
                active_states,
                theta=theta,
                j_vals=j_vals,
                hx_vals=hx_vals,
                hz_vals=hz_vals,
            )

            weights[:, :active_count] *= lam[:, None]
            if active_count < history_cap:
                reset_slot = active_count
                active_count += 1
            else:
                # Once the finite-memory buffer is full, overwrite the oldest
                # branch. Its omitted contribution is bounded by the geometric
                # tail mass of the contractive channel.
                reset_slot = overwrite_slot
                overwrite_slot = (overwrite_slot + 1) % history_cap
            states[:, reset_slot, :] = plus_state[None, :]
            weights[:, reset_slot] = 1.0 - lam

        return states[:, :active_count, :], weights[:, :active_count]

    def _run_flat_cupy_expectation_output(
        self,
        *,
        flat: np.ndarray,
        layout: _SwapTemplateLayout,
        angle_positioning_name: str,
        plus_state: np.ndarray,
        chunk_size: int,
        pauli_masks: list[tuple[int, int, int]],
    ) -> Any:
        cp = self._cupy or self._import_cupy()
        device = cp.cuda.Device(self.gpu_id) if self.gpu_id is not None else nullcontext()
        with device:
            n = int(self.qrc_cfg.num_qubits)
            K = len(pauli_masks)
            out = cp.empty((flat.shape[0], K), dtype=cp.float64)
            plus_gpu = cp.asarray(plus_state, dtype=cp.dtype(self.state_dtype.name))
            for offset in range(0, flat.shape[0], int(chunk_size)):
                end = min(offset + int(chunk_size), flat.shape[0])
                rows_gpu = cp.asarray(flat[offset:end], dtype=cp.float64)
                states, weights = self._run_chunk_cupy_ensemble_device(
                    rows=rows_gpu,
                    layout=layout,
                    angle_positioning_name=angle_positioning_name,
                    plus_state=plus_gpu,
                )
                out[offset:end] = self._expectations_from_ensemble_cupy(
                    states=states,
                    weights=weights,
                    pauli_masks=pauli_masks,
                    n=n,
                )
            cp.cuda.get_current_stream().synchronize()
            if self.output_backend == "cupy":
                return out
            return cp.asnumpy(out)

    def _expectations_from_ensemble_cupy(
        self,
        *,
        states: Any,
        weights: Any,
        pauli_masks: list[tuple[int, int, int]],
        n: int,
    ) -> Any:
        cp = self._cupy or self._import_cupy()
        B = int(states.shape[0])
        dim = 1 << int(n)
        rows = cp.arange(dim, dtype=cp.uint32)
        out = cp.empty((B, len(pauli_masks)), dtype=cp.float64)

        for k, (xmask, zmask, ny) in enumerate(pauli_masks):
            cols = rows ^ cp.uint32(xmask)
            phase = self._pauli_phase_vector_cupy(zmask=zmask, ny=ny, dim=dim)
            # Reduce every exact reset history in one GPU expression. This is
            # mathematically identical to the history loop but avoids O(K*w)
            # Python-dispatched kernels per chunk.
            exp_by_history = cp.sum(
                states
                * cp.conj(states[:, :, cols])
                * phase[None, None, :],
                axis=2,
            )
            out[:, k] = cp.sum(weights * exp_by_history, axis=1).real

        return out

    def _angle_positioning(self, z: np.ndarray, name: str) -> np.ndarray:
        if name == "linear":
            return self.angle_scale * z
        if name == "tanh":
            return self.angle_scale * np.tanh(z)
        raise ValueError(f"Unsupported angle_positioning_name={name!r}; expected 'linear' or 'tanh'.")

    def _angle_positioning_cupy(self, z: Any, name: str) -> Any:
        cp = self._cupy or self._import_cupy()
        if name == "linear":
            return self.angle_scale * z
        if name == "tanh":
            return self.angle_scale * cp.tanh(z)
        raise ValueError(f"Unsupported angle_positioning_name={name!r}; expected 'linear' or 'tanh'.")

    @classmethod
    def _pauli_masks_from_observables(
        cls,
        observables: Sequence[Operator | SparsePauliOp],
        *,
        n: int,
    ) -> tuple[list[tuple[int, int, int]], list[str]]:
        masks: list[tuple[int, int, int]] = []
        labels: list[str] = []
        for op in observables:
            label = cls._single_term_pauli_label(op)
            if label is None:
                raise ValueError(
                    "output_kind='expectation' supports only single-term SparsePauliOp observables in v1."
                )
            if len(label) != int(n):
                raise ValueError(f"Observable label length {len(label)} != num_qubits {n}: {label!r}")
            labels.append(label)
            masks.append(cls._masks_from_pauli_label(label))
        if not masks:
            raise ValueError("output_kind='expectation' requires at least one observable.")
        return masks, labels

    @staticmethod
    def _single_term_pauli_label(op: Operator | SparsePauliOp) -> str | None:
        if isinstance(op, SparsePauliOp) and len(op.paulis) == 1:
            return op.paulis[0].to_label()
        return None

    @staticmethod
    def _masks_from_pauli_label(label: str) -> tuple[int, int, int]:
        n = len(label)
        xmask = 0
        zmask = 0
        ny = 0
        for q in range(n):
            ch = label[n - 1 - q]
            if ch == "X":
                xmask |= 1 << q
            elif ch == "Z":
                zmask |= 1 << q
            elif ch == "Y":
                xmask |= 1 << q
                zmask |= 1 << q
                ny += 1
            elif ch != "I":
                raise ValueError(f"Unsupported Pauli character {ch!r} in label {label!r}.")
        return xmask, zmask, ny

    def _pauli_phase_vector_cupy(self, *, zmask: int, ny: int, dim: int) -> Any:
        cp = self._cupy or self._import_cupy()
        rows_cpu = np.arange(dim, dtype=np.uint32)
        parity_cpu = self._bitcount_parity(np.bitwise_and(rows_cpu, np.uint32(zmask)))
        sign = cp.asarray(1.0 - 2.0 * parity_cpu.astype(np.float64), dtype=cp.float64)
        return ((1j ** int(ny)) * sign).astype(cp.complex128, copy=False)

    @staticmethod
    def _bitcount_parity(arr: np.ndarray) -> np.ndarray:
        if hasattr(np, "bit_count"):
            return (np.bit_count(arr) & 1).astype(np.int8)
        v = np.vectorize(lambda x: int(x).bit_count() & 1, otypes=[np.int8])
        return v(arr)

    def _apply_reservoir_unitary(
        self,
        states: np.ndarray,
        *,
        theta: np.ndarray,
        j_vals: np.ndarray,
        hx_vals: np.ndarray,
        hz_vals: np.ndarray,
    ) -> None:
        n = int(self.qrc_cfg.num_qubits)

        for q in range(n):
            self._apply_ry(states, q, float(theta[q]))
        for e, (q1, q2) in enumerate(self.qrc_cfg.topology.edges):
            self._apply_rzz(states, int(q1), int(q2), float(j_vals[e]))
        for q in range(n):
            self._apply_rz(states, q, float(hz_vals[q]))
        for q in range(n):
            self._apply_rx(states, q, float(hx_vals[q]))

    def _density_from_ensemble(self, states: np.ndarray, weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=np.float64)
        rho = np.einsum("k,ki,kj->ij", weights, states, states.conj(), optimize=True)
        return np.asarray(rho, dtype=self.state_dtype)

    def _density_from_ensemble_batched(self, states: np.ndarray, weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=np.float64)
        rho = np.einsum("ba,bai,baj->bij", weights, states, states.conj(), optimize=True)
        return np.asarray(rho, dtype=self.state_dtype)

    def _density_from_ensemble_cupy(self, states: Any, weights: Any) -> Any:
        cp = self._cupy or self._import_cupy()
        rho = cp.einsum("ba,bai,baj->bij", weights, states, states.conj(), optimize=True)
        return rho.astype(cp.dtype(self.state_dtype.name), copy=False)

    @staticmethod
    def _plus_state(n: int) -> np.ndarray:
        dim = 1 << int(n)
        return np.full((dim,), 1.0 / np.sqrt(dim), dtype=complex)

    def _build_gate_index_cache(self, n: int) -> _GateIndexCache:
        dim = 1 << int(n)
        indices = np.arange(dim, dtype=np.uint64)

        single_i0: list[np.ndarray] = []
        single_i1: list[np.ndarray] = []
        for q in range(n):
            mask = np.uint64(1 << int(q))
            i0 = indices[(indices & mask) == 0].astype(np.intp, copy=False)
            i1 = (i0 | np.intp(mask)).astype(np.intp, copy=False)
            single_i0.append(i0)
            single_i1.append(i1)

        rzz_same: list[np.ndarray] = []
        for q1, q2 in self.qrc_cfg.topology.edges:
            b1 = (indices >> int(q1)) & 1
            b2 = (indices >> int(q2)) & 1
            rzz_same.append(b1 == b2)

        return _GateIndexCache(
            single_i0=tuple(single_i0),
            single_i1=tuple(single_i1),
            rzz_same=tuple(rzz_same),
        )

    def _get_cupy_gate_cache(self) -> _GateIndexCache:
        cp = self._cupy or self._import_cupy()
        if self._cupy_gate_cache is None:
            self._cupy_gate_cache = _GateIndexCache(
                single_i0=tuple(cp.asarray(arr, dtype=cp.int64) for arr in self._gate_cache.single_i0),
                single_i1=tuple(cp.asarray(arr, dtype=cp.int64) for arr in self._gate_cache.single_i1),
                rzz_same=tuple(cp.asarray(arr, dtype=cp.bool_) for arr in self._gate_cache.rzz_same),
            )
        return self._cupy_gate_cache

    @staticmethod
    def _apply_single_qubit(states: np.ndarray, q: int, matrix: np.ndarray) -> None:
        dim = states.shape[1]
        mask = 1 << int(q)
        for base in range(dim):
            if base & mask:
                continue
            i0 = base
            i1 = base | mask
            a0 = states[:, i0].copy()
            a1 = states[:, i1].copy()
            states[:, i0] = matrix[0, 0] * a0 + matrix[0, 1] * a1
            states[:, i1] = matrix[1, 0] * a0 + matrix[1, 1] * a1

    @classmethod
    def _apply_ry(cls, states: np.ndarray, q: int, theta: float) -> None:
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        cls._apply_single_qubit(states, q, np.asarray([[c, -s], [s, c]], dtype=states.dtype))

    @classmethod
    def _apply_rx(cls, states: np.ndarray, q: int, theta: float) -> None:
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        cls._apply_single_qubit(states, q, np.asarray([[c, -1j * s], [-1j * s, c]], dtype=states.dtype))

    @classmethod
    def _apply_rz(cls, states: np.ndarray, q: int, theta: float) -> None:
        phase_minus = np.exp(-0.5j * theta)
        phase_plus = np.exp(0.5j * theta)
        cls._apply_single_qubit(states, q, np.asarray([[phase_minus, 0.0], [0.0, phase_plus]], dtype=states.dtype))

    @staticmethod
    def _apply_rzz(states: np.ndarray, q1: int, q2: int, theta: float) -> None:
        dim = states.shape[1]
        indices = np.arange(dim, dtype=np.uint64)
        b1 = (indices >> int(q1)) & 1
        b2 = (indices >> int(q2)) & 1
        same = b1 == b2
        phases = np.where(same, np.exp(-0.5j * theta), np.exp(0.5j * theta)).astype(states.dtype, copy=False)
        states *= phases[None, :]

    def _apply_reservoir_unitary_batched(
        self,
        states: np.ndarray,
        *,
        theta: np.ndarray,
        j_vals: np.ndarray,
        hx_vals: np.ndarray,
        hz_vals: np.ndarray,
    ) -> None:
        n = int(self.qrc_cfg.num_qubits)

        for q in range(n):
            self._apply_ry_batched(states, q, theta[:, q])
        for e, _ in enumerate(self.qrc_cfg.topology.edges):
            self._apply_rzz_batched(states, j_vals[:, e], edge_index=e)
        for q in range(n):
            self._apply_rz_batched(states, q, hz_vals[:, q])
        for q in range(n):
            self._apply_rx_batched(states, q, hx_vals[:, q])

    def _apply_ry_batched(self, states: np.ndarray, q: int, theta: np.ndarray) -> None:
        c = np.cos(theta / 2.0)[:, None, None]
        s = np.sin(theta / 2.0)[:, None, None]
        i0 = self._gate_cache.single_i0[int(q)]
        i1 = self._gate_cache.single_i1[int(q)]
        a0 = states[:, :, i0].copy()
        a1 = states[:, :, i1].copy()
        states[:, :, i0] = c * a0 - s * a1
        states[:, :, i1] = s * a0 + c * a1

    def _apply_rx_batched(self, states: np.ndarray, q: int, theta: np.ndarray) -> None:
        c = np.cos(theta / 2.0)[:, None, None]
        s = np.sin(theta / 2.0)[:, None, None]
        i0 = self._gate_cache.single_i0[int(q)]
        i1 = self._gate_cache.single_i1[int(q)]
        a0 = states[:, :, i0].copy()
        a1 = states[:, :, i1].copy()
        states[:, :, i0] = c * a0 - 1j * s * a1
        states[:, :, i1] = -1j * s * a0 + c * a1

    def _apply_rz_batched(self, states: np.ndarray, q: int, theta: np.ndarray) -> None:
        phase_minus = np.exp(-0.5j * theta)[:, None, None]
        phase_plus = np.exp(0.5j * theta)[:, None, None]
        i0 = self._gate_cache.single_i0[int(q)]
        i1 = self._gate_cache.single_i1[int(q)]
        states[:, :, i0] *= phase_minus
        states[:, :, i1] *= phase_plus

    def _apply_rzz_batched(
        self,
        states: np.ndarray,
        theta: np.ndarray,
        *,
        edge_index: int,
    ) -> None:
        same = self._gate_cache.rzz_same[int(edge_index)]
        phase_same = np.exp(-0.5j * theta)[:, None]
        phase_diff = np.exp(0.5j * theta)[:, None]
        phases = np.where(same[None, :], phase_same, phase_diff).astype(states.dtype, copy=False)
        states *= phases[:, None, :]

    def _apply_reservoir_unitary_cupy(
        self,
        states: Any,
        *,
        theta: Any,
        j_vals: Any,
        hx_vals: Any,
        hz_vals: Any,
    ) -> None:
        n = int(self.qrc_cfg.num_qubits)

        for q in range(n):
            self._apply_ry_cupy(states, q, theta[:, q])
        for e, _ in enumerate(self.qrc_cfg.topology.edges):
            self._apply_rzz_cupy(states, j_vals[:, e], edge_index=e)
        for q in range(n):
            self._apply_rz_cupy(states, q, hz_vals[:, q])
        for q in range(n):
            self._apply_rx_cupy(states, q, hx_vals[:, q])

    def _apply_ry_cupy(self, states: Any, q: int, theta: Any) -> None:
        cp = self._cupy or self._import_cupy()
        cache = self._get_cupy_gate_cache()
        c = cp.cos(theta / 2.0)[:, None, None]
        s = cp.sin(theta / 2.0)[:, None, None]
        i0 = cache.single_i0[int(q)]
        i1 = cache.single_i1[int(q)]
        a0 = states[:, :, i0].copy()
        a1 = states[:, :, i1].copy()
        states[:, :, i0] = c * a0 - s * a1
        states[:, :, i1] = s * a0 + c * a1

    def _apply_rx_cupy(self, states: Any, q: int, theta: Any) -> None:
        cp = self._cupy or self._import_cupy()
        cache = self._get_cupy_gate_cache()
        c = cp.cos(theta / 2.0)[:, None, None]
        s = cp.sin(theta / 2.0)[:, None, None]
        i0 = cache.single_i0[int(q)]
        i1 = cache.single_i1[int(q)]
        a0 = states[:, :, i0].copy()
        a1 = states[:, :, i1].copy()
        states[:, :, i0] = c * a0 - 1j * s * a1
        states[:, :, i1] = -1j * s * a0 + c * a1

    def _apply_rz_cupy(self, states: Any, q: int, theta: Any) -> None:
        cp = self._cupy or self._import_cupy()
        cache = self._get_cupy_gate_cache()
        phase_minus = cp.exp(-0.5j * theta)[:, None, None]
        phase_plus = cp.exp(0.5j * theta)[:, None, None]
        i0 = cache.single_i0[int(q)]
        i1 = cache.single_i1[int(q)]
        states[:, :, i0] *= phase_minus
        states[:, :, i1] *= phase_plus

    def _apply_rzz_cupy(
        self,
        states: Any,
        theta: Any,
        *,
        edge_index: int,
    ) -> None:
        cp = self._cupy or self._import_cupy()
        cache = self._get_cupy_gate_cache()
        same = cache.rzz_same[int(edge_index)]
        phase_same = cp.exp(-0.5j * theta)[:, None]
        phase_diff = cp.exp(0.5j * theta)[:, None]
        phases = cp.where(same[None, :], phase_same, phase_diff).astype(states.dtype, copy=False)
        states *= phases[:, None, :]


class TruncatedReservoirChannelRunner(ExactReservoirChannelRunner):
    """Explicitly approximate finite-history/pruned reservoir runner."""

    def __init__(self, qrc_cfg: BaseQRConfig, **kwargs: Any) -> None:
        if kwargs.get("max_history") is None and float(kwargs.get("weight_atol", 0.0)) <= 0.0:
            raise ValueError(
                "TruncatedReservoirChannelRunner requires max_history or positive weight_atol."
            )
        super().__init__(qrc_cfg, _allow_approximate=True, **kwargs)
