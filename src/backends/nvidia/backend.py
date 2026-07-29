"""NVIDIA production backend for exact and CSMoM feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from src.backends.nvidia._engine import make_runner, pack_program
from src.backends.nvidia.csmom import sample_csmom_gpu
from src.core.capabilities import (
    BackendCapabilities,
    BackendKind,
    EstimatorKind,
    Precision,
)
from src.core.exceptions import BackendDependencyError, CompilationError
from src.core.program import QuaRKProgram
from src.core.requests import CompiledFeaturePlan, ExecutionSpec
from src.core.results import ExecutionMetadata, FeatureBatch, StateBatch
from src.estimators.csmom import CSMoMFeatureEstimator, reconstruct_csmom
from src.estimators.exact import ExactFeatureEstimator


@dataclass(frozen=True)
class _NvidiaPayload:
    program: QuaRKProgram
    estimator: ExactFeatureEstimator | CSMoMFeatureEstimator


class NvidiaBackend:
    """Exact CuPy production engine plus direct GPU shadow sampling."""

    def __init__(
        self,
        *,
        gpu_id: int | None = 0,
        chunk_size: int = 512,
        engine: str = "cupy",
    ) -> None:
        if engine != "cupy":
            raise ValueError("The production backend currently supports engine='cupy' only.")
        self.gpu_id = None if gpu_id is None else int(gpu_id)
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        self.engine = engine

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_kind=BackendKind.NVIDIA,
            supported_estimators=frozenset({EstimatorKind.EXACT, EstimatorKind.CSMOM}),
            exact_mixed_state_evolution=True,
            finite_shot_execution=True,
            direct_expectations=True,
            state_return=True,
            parameter_batching=True,
            gpu_execution=True,
            hardware_execution=False,
            stochastic_trajectories=True,
            mid_circuit_reset=False,
            noise_models=False,
            asynchronous_jobs=False,
            measurement_mitigation=False,
            supported_precisions=frozenset(
                {Precision.COMPLEX64, Precision.COMPLEX128}
            ),
        )

    def compile(self, program: QuaRKProgram, estimator: object) -> CompiledFeaturePlan:
        self.capabilities.require_estimator(estimator.kind)  # type: ignore[attr-defined]
        if not isinstance(estimator, (ExactFeatureEstimator, CSMoMFeatureEstimator)):
            raise CompilationError(f"Unsupported estimator object {type(estimator)!r}.")
        return CompiledFeaturePlan(
            backend_kind=BackendKind.NVIDIA,
            estimator_kind=estimator.kind,
            program_fingerprint=program.fingerprint(),
            payload=_NvidiaPayload(program=program, estimator=estimator),
            compilation_metadata={
                "engine": self.engine,
                "gpu_id": self.gpu_id,
                "channel_realization": "exact-weighted-pure-state-ensemble",
                "history_truncation": False,
                "branch_pruning": False,
            },
        )

    def _exact(
        self,
        program: QuaRKProgram,
        windows: np.ndarray,
        estimator: ExactFeatureEstimator,
        execution: ExecutionSpec,
    ) -> tuple[Any, StateBatch | None, dict[str, float], dict[str, int]]:
        try:
            import cupy as cp
        except ImportError as exc:
            raise BackendDependencyError(
                "NvidiaBackend requires CuPy built for the local CUDA runtime."
            ) from exc

        pack_started = perf_counter()
        flat, numeric_layout = pack_program(program, windows)
        pack_seconds = perf_counter() - pack_started
        output_backend = "cupy" if execution.retain_device_array else "numpy"
        runner = make_runner(
            program,
            state_dtype=estimator.precision.value,
            chunk_size=execution.chunk_size or self.chunk_size,
            gpu_id=self.gpu_id,
            output_backend=output_backend,
            output_kind="expectation",
        )
        masks = [
            runner._masks_from_pauli_label(label)
            for label in program.observables.labels
        ]
        device = cp.cuda.Device(self.gpu_id) if self.gpu_id is not None else cp.cuda.Device()
        pool = cp.get_default_memory_pool()
        baseline_pool_bytes = int(pool.total_bytes())
        with device:
            cp.cuda.Stream.null.synchronize()
            gpu_started = perf_counter()
            values = runner._run_flat_cupy_expectation_output(
                flat=flat,
                layout=numeric_layout,
                angle_positioning_name=program.angle_map,
                plus_state=runner._plus_state(program.num_qubits).astype(
                    estimator.precision.value, copy=False
                ),
                chunk_size=execution.chunk_size or self.chunk_size,
                pauli_masks=masks,
            ).reshape(
                windows.shape[0],
                program.num_reservoirs,
                program.observables.size,
            )
            cp.cuda.Stream.null.synchronize()
            gpu_seconds = perf_counter() - gpu_started
        peak_pool_bytes = max(0, int(pool.total_bytes()) - baseline_pool_bytes)

        diagnostic = None
        if estimator.return_states:
            state_runner = make_runner(
                program,
                state_dtype=estimator.precision.value,
                chunk_size=execution.chunk_size or self.chunk_size,
                gpu_id=self.gpu_id,
                output_backend=output_backend,
                output_kind="density_matrix",
            )
            with device:
                density = state_runner._run_flat_cupy_output(
                    flat=flat,
                    layout=numeric_layout,
                    angle_positioning_name=program.angle_map,
                    plus_state=state_runner._plus_state(program.num_qubits).astype(
                        estimator.precision.value, copy=False
                    ),
                    chunk_size=execution.chunk_size or self.chunk_size,
                ).reshape(
                    windows.shape[0],
                    program.num_reservoirs,
                    1 << program.num_qubits,
                    1 << program.num_qubits,
                )
            if not execution.retain_device_array:
                density = cp.asnumpy(density)
            diagnostic = StateBatch(density)
        timings = {
            "projection_pack_seconds": float(pack_seconds),
            "gpu_execution_seconds": float(gpu_seconds),
        }
        resources = {
            "gpu_pool_baseline_bytes": baseline_pool_bytes,
            "gpu_peak_increment_bytes": peak_pool_bytes,
        }
        return values, diagnostic, timings, resources

    def execute(
        self,
        compiled: CompiledFeaturePlan,
        windows: np.ndarray,
        execution: ExecutionSpec,
    ) -> FeatureBatch:
        if compiled.backend_kind is not BackendKind.NVIDIA:
            raise CompilationError("Compiled plan belongs to a different backend.")
        payload = compiled.payload
        if not isinstance(payload, _NvidiaPayload):
            raise CompilationError("Invalid NVIDIA compiled payload.")
        program = payload.program
        windows = program.validate_windows(windows)
        started = perf_counter()
        raw = None
        diagnostic = None

        if isinstance(payload.estimator, ExactFeatureEstimator):
            values, diagnostic, exact_timings, exact_resources = self._exact(
                program, windows, payload.estimator, execution
            )
            estimator_metadata: dict[str, Any] = {
                "semantics": "analytic_exact",
                "precision": payload.estimator.precision.value,
            }
            exact = True
        else:
            raw = sample_csmom_gpu(
                program,
                windows,
                payload.estimator,
                execution,
                state_dtype=Precision.COMPLEX128.value,
                gpu_id=self.gpu_id,
                default_chunk_size=self.chunk_size,
            )
            values = reconstruct_csmom(
                raw,
                program.observables.labels,
                median_blocks=payload.estimator.median_blocks,
            )
            estimator_metadata = {
                "semantics": "local_pauli_classical_shadow",
                "snapshots": payload.estimator.snapshots,
                "median_blocks": payload.estimator.median_blocks,
                "shadow_factor_by_observable": [
                    3 ** locality for locality in program.observables.localities
                ],
                "channel_sampling": "exact_branch_weight_sampling",
            }
            exact = False

        metadata = ExecutionMetadata(
            backend_kind=BackendKind.NVIDIA,
            estimator_kind=payload.estimator.kind,
            exact=exact,
            program_fingerprint=compiled.program_fingerprint,
            details={
                **dict(compiled.compilation_metadata),
                "elapsed_seconds": perf_counter() - started,
                "chunk_size": execution.chunk_size or self.chunk_size,
                "device_output": bool(execution.retain_device_array),
                "timings": exact_timings if exact else {},
                "resources": exact_resources if exact else {},
            },
        )
        return FeatureBatch(
            values=values,
            estimator_kind=payload.estimator.kind,
            estimator_metadata=estimator_metadata,
            execution_metadata=metadata,
            observable_labels=program.observables.labels,
            reservoir_ids=program.reservoirs.reservoir_ids,
            raw_snapshots=raw if getattr(payload.estimator, "retain_raw_snapshots", False) else None,
            diagnostic_states=diagnostic,
        )
