"""Independent Qiskit Aer CPU oracle backend."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from src.backends._legacy import template_pubs
from src.backends.balanced import balanced_density_matrices
from src.core.capabilities import (
    BackendCapabilities,
    BackendKind,
    EstimatorKind,
    Precision,
)
from src.core.exceptions import BackendDependencyError, CapabilityError, CompilationError
from src.core.program import BalancedReservoirParameters, QuaRKProgram
from src.core.requests import CompiledFeaturePlan, ExecutionSpec
from src.core.results import ExecutionMetadata, FeatureBatch, StateBatch
from src.estimators.csmom import (
    CSMoMFeatureEstimator,
    ShadowSnapshots,
    reconstruct_csmom,
)
from src.estimators.exact import ExactFeatureEstimator
from src.features.pauli import exact_pauli_expectations
from src.features.shadows import sample_local_pauli_shadows_density


@dataclass(frozen=True)
class _AerPayload:
    program: QuaRKProgram
    estimator: ExactFeatureEstimator | CSMoMFeatureEstimator


class AerCPUBackend:
    """Small, independent density-matrix oracle using the SWAP dilation."""

    def __init__(
        self,
        *,
        max_qubits: int = 4,
        max_windows: int = 32,
        optimization_level: int = 0,
    ) -> None:
        self.max_qubits = int(max_qubits)
        self.max_windows = int(max_windows)
        self.optimization_level = int(optimization_level)

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_kind=BackendKind.AER,
            supported_estimators=frozenset({EstimatorKind.EXACT, EstimatorKind.CSMOM}),
            exact_mixed_state_evolution=True,
            finite_shot_execution=True,
            direct_expectations=False,
            state_return=True,
            parameter_batching=True,
            gpu_execution=False,
            hardware_execution=False,
            stochastic_trajectories=False,
            mid_circuit_reset=True,
            noise_models=False,
            asynchronous_jobs=False,
            measurement_mitigation=False,
            supported_precisions=frozenset({Precision.COMPLEX128}),
            max_qubits=self.max_qubits,
            max_windows=self.max_windows,
        )

    def compile(self, program: QuaRKProgram, estimator: object) -> CompiledFeaturePlan:
        self.capabilities.require_estimator(estimator.kind)  # type: ignore[attr-defined]
        if program.num_qubits > self.max_qubits:
            raise CapabilityError(
                f"Aer oracle supports at most {self.max_qubits} reservoir qubits; "
                f"received {program.num_qubits}."
            )
        if isinstance(estimator, ExactFeatureEstimator):
            if estimator.precision is not Precision.COMPLEX128:
                raise CapabilityError("Aer oracle exposes only complex128 exact results.")
        elif not isinstance(estimator, CSMoMFeatureEstimator):
            raise CompilationError(f"Unsupported estimator object {type(estimator)!r}.")
        return CompiledFeaturePlan(
            backend_kind=BackendKind.AER,
            estimator_kind=estimator.kind,  # type: ignore[attr-defined]
            program_fingerprint=program.fingerprint(),
            payload=_AerPayload(program=program, estimator=estimator),
            compilation_metadata={
                "channel_realization": (
                    "dense-balanced-replacement-channel"
                    if isinstance(program.reservoirs, BalancedReservoirParameters)
                    else "2n+1-qubit-swap-dilation"
                ),
                "simulator_method": "density_matrix",
                "optimization_level": self.optimization_level,
            },
        )

    def _density_matrices(
        self,
        program: QuaRKProgram,
        windows: np.ndarray,
        execution: ExecutionSpec,
    ) -> np.ndarray:
        if isinstance(program.reservoirs, BalancedReservoirParameters):
            return balanced_density_matrices(program, windows)
        try:
            from src.backends.aer.legacy_runner import ExactAerCircuitsRunner
        except ImportError as exc:
            raise BackendDependencyError(
                "AerCPUBackend requires the optional Qiskit and qiskit-aer dependencies."
            ) from exc

        cfg, pubs = template_pubs(program, windows)
        result = ExactAerCircuitsRunner(cfg).run_pubs(
            pubs=pubs,
            seed_simulator=execution.seeds.integer("measurement_outcomes"),
            optimization_level=self.optimization_level,
            device="CPU",
            max_parallel_threads=execution.options.get("max_parallel_threads", 1),
            max_parallel_experiments=execution.options.get("max_parallel_experiments", 1),
            max_parallel_shots=1,
            chunk_size=execution.chunk_size,
        )
        return np.asarray(result.states)

    def execute(
        self,
        compiled: CompiledFeaturePlan,
        windows: np.ndarray,
        execution: ExecutionSpec,
    ) -> FeatureBatch:
        if compiled.backend_kind is not BackendKind.AER:
            raise CompilationError("Compiled plan belongs to a different backend.")
        payload = compiled.payload
        if not isinstance(payload, _AerPayload):
            raise CompilationError("Invalid Aer compiled payload.")
        program = payload.program
        windows = program.validate_windows(windows)
        if windows.shape[0] > self.max_windows:
            raise CapabilityError(
                f"Aer oracle supports at most {self.max_windows} windows per request; "
                f"received {windows.shape[0]}."
            )

        started = perf_counter()
        states = self._density_matrices(program, windows, execution)
        raw: ShadowSnapshots | None = None
        diagnostic_states: StateBatch | None = None
        if isinstance(payload.estimator, ExactFeatureEstimator):
            values = exact_pauli_expectations(states, program.observables.labels)
            if payload.estimator.return_states:
                diagnostic_states = StateBatch(states)
            estimator_metadata: dict[str, Any] = {
                "semantics": "analytic_exact",
                "precision": payload.estimator.precision.value,
            }
            exact = True
        else:
            estimator = payload.estimator
            basis_rng = execution.seeds.generator("shadow_bases")
            bases = basis_rng.integers(
                0,
                3,
                size=(
                    windows.shape[0],
                    program.num_reservoirs,
                    estimator.snapshots,
                    program.num_qubits,
                ),
                dtype=np.uint8,
            )
            raw = sample_local_pauli_shadows_density(
                states,
                bases,
                outcome_rng=execution.seeds.generator("measurement_outcomes"),
            )
            values = reconstruct_csmom(
                raw,
                program.observables.labels,
                median_blocks=estimator.median_blocks,
            )
            estimator_metadata = {
                "semantics": "local_pauli_classical_shadow",
                "snapshots": estimator.snapshots,
                "median_blocks": estimator.median_blocks,
                "shadow_factor_by_observable": [
                    3 ** locality for locality in program.observables.localities
                ],
                "channel_sampling": "exact_dilation_density_matrix",
            }
            exact = False

        metadata = ExecutionMetadata(
            backend_kind=BackendKind.AER,
            estimator_kind=payload.estimator.kind,
            exact=exact,
            program_fingerprint=compiled.program_fingerprint,
            details={
                **dict(compiled.compilation_metadata),
                "elapsed_seconds": perf_counter() - started,
                "windows": int(windows.shape[0]),
                "reservoirs": program.num_reservoirs,
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
            diagnostic_states=diagnostic_states,
        )
