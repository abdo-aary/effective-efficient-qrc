"""IBM Runtime adapter for stochastic-channel CSMoM execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

import numpy as np

from src.backends.ibm.csmom import (
    CircuitGroup,
    build_grouped_shadow_circuits,
    counts_to_snapshots,
)
from src.backends.ibm.jobs import IBMJobHandle
from src.backends.ibm.trajectories import sample_hardware_shadow_plan
from src.core.capabilities import (
    BackendCapabilities,
    BackendKind,
    EstimatorKind,
)
from src.core.exceptions import (
    BackendDependencyError,
    CapabilityError,
    CompilationError,
    ExecutionError,
)
from src.core.program import QuaRKProgram
from src.core.requests import CompiledFeaturePlan, ExecutionSpec
from src.core.results import ExecutionMetadata, FeatureBatch
from src.estimators.csmom import CSMoMFeatureEstimator, reconstruct_csmom


@dataclass(frozen=True)
class _IBMPayload:
    program: QuaRKProgram
    estimator: CSMoMFeatureEstimator


class IBMRuntimeBackend:
    """Execute CSMoM features on IBM hardware or a local circuit smoke backend."""

    def __init__(
        self,
        *,
        backend_name: str,
        service: Any | None = None,
        optimization_level: int = 1,
        local_simulator: bool = False,
    ) -> None:
        self.backend_name = str(backend_name)
        self.service = service
        self.optimization_level = int(optimization_level)
        self.local_simulator = bool(local_simulator)

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_kind=BackendKind.IBM,
            supported_estimators=frozenset({EstimatorKind.CSMOM}),
            exact_mixed_state_evolution=False,
            finite_shot_execution=True,
            direct_expectations=False,
            state_return=False,
            parameter_batching=False,
            gpu_execution=False,
            hardware_execution=not self.local_simulator,
            stochastic_trajectories=True,
            mid_circuit_reset=False,
            noise_models=False,
            asynchronous_jobs=not self.local_simulator,
            measurement_mitigation=False,
            supported_precisions=frozenset(),
        )

    def compile(self, program: QuaRKProgram, estimator: object) -> CompiledFeaturePlan:
        # This check guarantees exact requests fail before service access or circuit work.
        self.capabilities.require_estimator(estimator.kind)  # type: ignore[attr-defined]
        if not isinstance(estimator, CSMoMFeatureEstimator):
            raise CompilationError("IBM Runtime supports only CSMoMFeatureEstimator.")
        return CompiledFeaturePlan(
            backend_kind=BackendKind.IBM,
            estimator_kind=EstimatorKind.CSMOM,
            program_fingerprint=program.fingerprint(),
            payload=_IBMPayload(program=program, estimator=estimator),
            compilation_metadata={
                "backend_name": self.backend_name,
                "channel_realization": "independent-pre-sampled-last-reset-trajectories",
                "optimization_level": self.optimization_level,
                "local_simulator": self.local_simulator,
            },
        )

    def _backend(self):
        if self.local_simulator:
            try:
                from qiskit_aer import AerSimulator
            except ImportError as exc:
                raise BackendDependencyError(
                    "Local IBM trajectory validation requires qiskit-aer."
                ) from exc
            return AerSimulator()
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
        except ImportError as exc:
            raise BackendDependencyError(
                "IBM hardware execution requires qiskit-ibm-runtime."
            ) from exc
        service = self.service or QiskitRuntimeService()
        return service.backend(self.backend_name)

    def _runtime_service(self):
        if self.service is not None:
            return self.service
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
        except ImportError as exc:
            raise BackendDependencyError(
                "IBM job resumption requires qiskit-ibm-runtime."
            ) from exc
        return QiskitRuntimeService()

    def _transpile(self, backend: Any, groups: list[CircuitGroup], seed: int):
        try:
            from qiskit.transpiler.preset_passmanagers import (
                generate_preset_pass_manager,
            )
        except ImportError as exc:
            raise BackendDependencyError("IBM circuit transpilation requires Qiskit.") from exc
        manager = generate_preset_pass_manager(
            backend=backend,
            optimization_level=self.optimization_level,
            seed_transpiler=int(seed),
        )
        circuits = manager.run([group.circuit for group in groups])
        return list(circuits)

    @staticmethod
    def _transpilation_metadata(circuits: list[Any]) -> dict[str, Any]:
        depths = [int(circuit.depth()) for circuit in circuits]
        two_qubit = []
        layouts = []
        for circuit in circuits:
            operations = circuit.count_ops()
            two_qubit.append(
                int(
                    sum(
                        count
                        for name, count in operations.items()
                        if name in {"cx", "cz", "ecr", "rzz", "swap"}
                    )
                )
            )
            layouts.append(str(getattr(circuit, "layout", None)))
        return {
            "transpiled_depths": depths,
            "two_qubit_gate_counts": two_qubit,
            "physical_layouts": layouts,
        }

    def _run_local(
        self,
        backend: Any,
        circuits: list[Any],
        groups: list[CircuitGroup],
        seed: int,
    ) -> list[dict[str, int]]:
        counts: list[dict[str, int]] = []
        sequences = np.random.SeedSequence(int(seed)).spawn(len(groups))
        group_seeds = [
            int(sequence.generate_state(1, dtype=np.uint32)[0])
            for sequence in sequences
        ]
        for circuit, group, group_seed in zip(circuits, groups, group_seeds):
            result = backend.run(
                circuit,
                shots=len(group.snapshot_indices),
                seed_simulator=group_seed,
            ).result()
            counts.append(dict(result.get_counts(0)))
        return counts

    def _run_runtime(
        self,
        backend: Any,
        circuits: list[Any],
        groups: list[CircuitGroup],
    ) -> tuple[list[dict[str, int]], IBMJobHandle, Any]:
        try:
            from qiskit.primitives.containers.sampler_pub import SamplerPub
            from qiskit_ibm_runtime import SamplerV2
        except ImportError as exc:
            raise BackendDependencyError(
                "IBM hardware execution requires compatible Qiskit Runtime V2 primitives."
            ) from exc
        pubs = [
            SamplerPub(circuit, shots=len(group.snapshot_indices))
            for circuit, group in zip(circuits, groups)
        ]
        sampler = SamplerV2(mode=backend)
        job = sampler.run(pubs)
        handle = IBMJobHandle(
            job_id=str(job.job_id()),
            backend_name=self.backend_name,
            program_fingerprint="",
            estimator_kind=EstimatorKind.CSMOM.value,
            submitted_at=datetime.now(timezone.utc).isoformat(),
            group_count=len(groups),
        )
        result = job.result()
        counts = [dict(pub_result.data.meas.get_counts()) for pub_result in result]
        return counts, handle, job

    def submit(
        self,
        compiled: CompiledFeaturePlan,
        windows: np.ndarray,
        execution: ExecutionSpec,
    ) -> IBMJobHandle:
        """Submit an asynchronous Runtime job and return a persistable handle."""

        if self.local_simulator:
            raise CapabilityError("Asynchronous submission is unavailable in local simulator mode.")
        if compiled.backend_kind is not BackendKind.IBM:
            raise CompilationError("Compiled plan belongs to a different backend.")
        payload = compiled.payload
        if not isinstance(payload, _IBMPayload):
            raise CompilationError("Invalid IBM compiled payload.")
        windows = payload.program.validate_windows(windows)
        plan = sample_hardware_shadow_plan(
            payload.program, windows, payload.estimator, execution
        )
        groups = build_grouped_shadow_circuits(payload.program, windows, plan)
        backend = self._backend()
        circuits = self._transpile(
            backend, groups, execution.seeds.integer("replicate")
        )
        try:
            from qiskit.primitives.containers.sampler_pub import SamplerPub
            from qiskit_ibm_runtime import SamplerV2
        except ImportError as exc:
            raise BackendDependencyError(
                "IBM submission requires compatible Runtime V2 primitives."
            ) from exc
        pubs = [
            SamplerPub(circuit, shots=len(group.snapshot_indices))
            for circuit, group in zip(circuits, groups)
        ]
        job = SamplerV2(mode=backend).run(pubs)
        return IBMJobHandle(
            job_id=str(job.job_id()),
            backend_name=self.backend_name,
            program_fingerprint=compiled.program_fingerprint,
            estimator_kind=EstimatorKind.CSMOM.value,
            submitted_at=datetime.now(timezone.utc).isoformat(),
            group_count=len(groups),
        )

    def resume(
        self,
        handle: IBMJobHandle,
        compiled: CompiledFeaturePlan,
        windows: np.ndarray,
        execution: ExecutionSpec,
    ) -> FeatureBatch:
        """Resume and decode a previously submitted Runtime job."""

        if self.local_simulator:
            raise CapabilityError("Runtime job resumption is unavailable in local mode.")
        if handle.program_fingerprint != compiled.program_fingerprint:
            raise CompilationError("Job handle and compiled program fingerprints differ.")
        payload = compiled.payload
        if not isinstance(payload, _IBMPayload):
            raise CompilationError("Invalid IBM compiled payload.")
        program = payload.program
        windows = program.validate_windows(windows)
        plan = sample_hardware_shadow_plan(program, windows, payload.estimator, execution)
        groups = build_grouped_shadow_circuits(program, windows, plan)
        if len(groups) != handle.group_count:
            raise CompilationError(
                "Reconstructed circuit grouping differs from the submitted job."
            )
        job = self._runtime_service().job(handle.job_id)
        result = job.result()
        counts = [dict(pub_result.data.meas.get_counts()) for pub_result in result]
        raw = counts_to_snapshots(
            program=program,
            plan=plan,
            groups=groups,
            counts_by_group=counts,
        )
        values = reconstruct_csmom(
            raw,
            program.observables.labels,
            median_blocks=payload.estimator.median_blocks,
        )
        metadata = ExecutionMetadata(
            backend_kind=BackendKind.IBM,
            estimator_kind=EstimatorKind.CSMOM,
            exact=False,
            program_fingerprint=compiled.program_fingerprint,
            details={
                **dict(compiled.compilation_metadata),
                "job": handle.to_dict(),
                "job_status": str(job.status()),
                "resumed": True,
                "circuit_groups": len(groups),
                "sampled_reset_count": int(plan.reset_counts.sum()),
            },
        )
        return FeatureBatch(
            values=values,
            estimator_kind=EstimatorKind.CSMOM,
            estimator_metadata={
                "semantics": "local_pauli_classical_shadow",
                "snapshots": payload.estimator.snapshots,
                "median_blocks": payload.estimator.median_blocks,
                "channel_sampling": "independent_stochastic_trajectory_per_snapshot",
                "raw_retention": "mandatory_hardware_provenance",
            },
            execution_metadata=metadata,
            observable_labels=program.observables.labels,
            reservoir_ids=program.reservoirs.reservoir_ids,
            raw_snapshots=raw,
        )

    def execute(
        self,
        compiled: CompiledFeaturePlan,
        windows: np.ndarray,
        execution: ExecutionSpec,
    ) -> FeatureBatch:
        if compiled.backend_kind is not BackendKind.IBM:
            raise CompilationError("Compiled plan belongs to a different backend.")
        payload = compiled.payload
        if not isinstance(payload, _IBMPayload):
            raise CompilationError("Invalid IBM compiled payload.")
        program = payload.program
        windows = program.validate_windows(windows)
        started = perf_counter()
        plan = sample_hardware_shadow_plan(
            program, windows, payload.estimator, execution
        )
        groups = build_grouped_shadow_circuits(program, windows, plan)
        backend = self._backend()
        circuits = self._transpile(
            backend,
            groups,
            execution.seeds.integer("replicate"),
        )
        transpilation = self._transpilation_metadata(circuits)
        handle = None
        job = None
        if self.local_simulator:
            counts = self._run_local(
                backend,
                circuits,
                groups,
                execution.seeds.integer("measurement_outcomes"),
            )
        else:
            counts, handle, job = self._run_runtime(backend, circuits, groups)
            handle = IBMJobHandle(
                job_id=handle.job_id,
                backend_name=handle.backend_name,
                program_fingerprint=compiled.program_fingerprint,
                estimator_kind=handle.estimator_kind,
                submitted_at=handle.submitted_at,
                group_count=handle.group_count,
            )

        try:
            raw = counts_to_snapshots(
                program=program,
                plan=plan,
                groups=groups,
                counts_by_group=counts,
            )
        except Exception as exc:
            raise ExecutionError("Could not decode IBM CSMoM measurement results.") from exc
        values = reconstruct_csmom(
            raw,
            program.observables.labels,
            median_blocks=payload.estimator.median_blocks,
        )
        backend_properties = None
        if hasattr(backend, "properties"):
            try:
                properties = backend.properties()
                backend_properties = {
                    "last_update_date": str(
                        getattr(properties, "last_update_date", None)
                    )
                }
            except Exception:
                backend_properties = {"unavailable": True}
        metadata = ExecutionMetadata(
            backend_kind=BackendKind.IBM,
            estimator_kind=EstimatorKind.CSMOM,
            exact=False,
            program_fingerprint=compiled.program_fingerprint,
            details={
                **dict(compiled.compilation_metadata),
                **transpilation,
                "elapsed_seconds": perf_counter() - started,
                "circuit_groups": len(groups),
                "total_shadow_snapshots": int(np.prod(plan.bases.shape[:3])),
                "sampled_reset_count": int(plan.reset_counts.sum()),
                "job": None if handle is None else handle.to_dict(),
                "job_status": None if job is None else str(job.status()),
                "backend_properties": backend_properties,
            },
        )
        return FeatureBatch(
            values=values,
            estimator_kind=EstimatorKind.CSMOM,
            estimator_metadata={
                "semantics": "local_pauli_classical_shadow",
                "snapshots": payload.estimator.snapshots,
                "median_blocks": payload.estimator.median_blocks,
                "shadow_factor_by_observable": [
                    3 ** locality for locality in program.observables.localities
                ],
                "channel_sampling": "independent_stochastic_trajectory_per_snapshot",
                "raw_retention": "mandatory_hardware_provenance",
            },
            execution_metadata=metadata,
            observable_labels=program.observables.labels,
            reservoir_ids=program.reservoirs.reservoir_ids,
            raw_snapshots=raw,
        )
