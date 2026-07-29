"""Backend and estimator capability declarations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EstimatorKind(str, Enum):
    EXACT = "exact"
    CSMOM = "csmom"


class BackendKind(str, Enum):
    AER = "aer_cpu"
    NVIDIA = "nvidia_gpu"
    IBM = "ibm_runtime"


class Precision(str, Enum):
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"


@dataclass(frozen=True)
class BackendCapabilities:
    """Immutable statement of backend behavior.

    Unknown limits are represented by ``None`` rather than guessed values.
    """

    backend_kind: BackendKind
    supported_estimators: frozenset[EstimatorKind]
    exact_mixed_state_evolution: bool
    finite_shot_execution: bool
    direct_expectations: bool
    state_return: bool
    parameter_batching: bool
    gpu_execution: bool
    hardware_execution: bool
    stochastic_trajectories: bool
    mid_circuit_reset: bool
    noise_models: bool
    asynchronous_jobs: bool
    measurement_mitigation: bool
    supported_precisions: frozenset[Precision]
    max_qubits: int | None = None
    max_windows: int | None = None

    def require_estimator(self, estimator: EstimatorKind) -> None:
        from .exceptions import UnsupportedEstimatorError

        if estimator not in self.supported_estimators:
            supported = ", ".join(sorted(item.value for item in self.supported_estimators))
            raise UnsupportedEstimatorError(
                f"{self.backend_kind.value} does not support estimator={estimator.value!r}; "
                f"supported estimators: {supported or 'none'}."
            )
