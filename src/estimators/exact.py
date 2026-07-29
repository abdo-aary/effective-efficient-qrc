"""Exact local-observable feature estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.capabilities import EstimatorKind, Precision
from src.core.exceptions import CapabilityError
from src.core.program import QuaRKProgram
from src.core.requests import ExecutionSpec
from src.core.results import FeatureBatch


@dataclass(frozen=True)
class ExactFeatureEstimator:
    """Compute mathematical expectation values without sampling.

    This estimator is deliberately unavailable on physical hardware.
    """

    precision: Precision = Precision.COMPLEX128
    return_states: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "precision", Precision(self.precision))

    @property
    def kind(self) -> EstimatorKind:
        return EstimatorKind.EXACT

    def estimate(
        self,
        program: QuaRKProgram,
        windows: np.ndarray,
        backend: object,
        execution: ExecutionSpec | None = None,
    ) -> FeatureBatch:
        execution = execution or ExecutionSpec()
        capabilities = backend.capabilities  # type: ignore[attr-defined]
        capabilities.require_estimator(self.kind)
        if not capabilities.exact_mixed_state_evolution:
            raise CapabilityError(
                f"{capabilities.backend_kind.value} cannot provide exact mixed-state evolution."
            )
        if self.return_states and not capabilities.state_return:
            raise CapabilityError(
                f"{capabilities.backend_kind.value} cannot return diagnostic states."
            )
        windows = program.validate_windows(windows)
        compiled = backend.compile(program, self)  # type: ignore[attr-defined]
        return backend.execute(compiled, windows, execution)  # type: ignore[attr-defined]
