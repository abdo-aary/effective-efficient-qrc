"""Feature-estimator protocol."""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

import numpy as np

from src.core.capabilities import EstimatorKind

if TYPE_CHECKING:
    from src.backends.base import QuaRKBackend
    from src.core.program import QuaRKProgram
    from src.core.requests import ExecutionSpec
    from src.core.results import FeatureBatch


class FeatureEstimator(Protocol):
    """Statistical semantics applied by a QuaRK execution backend."""

    @property
    def kind(self) -> EstimatorKind:
        ...

    def estimate(
        self,
        program: "QuaRKProgram",
        windows: np.ndarray,
        backend: "QuaRKBackend",
        execution: "ExecutionSpec",
    ) -> "FeatureBatch":
        ...
