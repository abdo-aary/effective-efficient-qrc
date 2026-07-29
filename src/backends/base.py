"""Narrow protocol shared by the three principal QuaRK backends."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from src.core.capabilities import BackendCapabilities
from src.core.program import QuaRKProgram
from src.core.requests import CompiledFeaturePlan, ExecutionSpec
from src.core.results import FeatureBatch
from src.estimators.base import FeatureEstimator


class QuaRKBackend(Protocol):
    @property
    def capabilities(self) -> BackendCapabilities:
        ...

    def compile(
        self,
        program: QuaRKProgram,
        estimator: FeatureEstimator,
    ) -> CompiledFeaturePlan:
        ...

    def execute(
        self,
        compiled: CompiledFeaturePlan,
        windows: np.ndarray,
        execution: ExecutionSpec,
    ) -> FeatureBatch:
        ...
