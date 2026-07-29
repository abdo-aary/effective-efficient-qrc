"""Backend-neutral QuaRK domain objects.

This package deliberately has no dependency on Qiskit, CuPy, Hydra, or IBM
Runtime.  Execution-framework objects belong in :mod:`src.backends`.
"""

from .capabilities import BackendCapabilities, BackendKind, EstimatorKind, Precision
from .exceptions import (
    BackendDependencyError,
    CapabilityError,
    CompilationError,
    ExecutionError,
    QuaRKError,
    UnsupportedEstimatorError,
)
from .observables import ObservableSet
from .program import (
    ProjectionSpec,
    QuaRKProgram,
    ResetChannelSpec,
    ReservoirParameters,
    ReservoirTopology,
)
from .requests import CompiledFeaturePlan, ExecutionSpec
from .results import ExecutionMetadata, FeatureBatch, StateBatch
from .seeds import SeedBundle

__all__ = [
    "BackendCapabilities",
    "BackendDependencyError",
    "BackendKind",
    "CapabilityError",
    "CompilationError",
    "CompiledFeaturePlan",
    "EstimatorKind",
    "ExecutionError",
    "ExecutionMetadata",
    "ExecutionSpec",
    "FeatureBatch",
    "ObservableSet",
    "Precision",
    "ProjectionSpec",
    "QuaRKError",
    "QuaRKProgram",
    "ResetChannelSpec",
    "ReservoirParameters",
    "ReservoirTopology",
    "SeedBundle",
    "StateBatch",
    "UnsupportedEstimatorError",
]
