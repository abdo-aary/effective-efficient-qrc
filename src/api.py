"""Stable public API for backend-neutral QuaRK execution."""

from __future__ import annotations

from typing import Any

from src.backends.aer import AerCPUBackend
from src.backends.ibm import IBMRuntimeBackend
from src.backends.nvidia import NvidiaBackend
from src.core import (
    BalancedReservoirParameters,
    BackendCapabilities,
    BackendKind,
    CompiledFeaturePlan,
    EstimatorKind,
    ExecutionMetadata,
    ExecutionSpec,
    FeatureBatch,
    ObservableSet,
    Precision,
    ProjectionSpec,
    QuaRKProgram,
    ResetChannelSpec,
    ReservoirParameters,
    ReservoirTopology,
    SeedBundle,
)
from src.estimators import CSMoMFeatureEstimator, ExactFeatureEstimator


def program_from_legacy_components(
    *,
    qrc_cfg: Any,
    observables: Any,
    window_length: int,
    num_reservoirs: int,
    lam_0: float,
    reservoir_seed: int,
    eps: float = 1e-8,
    angle_map: str = "linear",
    dynamics_mode: str = "random",
) -> QuaRKProgram:
    """Convert current configuration components without importing Hydra.

    This boundary function preserves the legacy reservoir draw order while
    producing a backend-neutral immutable program.
    """

    topology = ReservoirTopology(
        num_qubits=int(qrc_cfg.num_qubits),
        edges=tuple(tuple(edge) for edge in qrc_cfg.topology.edges),
        name="ring",
    )
    parameters = ReservoirParameters.sample_legacy(
        num_reservoirs=int(num_reservoirs),
        num_qubits=topology.num_qubits,
        num_edges=len(topology.edges),
        lambda_first=float(lam_0),
        seed=int(reservoir_seed),
        eps=float(eps),
        dynamics_mode=str(dynamics_mode),
    )
    return QuaRKProgram(
        projection=ProjectionSpec(
            matrix=qrc_cfg.projection,
            mode=str(getattr(qrc_cfg, "projection_mode", "legacy")),
        ),
        topology=topology,
        reservoirs=parameters,
        reset_channel=ResetChannelSpec(),
        observables=ObservableSet.from_qiskit(observables),
        window_length=int(window_length),
        angle_map=str(angle_map),
    )


__all__ = [
    "AerCPUBackend",
    "BalancedReservoirParameters",
    "BackendCapabilities",
    "BackendKind",
    "CSMoMFeatureEstimator",
    "CompiledFeaturePlan",
    "EstimatorKind",
    "ExactFeatureEstimator",
    "ExecutionMetadata",
    "ExecutionSpec",
    "FeatureBatch",
    "IBMRuntimeBackend",
    "NvidiaBackend",
    "ObservableSet",
    "Precision",
    "ProjectionSpec",
    "QuaRKProgram",
    "ResetChannelSpec",
    "ReservoirParameters",
    "ReservoirTopology",
    "SeedBundle",
    "program_from_legacy_components",
]
