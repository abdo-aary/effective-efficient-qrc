"""Structured feature and execution results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np

from .capabilities import BackendKind, EstimatorKind


@dataclass(frozen=True)
class StateBatch:
    """Optional diagnostic reduced states, distinct from feature values."""

    density_matrices: Any

    def __post_init__(self) -> None:
        if getattr(self.density_matrices, "ndim", None) != 4:
            raise ValueError(
                "Diagnostic density matrices must have shape (N,R,D,D)."
            )


@dataclass(frozen=True)
class ExecutionMetadata:
    backend_kind: BackendKind
    estimator_kind: EstimatorKind
    exact: bool
    program_fingerprint: str
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureBatch:
    values: Any
    estimator_kind: EstimatorKind
    estimator_metadata: Mapping[str, Any]
    execution_metadata: ExecutionMetadata
    observable_labels: tuple[str, ...]
    reservoir_ids: tuple[str, ...]
    raw_snapshots: Any | None = field(default=None, repr=False)
    diagnostic_states: StateBatch | None = field(default=None, repr=False)
    ordering_version: str = "reservoir-major-observable-minor/v1"

    def __post_init__(self) -> None:
        if getattr(self.values, "ndim", None) != 3:
            raise ValueError(f"Feature values must have shape (N,R,K), got {self.values.shape}.")
        _, R, K = self.values.shape
        if R != len(self.reservoir_ids):
            raise ValueError("Feature reservoir axis does not match reservoir_ids.")
        if K != len(self.observable_labels):
            raise ValueError("Feature observable axis does not match observable_labels.")

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.values.shape)

    def flatten_for_readout(self) -> Any:
        N, R, K = self.values.shape
        return self.values.reshape(N, R * K)

    def as_numpy(self) -> "FeatureBatch":
        values = self.values
        if hasattr(values, "get"):
            values = values.get()
        return FeatureBatch(
            values=np.asarray(values),
            estimator_kind=self.estimator_kind,
            estimator_metadata=dict(self.estimator_metadata),
            execution_metadata=self.execution_metadata,
            observable_labels=self.observable_labels,
            reservoir_ids=self.reservoir_ids,
            raw_snapshots=self.raw_snapshots,
            diagnostic_states=self.diagnostic_states,
            ordering_version=self.ordering_version,
        )
