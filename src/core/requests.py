"""Compilation and execution requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .capabilities import BackendKind, EstimatorKind
from .seeds import SeedBundle


@dataclass(frozen=True)
class ExecutionSpec:
    seeds: SeedBundle = field(default_factory=lambda: SeedBundle.from_root(0))
    chunk_size: int | None = None
    retain_device_array: bool = False
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.chunk_size is not None and int(self.chunk_size) < 1:
            raise ValueError("chunk_size must be positive when supplied.")
        object.__setattr__(
            self, "chunk_size", None if self.chunk_size is None else int(self.chunk_size)
        )
        object.__setattr__(self, "options", dict(self.options))


@dataclass(frozen=True)
class CompiledFeaturePlan:
    backend_kind: BackendKind
    estimator_kind: EstimatorKind
    program_fingerprint: str
    payload: Any = field(repr=False)
    compilation_metadata: Mapping[str, Any] = field(default_factory=dict)
