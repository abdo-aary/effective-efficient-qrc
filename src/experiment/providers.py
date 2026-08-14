"""Behavioral protocols and deterministic fake providers for experiment plans."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .domain import (
    AcquisitionSpec,
    ComparisonSpec,
    DataSpec,
    EvaluationSpec,
    FeatureViewSpec,
    FitSpec,
    content_digest,
)
from .payload import NodePayload
from .seeding import PathSeedTree


@dataclass(frozen=True)
class ProviderIdentity:
    kind: str
    algorithm_version: str
    backend_kind: str
    numerical_precision: str


class DataProvider(Protocol):
    @property
    def identity(self) -> ProviderIdentity:
        ...

    def prepare(self, spec: DataSpec, seeds: PathSeedTree) -> NodePayload:
        ...


class RepresentationProvider(Protocol):
    @property
    def identity(self) -> ProviderIdentity:
        ...

    def acquire(
        self, spec: AcquisitionSpec, data: NodePayload, seeds: PathSeedTree
    ) -> NodePayload:
        ...

    def derive(
        self, spec: FeatureViewSpec, acquisition: NodePayload, data: NodePayload
    ) -> NodePayload:
        ...


class ReadoutProvider(Protocol):
    @property
    def identity(self) -> ProviderIdentity:
        ...

    def fit(self, spec: FitSpec, features: NodePayload) -> NodePayload:
        ...

    def evaluate(
        self,
        spec: EvaluationSpec,
        fit: NodePayload,
        features: NodePayload,
        data: NodePayload,
    ) -> NodePayload:
        ...

    def compare(
        self, spec: ComparisonSpec, evaluations: Sequence[NodePayload]
    ) -> NodePayload:
        ...


class FakeProviders(DataProvider, RepresentationProvider, ReadoutProvider):
    """Fast deterministic provider used to validate orchestration without numerics."""

    identity = ProviderIdentity(
        kind="fake",
        algorithm_version="fake-provider/v2",
        backend_kind="fake",
        numerical_precision="not_applicable",
    )

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()

    def _payload(self, operation: str, spec: object, **extra: Any) -> NodePayload:
        self.calls[operation] += 1
        return NodePayload(
            metadata={
                "provider": "fake",
                "operation": operation,
                "spec_digest": content_digest(spec),
                **extra,
            }
        )

    def prepare(self, spec: DataSpec, seeds: PathSeedTree) -> NodePayload:
        return self._payload(
            "prepare",
            spec,
            seed=seeds.integer(f"data/{spec.split}/{spec.trajectory_id}"),
            sample_count=spec.sample_count,
            task_ids=list(spec.task_ids),
        )

    def acquire(
        self, spec: AcquisitionSpec, data: NodePayload, seeds: PathSeedTree
    ) -> NodePayload:
        path = spec.randomness_path or f"representation/{spec.id}"
        return self._payload(
            "acquire",
            spec,
            seed=seeds.integer(path),
            data_digest=data.digest,
            kind=spec.kind.value,
        )

    def derive(
        self, spec: FeatureViewSpec, acquisition: NodePayload, data: NodePayload
    ) -> NodePayload:
        return self._payload(
            "derive",
            spec,
            acquisition_digest=acquisition.digest,
            data_digest=data.digest,
        )

    def fit(self, spec: FitSpec, features: NodePayload) -> NodePayload:
        return self._payload(
            "fit",
            spec,
            feature_digest=features.digest,
            candidate_count=spec.candidate_count,
        )

    def evaluate(
        self,
        spec: EvaluationSpec,
        fit: NodePayload,
        features: NodePayload,
        data: NodePayload,
    ) -> NodePayload:
        token = content_digest(
            (spec, fit.semantic_value(), features.semantic_value(), data.semantic_value())
        )
        risk = int(token[:12], 16) / float(16**12)
        return self._payload(
            "evaluate",
            spec,
            fit_digest=fit.digest,
            feature_digest=features.digest,
            risk=risk,
            mode=spec.mode,
        )

    def compare(
        self, spec: ComparisonSpec, evaluations: Sequence[NodePayload]
    ) -> NodePayload:
        risks = [float(item.get("risk", 0.0)) for item in evaluations]
        if spec.kind.value.endswith("_gap") or spec.kind.value == "selection_regret":
            value = abs(risks[0] - risks[1])
        else:
            value = sum(risks) / len(risks)
        return self._payload(
            "compare",
            spec,
            evaluation_digests=[item.digest for item in evaluations],
            value=value,
        )
