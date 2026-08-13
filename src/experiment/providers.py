"""Behavioral protocols and deterministic fake providers for experiment plans."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Protocol, Sequence

from .domain import (
    AcquisitionSpec,
    ComparisonSpec,
    DataSpec,
    EvaluationSpec,
    FeatureViewSpec,
    FitSpec,
    content_digest,
)
from .seeding import PathSeedTree


Payload = Mapping[str, Any]


class DataProvider(Protocol):
    def prepare(self, spec: DataSpec, seeds: PathSeedTree) -> Payload:
        ...


class RepresentationProvider(Protocol):
    def acquire(self, spec: AcquisitionSpec, data: Payload, seeds: PathSeedTree) -> Payload:
        ...

    def derive(self, spec: FeatureViewSpec, acquisition: Payload, data: Payload) -> Payload:
        ...


class ReadoutProvider(Protocol):
    def fit(self, spec: FitSpec, features: Payload) -> Payload:
        ...

    def evaluate(self, spec: EvaluationSpec, fit: Payload, features: Payload, data: Payload) -> Payload:
        ...

    def compare(self, spec: ComparisonSpec, evaluations: Sequence[Payload]) -> Payload:
        ...


class FakeProviders(DataProvider, RepresentationProvider, ReadoutProvider):
    """Fast deterministic provider used to validate orchestration without numerics."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()

    def _payload(self, operation: str, spec: object, **extra: Any) -> dict[str, Any]:
        self.calls[operation] += 1
        return {
            "provider": "fake",
            "operation": operation,
            "spec_digest": content_digest(spec),
            **extra,
        }

    def prepare(self, spec: DataSpec, seeds: PathSeedTree) -> Payload:
        return self._payload(
            "prepare",
            spec,
            seed=seeds.integer(f"data/{spec.split}/{spec.trajectory_id}"),
            sample_count=spec.sample_count,
            task_ids=list(spec.task_ids),
        )

    def acquire(self, spec: AcquisitionSpec, data: Payload, seeds: PathSeedTree) -> Payload:
        path = spec.randomness_path or f"representation/{spec.id}"
        return self._payload(
            "acquire",
            spec,
            seed=seeds.integer(path),
            data_digest=content_digest(data),
            kind=spec.kind.value,
        )

    def derive(self, spec: FeatureViewSpec, acquisition: Payload, data: Payload) -> Payload:
        return self._payload(
            "derive",
            spec,
            acquisition_digest=content_digest(acquisition),
            data_digest=content_digest(data),
        )

    def fit(self, spec: FitSpec, features: Payload) -> Payload:
        return self._payload(
            "fit",
            spec,
            feature_digest=content_digest(features),
            candidate_count=spec.candidate_count,
        )

    def evaluate(self, spec: EvaluationSpec, fit: Payload, features: Payload, data: Payload) -> Payload:
        token = content_digest((spec, fit, features, data))
        risk = int(token[:12], 16) / float(16**12)
        return self._payload(
            "evaluate",
            spec,
            fit_digest=content_digest(fit),
            feature_digest=content_digest(features),
            risk=risk,
            mode=spec.mode,
        )

    def compare(self, spec: ComparisonSpec, evaluations: Sequence[Payload]) -> Payload:
        risks = [float(item.get("risk", 0.0)) for item in evaluations]
        if spec.kind.value.endswith("_gap") or spec.kind.value == "selection_regret":
            value = abs(risks[0] - risks[1])
        else:
            value = sum(risks) / len(risks)
        return self._payload(
            "compare",
            spec,
            evaluation_digests=[content_digest(item) for item in evaluations],
            value=value,
        )

