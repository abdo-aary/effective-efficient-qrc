"""Immutable domain model for reproducible empirical-evaluation plans."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


class CampaignId(str, Enum):
    REPRESENTATION = "campaign_i"
    FIDELITY = "campaign_ii"
    FROZEN_POOL = "campaign_iii"


class Stage(str, Enum):
    PREPARE = "prepare"
    ACQUIRE = "acquire"
    ANALYZE = "analyze"
    FINALIZE = "finalize"

    @property
    def index(self) -> int:
        return tuple(Stage).index(self)


class AcquisitionKind(str, Enum):
    EXACT = "exact"
    GROUPED_MEASUREMENT = "grouped_measurement"
    CAUSAL_PROXY = "causal_proxy"
    RAW_HISTORY = "raw_history"
    PROJECTED_HISTORY = "projected_history"
    CLASSICAL_RESERVOIR = "classical_reservoir"


class PrefixAxis(str, Enum):
    R = "R"
    S = "S"
    M = "M"
    N = "N"
    WIDTH = "n"
    WINDOW = "w"


class SelectionRule(str, Enum):
    FIXED = "fixed"
    FULL_SAMPLE_ERM = "full_sample_erm"
    POOL_ERM = "pool_erm"
    HOMOGENEOUS_ERM = "homogeneous_erm"


class RiskRole(str, Enum):
    TRAIN = "train"
    HELDOUT = "heldout"
    TEST = "test"


class ComparisonKind(str, Enum):
    NEMSE = "nemse"
    SHOT_GAP = "shot_gap"
    WINDOW_GAP = "window_gap"
    GENERALIZATION_GAP = "generalization_gap"
    SELECTION_REGRET = "selection_regret"
    DIAGNOSTIC = "diagnostic"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported plan parameter value {type(value).__name__}.")


def frozen_params(value: Mapping[str, Any] | Iterable[tuple[str, Any]] = ()) -> tuple[tuple[str, Any], ...]:
    """Return deterministic, recursively immutable key/value parameters."""

    source = dict(value)
    return tuple(sorted((str(key), _freeze(item)) for key, item in source.items()))


def json_value(value: Any) -> Any:
    """Convert domain values to deterministic JSON-compatible structures."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
            return {item[0]: json_value(item[1]) for item in value}
        return [json_value(item) for item in value]
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot serialize {type(value).__name__} as experiment JSON.")


def content_digest(value: Any) -> str:
    payload = json.dumps(json_value(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RepetitionSpec:
    index: int
    root_seed: int

    def __post_init__(self) -> None:
        if int(self.index) < 0 or int(self.root_seed) < 0:
            raise ValueError("Repetition index and root seed must be nonnegative.")
        object.__setattr__(self, "index", int(self.index))
        object.__setattr__(self, "root_seed", int(self.root_seed))


@dataclass(frozen=True)
class ResourceLimit:
    axis: PrefixAxis
    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", PrefixAxis(self.axis))
        if int(self.value) < 1:
            raise ValueError("Resource values must be positive.")
        object.__setattr__(self, "value", int(self.value))


@dataclass(frozen=True)
class DataSpec:
    id: str
    study_id: str
    split: str
    trajectory_id: str
    task_ids: tuple[str, ...]
    sample_count: int
    window_length: int
    gap: int
    input_dim: int
    pairing_key: str
    dependence_half_life: float | None = None
    projection_conditioned_tasks: tuple[str, ...] = ()
    parameters: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.split not in {"train", "test"}:
            raise ValueError("Data split must be train or test.")
        if min(int(self.sample_count), int(self.window_length), int(self.input_dim)) < 1:
            raise ValueError("Data sample count, window length, and input dimension must be positive.")
        if int(self.gap) < 0 or not self.task_ids:
            raise ValueError("Data gap must be nonnegative and tasks must be nonempty.")
        if not set(self.projection_conditioned_tasks).issubset(self.task_ids):
            raise ValueError("Projection-conditioned tasks must be present in task_ids.")
        object.__setattr__(self, "task_ids", tuple(str(item) for item in self.task_ids))
        object.__setattr__(self, "projection_conditioned_tasks", tuple(self.projection_conditioned_tasks))
        object.__setattr__(self, "parameters", frozen_params(self.parameters))


@dataclass(frozen=True)
class AcquisitionSpec:
    id: str
    study_id: str
    data_id: str
    kind: AcquisitionKind
    split: str
    prefix_limits: tuple[ResourceLimit, ...]
    fixed_resources: tuple[tuple[str, Any], ...]
    randomness_path: str
    pairing_key: str
    program_key: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", AcquisitionKind(self.kind))
        limits = tuple(self.prefix_limits)
        axes = tuple(limit.axis for limit in limits)
        if len(set(axes)) != len(axes):
            raise ValueError(f"Acquisition {self.id!r} repeats a prefix axis.")
        if {PrefixAxis.R, PrefixAxis.S, PrefixAxis.M}.issubset(axes):
            raise ValueError("A maximal acquisition may not combine R, S, and M prefix axes.")
        if self.split not in {"train", "test"}:
            raise ValueError("Acquisition split must be train or test.")
        if self.kind is AcquisitionKind.GROUPED_MEASUREMENT and not self.randomness_path:
            raise ValueError("Measured acquisitions require an explicit randomness path.")
        object.__setattr__(self, "prefix_limits", limits)
        object.__setattr__(self, "fixed_resources", frozen_params(self.fixed_resources))

    def limit(self, axis: PrefixAxis) -> int | None:
        return next((item.value for item in self.prefix_limits if item.axis is axis), None)


@dataclass(frozen=True)
class FeatureViewSpec:
    id: str
    study_id: str
    acquisition_id: str
    data_id: str
    split: str
    task_ids: tuple[str, ...]
    prefixes: tuple[ResourceLimit, ...] = ()
    observable_bank: str = "complete"
    pairing_key: str = ""
    parameters: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        selections = tuple(self.prefixes)
        axes = tuple(item.axis for item in selections)
        if len(set(axes)) != len(axes):
            raise ValueError(f"Feature view {self.id!r} repeats a prefix axis.")
        object.__setattr__(self, "prefixes", selections)
        object.__setattr__(self, "task_ids", tuple(str(item) for item in self.task_ids))
        object.__setattr__(self, "parameters", frozen_params(self.parameters))


@dataclass(frozen=True)
class FitSpec:
    id: str
    study_id: str
    feature_view_id: str
    task_ids: tuple[str, ...]
    readout_key: str
    selection_rule: SelectionRule = SelectionRule.FIXED
    candidate_count: int = 1
    pairing_key: str = ""
    parameters: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "selection_rule", SelectionRule(self.selection_rule))
        if int(self.candidate_count) < 1 or not self.task_ids:
            raise ValueError("Fits require at least one candidate and task.")
        if self.selection_rule is SelectionRule.POOL_ERM and int(self.candidate_count) < 2:
            raise ValueError("Pool ERM requires at least two candidates.")
        object.__setattr__(self, "candidate_count", int(self.candidate_count))
        object.__setattr__(self, "task_ids", tuple(str(item) for item in self.task_ids))
        object.__setattr__(self, "parameters", frozen_params(self.parameters))


@dataclass(frozen=True)
class EvaluationSpec:
    id: str
    study_id: str
    fit_id: str
    feature_view_id: str
    data_id: str
    task_ids: tuple[str, ...]
    risk_role: RiskRole
    denominator_key: str
    mode: str = "predictor"
    pairing_key: str = ""
    parameters: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "risk_role", RiskRole(self.risk_role))
        object.__setattr__(self, "task_ids", tuple(str(item) for item in self.task_ids))
        object.__setattr__(self, "parameters", frozen_params(self.parameters))


@dataclass(frozen=True)
class ComparisonSpec:
    id: str
    study_id: str
    kind: ComparisonKind
    evaluation_ids: tuple[str, ...]
    denominator_key: str = ""
    pairing_key: str = ""
    parameters: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ComparisonKind(self.kind))
        if not self.evaluation_ids:
            raise ValueError("Comparisons require at least one evaluation.")
        object.__setattr__(self, "evaluation_ids", tuple(self.evaluation_ids))
        object.__setattr__(self, "parameters", frozen_params(self.parameters))


@dataclass(frozen=True)
class ExperimentPlan:
    campaign: CampaignId
    repetition: RepetitionSpec
    manifest_digest: str
    data: tuple[DataSpec, ...]
    acquisitions: tuple[AcquisitionSpec, ...]
    feature_views: tuple[FeatureViewSpec, ...]
    fits: tuple[FitSpec, ...]
    evaluations: tuple[EvaluationSpec, ...]
    comparisons: tuple[ComparisonSpec, ...]
    metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "campaign", CampaignId(self.campaign))
        object.__setattr__(self, "metadata", frozen_params(self.metadata))
        self.validate()

    @property
    def digest(self) -> str:
        return content_digest(self)

    def to_dict(self) -> dict[str, Any]:
        return json_value(self)

    @property
    def study_ids(self) -> tuple[str, ...]:
        return tuple(sorted({item.study_id for item in self.data}))

    def validate(self) -> None:
        groups = (self.data, self.acquisitions, self.feature_views, self.fits, self.evaluations, self.comparisons)
        for group in groups:
            ids = [item.id for item in group]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Plan contains duplicate IDs in {type(group[0]).__name__ if group else 'group'}.")
            if any(not item.study_id.strip() for item in group):
                raise ValueError("Every plan node requires an explicit nonempty study_id.")

        data = {item.id: item for item in self.data}
        acquisitions = {item.id: item for item in self.acquisitions}
        views = {item.id: item for item in self.feature_views}
        fits = {item.id: item for item in self.fits}
        evaluations = {item.id: item for item in self.evaluations}

        for acquisition in self.acquisitions:
            source = _require(data, acquisition.data_id, "data", acquisition.id)
            if source.split != acquisition.split:
                raise ValueError(f"Acquisition {acquisition.id!r} crosses data splits.")
            if source.pairing_key != acquisition.pairing_key:
                raise ValueError(f"Acquisition {acquisition.id!r} breaks its data pairing.")
            _same_study(acquisition, source)

        for view in self.feature_views:
            acquisition = _require(acquisitions, view.acquisition_id, "acquisition", view.id)
            source = _require(data, view.data_id, "data", view.id)
            if acquisition.data_id != view.data_id or source.split != view.split:
                raise ValueError(f"Feature view {view.id!r} does not match its acquisition data.")
            if view.pairing_key != source.pairing_key or view.pairing_key != acquisition.pairing_key:
                raise ValueError(f"Feature view {view.id!r} breaks its data pairing.")
            if not set(view.task_ids).issubset(source.task_ids):
                raise ValueError(f"Feature view {view.id!r} requests unknown tasks.")
            _same_study(view, acquisition, source)
            for selection in view.prefixes:
                limit = acquisition.limit(selection.axis)
                if limit is None or selection.value > limit:
                    raise ValueError(
                        f"Feature view {view.id!r} selects {selection.axis.value}={selection.value} "
                        f"outside acquisition {acquisition.id!r}."
                    )

        for fit in self.fits:
            view = _require(views, fit.feature_view_id, "feature view", fit.id)
            source = data[view.data_id]
            if source.split != "train" or view.split != "train":
                raise ValueError(f"Fit {fit.id!r} depends on non-training data.")
            if fit.pairing_key != view.pairing_key:
                raise ValueError(f"Fit {fit.id!r} breaks its feature pairing.")
            if not set(fit.task_ids).issubset(view.task_ids):
                raise ValueError(f"Fit {fit.id!r} requests tasks missing from its feature view.")
            _same_study(fit, view, source)
            pool_prefix = next((item.value for item in view.prefixes if item.axis is PrefixAxis.S), None)
            if pool_prefix is not None and fit.candidate_count != pool_prefix:
                raise ValueError(f"Fit {fit.id!r} does not use its complete candidate prefix.")

        for evaluation in self.evaluations:
            fit = _require(fits, evaluation.fit_id, "fit", evaluation.id)
            view = _require(views, evaluation.feature_view_id, "feature view", evaluation.id)
            source = _require(data, evaluation.data_id, "data", evaluation.id)
            if view.data_id != source.id or not set(evaluation.task_ids).issubset(view.task_ids):
                raise ValueError(f"Evaluation {evaluation.id!r} does not match its data or tasks.")
            if (
                evaluation.pairing_key != fit.pairing_key
                or evaluation.pairing_key != view.pairing_key
                or evaluation.pairing_key != source.pairing_key
            ):
                raise ValueError(f"Evaluation {evaluation.id!r} breaks its experiment pairing.")
            if not set(evaluation.task_ids).issubset(fit.task_ids):
                raise ValueError(f"Evaluation {evaluation.id!r} requests tasks not fitted by {fit.id!r}.")
            _same_study(evaluation, fit, view, source)
            expected_split = "train" if evaluation.risk_role is RiskRole.TRAIN else "test"
            if source.split != expected_split:
                raise ValueError(f"Evaluation {evaluation.id!r} uses the wrong data split.")

        for comparison in self.comparisons:
            members = [_require(evaluations, item, "evaluation", comparison.id) for item in comparison.evaluation_ids]
            _same_study(comparison, *members)
            if comparison.denominator_key and any(
                member.denominator_key != comparison.denominator_key for member in members
            ):
                raise ValueError(f"Comparison {comparison.id!r} does not share one denominator.")
            if comparison.pairing_key and any(member.pairing_key != comparison.pairing_key for member in members):
                raise ValueError(f"Comparison {comparison.id!r} combines unpaired evaluations.")
            if comparison.kind in {ComparisonKind.SHOT_GAP, ComparisonKind.WINDOW_GAP, ComparisonKind.GENERALIZATION_GAP}:
                if len(members) != 2 or len({member.fit_id for member in members}) != 1:
                    raise ValueError(f"{comparison.kind.value} must evaluate the same frozen fit twice.")
            if comparison.kind is ComparisonKind.SHOT_GAP:
                kinds = {acquisitions[views[item.feature_view_id].acquisition_id].kind for item in members}
                if kinds != {AcquisitionKind.GROUPED_MEASUREMENT, AcquisitionKind.EXACT}:
                    raise ValueError("Shot gaps require measured and exact held-out features.")
                measured = next(item for item in members if acquisitions[views[item.feature_view_id].acquisition_id].kind is AcquisitionKind.GROUPED_MEASUREMENT)
                train_view = views[fits[measured.fit_id].feature_view_id]
                train_acquisition = acquisitions[train_view.acquisition_id]
                heldout_acquisition = acquisitions[views[measured.feature_view_id].acquisition_id]
                if train_acquisition.kind is not AcquisitionKind.GROUPED_MEASUREMENT:
                    raise ValueError("Shot-gap fits require measured training features.")
                if train_acquisition.randomness_path == heldout_acquisition.randomness_path:
                    raise ValueError("Shot-gap training and held-out measurements must be independent.")
                train_shots = next((item.value for item in train_view.prefixes if item.axis is PrefixAxis.M), None)
                heldout_view = views[measured.feature_view_id]
                heldout_shots = next((item.value for item in heldout_view.prefixes if item.axis is PrefixAxis.M), None)
                if train_shots != heldout_shots:
                    raise ValueError("Shot-gap evaluations must share one shot prefix.")
            if comparison.kind is ComparisonKind.WINDOW_GAP:
                kinds = {acquisitions[views[item.feature_view_id].acquisition_id].kind for item in members}
                if kinds != {AcquisitionKind.EXACT, AcquisitionKind.CAUSAL_PROXY}:
                    raise ValueError("Window gaps require finite-window exact and causal-proxy features.")
            if comparison.kind is ComparisonKind.GENERALIZATION_GAP:
                if {item.risk_role for item in members} != {RiskRole.TRAIN, RiskRole.TEST}:
                    raise ValueError("Generalization gaps require training and independent-test risks.")
            if comparison.kind is ComparisonKind.SELECTION_REGRET:
                if len(members) != 2 or {item.mode for item in members} != {"selected", "oracle"}:
                    raise ValueError("Selection regret requires selected and oracle evaluations.")
                if len({item.fit_id for item in members}) != 1:
                    raise ValueError("Selected and oracle risks must use one frozen candidate fit collection.")


def _same_study(owner: Any, *dependencies: Any) -> None:
    if any(item.study_id != owner.study_id for item in dependencies):
        raise ValueError(
            f"Node {owner.id!r} in study {owner.study_id!r} has a cross-study dependency."
        )


def _require(values: Mapping[str, Any], key: str, kind: str, owner: str) -> Any:
    try:
        return values[key]
    except KeyError as exc:
        raise ValueError(f"{owner!r} references unknown {kind} {key!r}.") from exc

