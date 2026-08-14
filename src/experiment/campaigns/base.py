"""Shared construction helpers for deterministic campaign planners."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..domain import (
    AcquisitionKind,
    AcquisitionSpec,
    CampaignId,
    ComparisonKind,
    ComparisonSpec,
    DataSpec,
    EvaluationSpec,
    ExperimentPlan,
    FeatureViewSpec,
    FitSpec,
    PrefixAxis,
    RepetitionSpec,
    ResourceLimit,
    RiskRole,
    SelectionRule,
)
from ..manifest import RunManifest


def limits(**values: int) -> tuple[ResourceLimit, ...]:
    return tuple(ResourceLimit(PrefixAxis(key), value) for key, value in values.items())


@dataclass
class PlanBuilder:
    manifest: RunManifest
    campaign: CampaignId
    repetition: RepetitionSpec
    data: list[DataSpec] = field(default_factory=list)
    acquisitions: list[AcquisitionSpec] = field(default_factory=list)
    views: list[FeatureViewSpec] = field(default_factory=list)
    fits: list[FitSpec] = field(default_factory=list)
    evaluations: list[EvaluationSpec] = field(default_factory=list)
    comparisons: list[ComparisonSpec] = field(default_factory=list)

    def add_paired_data(
        self,
        node_stem: str,
        *,
        study_id: str,
        tasks: Sequence[str],
        input_dim: int,
        sample_count: int = 1024,
        window_length: int = 512,
        gap: int = 0,
        dependence_half_life: float | None = None,
        projection_conditioned_tasks: Sequence[str] = (),
        parameters: Mapping[str, Any] | None = None,
    ) -> tuple[str, str, str]:
        pairing = f"{self.campaign.value}/{node_stem}/rep={self.repetition.index}"
        train_id, test_id = f"{node_stem}/data/train", f"{node_stem}/data/test"
        common = dict(
            study_id=study_id,
            trajectory_id=f"{node_stem}/trajectory",
            task_ids=tuple(tasks),
            window_length=int(window_length),
            gap=int(gap),
            input_dim=int(input_dim),
            pairing_key=pairing,
            dependence_half_life=dependence_half_life,
            projection_conditioned_tasks=tuple(projection_conditioned_tasks),
            parameters=parameters or {},
        )
        self.data.append(DataSpec(id=train_id, split="train", sample_count=int(sample_count), **common))
        self.data.append(
            DataSpec(
                id=test_id,
                split="test",
                sample_count=int(self.manifest.pre_run.heldout_test_size),
                **common,
            )
        )
        return train_id, test_id, pairing

    def add_acquisition(
        self,
        node_id: str,
        *,
        data_id: str,
        split: str,
        kind: AcquisitionKind,
        pairing_key: str,
        prefix_limits: tuple[ResourceLimit, ...] = (),
        fixed_resources: Mapping[str, Any] | None = None,
        randomness_path: str = "",
        program_key: str = "quark",
    ) -> str:
        study_id = next(item.study_id for item in self.data if item.id == data_id)
        self.acquisitions.append(
            AcquisitionSpec(
                id=node_id,
                study_id=study_id,
                data_id=data_id,
                kind=kind,
                split=split,
                prefix_limits=prefix_limits,
                fixed_resources=fixed_resources or {},
                randomness_path=randomness_path,
                pairing_key=pairing_key,
                program_key=program_key,
            )
        )
        return node_id

    def add_view(
        self,
        node_id: str,
        *,
        acquisition_id: str,
        data_id: str,
        split: str,
        tasks: Sequence[str],
        pairing_key: str,
        prefixes: tuple[ResourceLimit, ...] = (),
        observable_bank: str = "complete",
        parameters: Mapping[str, Any] | None = None,
    ) -> str:
        study_id = next(
            item.study_id for item in self.acquisitions if item.id == acquisition_id
        )
        self.views.append(
            FeatureViewSpec(
                id=node_id,
                study_id=study_id,
                acquisition_id=acquisition_id,
                data_id=data_id,
                split=split,
                task_ids=tuple(tasks),
                prefixes=prefixes,
                observable_bank=observable_bank,
                pairing_key=pairing_key,
                parameters=parameters or {},
            )
        )
        return node_id

    def add_predictive_pair(
        self,
        node_id: str,
        *,
        train_view: str,
        test_view: str,
        tasks: Sequence[str],
        pairing_key: str,
        selection_rule: SelectionRule = SelectionRule.FIXED,
        candidate_count: int = 1,
        fit_parameters: Mapping[str, Any] | None = None,
    ) -> tuple[str, ...]:
        evaluation_ids: list[str] = []
        train_feature = next(view for view in self.views if view.id == train_view)
        test_feature = next(view for view in self.views if view.id == test_view)
        if train_feature.study_id != test_feature.study_id:
            raise ValueError("Predictive pairs cannot cross studies.")
        study_id = train_feature.study_id
        test_data_id = test_feature.data_id
        for task in tasks:
            fit_id = f"{node_id}/fit/{task}"
            eval_id = f"{node_id}/eval/{task}"
            denominator = f"{pairing_key}/labels/{task}"
            self.fits.append(
                FitSpec(
                    id=fit_id,
                    study_id=study_id,
                    feature_view_id=train_view,
                    task_ids=(task,),
                    readout_key="fixed_rms_matern_ivanov",
                    selection_rule=selection_rule,
                    candidate_count=candidate_count,
                    pairing_key=pairing_key,
                    parameters=fit_parameters or {},
                )
            )
            self.evaluations.append(
                EvaluationSpec(
                    id=eval_id,
                    study_id=study_id,
                    fit_id=fit_id,
                    feature_view_id=test_view,
                    data_id=test_data_id,
                    task_ids=(task,),
                    risk_role=RiskRole.TEST,
                    denominator_key=denominator,
                    pairing_key=pairing_key,
                )
            )
            self.comparisons.append(
                ComparisonSpec(
                    id=f"{node_id}/nemse/{task}",
                    study_id=study_id,
                    kind=ComparisonKind.NEMSE,
                    evaluation_ids=(eval_id,),
                    denominator_key=denominator,
                    pairing_key=pairing_key,
                )
            )
            evaluation_ids.append(eval_id)
        return tuple(evaluation_ids)

    def build(self, **metadata: Any) -> ExperimentPlan:
        return ExperimentPlan(
            campaign=self.campaign,
            repetition=self.repetition,
            manifest_digest=self.manifest.digest,
            data=tuple(self.data),
            acquisitions=tuple(self.acquisitions),
            feature_views=tuple(self.views),
            fits=tuple(self.fits),
            evaluations=tuple(self.evaluations),
            comparisons=tuple(self.comparisons),
            metadata=metadata,
        )

