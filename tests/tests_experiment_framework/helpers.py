from __future__ import annotations

from src.experiment.domain import (
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
    RepetitionSpec,
    RiskRole,
)


def minimal_plan(manifest_digest: str = "manifest") -> ExperimentPlan:
    pairing = "test/pair"
    tasks = ("task",)
    train = DataSpec(
        id="data/train",
        split="train",
        trajectory_id="trajectory",
        task_ids=tasks,
        sample_count=8,
        window_length=4,
        gap=0,
        input_dim=1,
        pairing_key=pairing,
    )
    test = DataSpec(
        id="data/test",
        split="test",
        trajectory_id="trajectory",
        task_ids=tasks,
        sample_count=4,
        window_length=4,
        gap=0,
        input_dim=1,
        pairing_key=pairing,
    )
    train_acq = AcquisitionSpec(
        id="acquire/train",
        data_id=train.id,
        kind=AcquisitionKind.EXACT,
        split="train",
        prefix_limits=(),
        fixed_resources={"n": 2, "R": 1},
        randomness_path="design",
        pairing_key=pairing,
        program_key="fake",
    )
    test_acq = AcquisitionSpec(
        id="acquire/test",
        data_id=test.id,
        kind=AcquisitionKind.EXACT,
        split="test",
        prefix_limits=(),
        fixed_resources={"n": 2, "R": 1},
        randomness_path="design",
        pairing_key=pairing,
        program_key="fake",
    )
    train_view = FeatureViewSpec(
        id="view/train",
        acquisition_id=train_acq.id,
        data_id=train.id,
        split="train",
        task_ids=tasks,
        pairing_key=pairing,
    )
    test_view = FeatureViewSpec(
        id="view/test",
        acquisition_id=test_acq.id,
        data_id=test.id,
        split="test",
        task_ids=tasks,
        pairing_key=pairing,
    )
    fit = FitSpec(
        id="fit",
        feature_view_id=train_view.id,
        task_ids=tasks,
        readout_key="fake",
        pairing_key=pairing,
    )
    evaluation = EvaluationSpec(
        id="evaluation",
        fit_id=fit.id,
        feature_view_id=test_view.id,
        data_id=test.id,
        task_ids=tasks,
        risk_role=RiskRole.TEST,
        denominator_key="test/pair/labels/task",
        pairing_key=pairing,
    )
    comparison = ComparisonSpec(
        id="comparison",
        kind=ComparisonKind.NEMSE,
        evaluation_ids=(evaluation.id,),
        denominator_key=evaluation.denominator_key,
        pairing_key=pairing,
    )
    return ExperimentPlan(
        campaign=CampaignId.REPRESENTATION,
        repetition=RepetitionSpec(0, 42),
        manifest_digest=manifest_digest,
        data=(train, test),
        acquisitions=(train_acq, test_acq),
        feature_views=(train_view, test_view),
        fits=(fit,),
        evaluations=(evaluation,),
        comparisons=(comparison,),
    )

