from __future__ import annotations

from pathlib import Path

from src.experiment import ExperimentRunner, LocalArtifactStore, Stage
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
    PrefixAxis,
    RepetitionSpec,
    ResourceLimit,
    RiskRole,
)
from src.experiment.manifest import load_manifest
from src.experiment.numerical import NumericalProviderBundle


ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "experiments/empirical_evaluation/manifests/smoke.yaml"


def _canary_plan(manifest) -> ExperimentPlan:
    study = "memory_vs_lag"
    tasks = ("F_mem_0", "F_mem_1")
    pairing = "canary/pair"
    train = DataSpec(
        id="canary/data/train",
        study_id=study,
        split="train",
        trajectory_id="canary/trajectory",
        task_ids=tasks,
        sample_count=6,
        window_length=3,
        gap=0,
        input_dim=1,
        pairing_key=pairing,
    )
    test = DataSpec(
        id="canary/data/test",
        study_id=study,
        split="test",
        trajectory_id="canary/trajectory",
        task_ids=tasks,
        sample_count=4,
        window_length=3,
        gap=0,
        input_dim=1,
        pairing_key=pairing,
    )
    acquisitions = []
    views = []
    for data in (train, test):
        acquisition = AcquisitionSpec(
            id=f"canary/acquire/{data.split}",
            study_id=study,
            data_id=data.id,
            kind=AcquisitionKind.EXACT,
            split=data.split,
            prefix_limits=(ResourceLimit(PrefixAxis.R, 2),),
            fixed_resources={"n": 3, "tau_plus": 2},
            randomness_path="design/branches",
            pairing_key=pairing,
            program_key="quark",
        )
        acquisitions.append(acquisition)
        views.append(
            FeatureViewSpec(
                id=f"canary/view/{data.split}",
                study_id=study,
                acquisition_id=acquisition.id,
                data_id=data.id,
                split=data.split,
                task_ids=tasks,
                prefixes=(ResourceLimit(PrefixAxis.R, 2),),
                pairing_key=pairing,
            )
        )
    fit = FitSpec(
        id="canary/fit",
        study_id=study,
        feature_view_id=views[0].id,
        task_ids=tasks,
        readout_key="fixed_rms_matern_ivanov",
        pairing_key=pairing,
    )
    evaluations = []
    comparisons = []
    for lag, task in enumerate(tasks):
        evaluation = EvaluationSpec(
            id=f"canary/eval/{task}",
            study_id=study,
            fit_id=fit.id,
            feature_view_id=views[1].id,
            data_id=test.id,
            task_ids=(task,),
            risk_role=RiskRole.TEST,
            denominator_key=f"{pairing}/{task}",
            pairing_key=pairing,
            parameters={"tau_plus": 2, "lag": lag, "n": 3, "R": 2},
        )
        evaluations.append(evaluation)
        comparisons.append(
            ComparisonSpec(
                id=f"canary/nemse/{task}",
                study_id=study,
                kind=ComparisonKind.NEMSE,
                evaluation_ids=(evaluation.id,),
                denominator_key=evaluation.denominator_key,
                pairing_key=pairing,
                parameters={"tau_plus": 2, "lag": lag, "n": 3, "R": 2},
            )
        )
    return ExperimentPlan(
        campaign=CampaignId.REPRESENTATION,
        repetition=RepetitionSpec(0, 991),
        manifest_digest=manifest.digest,
        data=(train, test),
        acquisitions=tuple(acquisitions),
        feature_views=tuple(views),
        fits=(fit,),
        evaluations=tuple(evaluations),
        comparisons=tuple(comparisons),
    )


def test_complete_aer_numerical_canary_resumes_without_reexecution(tmp_path):
    manifest = load_manifest(SMOKE)
    plan = _canary_plan(manifest)
    bundle = NumericalProviderBundle.create(
        manifest, backend="aer", gpu_id=None, chunk_size=2
    )
    bundle.preflight(plan, study="memory_vs_lag")
    store = LocalArtifactStore(tmp_path, plan)
    runner = ExperimentRunner(
        data_provider=bundle.data,
        representation_provider=bundle.representation,
        readout_provider=bundle.readout,
        artifact_store=store,
    )
    first = runner.run(plan, study="memory_vs_lag")
    second = runner.run(plan, study="memory_vs_lag")

    assert first.completed_stage is Stage.FINALIZE
    assert first.finalization_scope == "study"
    assert first.selected_studies == ("memory_vs_lag",)
    assert second.reused_nodes == len(second.node_digests)
    assert first.node_digests == second.node_digests
    comparison_rows = LocalArtifactStore.aggregate_comparisons(
        tmp_path,
        campaign=CampaignId.REPRESENTATION.value,
        study="memory_vs_lag",
    )
    assert len(comparison_rows) == 2
    assert all(row["denominator_ddof"] == 0 for row in comparison_rows)
    assert all(row["value"] >= 0.0 for row in comparison_rows)
