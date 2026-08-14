from __future__ import annotations

import json

import pytest

from pathlib import Path

from src.experiment import (
    CampaignId,
    ExperimentRunner,
    LocalArtifactStore,
    Stage,
    load_manifest,
    plan_experiment,
)
from src.experiment.artifact_store import ArtifactIntegrityError
from src.experiment.providers import FakeProviders

from .helpers import minimal_plan


def _runner(tmp_path, plan, providers):
    return ExperimentRunner(
        data_provider=providers,
        representation_provider=providers,
        readout_provider=providers,
        artifact_store=LocalArtifactStore(tmp_path, plan),
    )


def test_complete_run_resumes_without_provider_calls(tmp_path):
    plan = minimal_plan()
    providers = FakeProviders()
    first = _runner(tmp_path, plan, providers).run(plan)
    calls = providers.calls.copy()
    second = _runner(tmp_path, plan, providers).run(plan)
    assert first.completed_stage is Stage.FINALIZE
    assert first.plan_digest == second.plan_digest
    assert providers.calls == calls
    assert second.reused_nodes == len(second.node_digests)


def test_corruption_is_detected_and_failure_is_append_only(tmp_path):
    plan = minimal_plan()
    providers = FakeProviders()
    result = _runner(tmp_path, plan, providers).run(plan, through=Stage.PREPARE)
    payload = next(result.artifact_path.parents[2].glob("*/**/payload.json"))
    payload.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError):
        _runner(tmp_path, plan, providers).run(plan, through=Stage.PREPARE)
    failures = list((tmp_path / "failures").glob("**/status.json"))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text(encoding="utf-8"))["status"] == "failed"


def test_provider_failure_does_not_create_success_cache(tmp_path):
    class FailingProviders(FakeProviders):
        def prepare(self, spec, seeds):
            raise RuntimeError("expected provider failure")

    plan = minimal_plan("failing")
    with pytest.raises(RuntimeError, match="expected provider failure"):
        _runner(tmp_path, plan, FailingProviders()).run(plan, through=Stage.PREPARE)
    assert not list((tmp_path / "runs").glob("**/manifest.json"))
    assert len(list((tmp_path / "failures").glob("**/status.json"))) == 1



def test_runner_rejects_store_bound_to_another_plan(tmp_path):
    plan = minimal_plan("requested")
    other = minimal_plan("other")
    providers = FakeProviders()
    runner = ExperimentRunner(
        data_provider=providers,
        representation_provider=providers,
        readout_provider=providers,
        artifact_store=LocalArtifactStore(tmp_path, other),
    )
    with pytest.raises(ValueError, match="different experiment plan"):
        runner.run(plan)
    assert not providers.calls


def test_stopping_stage_does_not_change_node_identity(tmp_path):
    plan = minimal_plan("staged")
    providers = FakeProviders()
    partial = _runner(tmp_path, plan, providers).run(plan, through=Stage.PREPARE)
    prepare_calls = providers.calls["prepare"]
    complete = _runner(tmp_path, plan, providers).run(plan)

    assert providers.calls["prepare"] == prepare_calls
    assert dict(partial.node_digests).items() <= dict(complete.node_digests).items()
    assert partial.plan_digest == complete.plan_digest


def test_selected_study_executes_and_finalizes_only_its_explicit_nodes(tmp_path):
    root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(
        root / "experiments/empirical_evaluation/manifests/smoke.yaml"
    )
    plan = plan_experiment(manifest, CampaignId.REPRESENTATION, 0)
    providers = FakeProviders()
    result = _runner(tmp_path, plan, providers).run(plan, study="memory_vs_lag")
    selected_count = sum(
        item.study_id == "memory_vs_lag"
        for group in (
            plan.data,
            plan.acquisitions,
            plan.feature_views,
            plan.fits,
            plan.evaluations,
            plan.comparisons,
        )
        for item in group
    )
    assert result.finalization_scope == "study"
    assert result.selected_studies == ("memory_vs_lag",)
    assert len(result.node_digests) == selected_count + 1
    assert all(
        "multiplex_vs_modes" not in key and "task_atlas" not in key
        for key, _ in result.node_digests
    )

    calls = providers.calls.copy()
    resumed = _runner(tmp_path, plan, providers).run(plan, study="memory_vs_lag")
    assert providers.calls == calls
    assert resumed.reused_nodes == len(resumed.node_digests)


def test_aggregation_requires_completed_study_finalization(tmp_path):
    plan = minimal_plan("aggregation-completion")
    providers = FakeProviders()
    _runner(tmp_path, plan, providers).run(
        plan, through=Stage.ANALYZE, study="minimal"
    )
    assert LocalArtifactStore.aggregate_comparisons(
        tmp_path, study="minimal"
    ) == []

    _runner(tmp_path, plan, providers).run(plan, study="minimal")
    rows = LocalArtifactStore.aggregate_comparisons(tmp_path, study="minimal")
    assert len(rows) == 1
    assert rows[0]["comparison_id"] == "comparison"
