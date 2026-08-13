from __future__ import annotations

import json

import pytest

from src.experiment import ExperimentRunner, LocalArtifactStore, Stage
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
