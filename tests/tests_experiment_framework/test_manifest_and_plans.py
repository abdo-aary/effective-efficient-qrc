from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.experiment import CampaignId, load_manifest, plan_experiment
from src.experiment.campaigns.constants import ATLAS_TASKS, N_WIDTH_GRID
from src.experiment.domain import PrefixAxis
from src.experiment.manifest import load_contract


ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "experiments/empirical_evaluation/manifests/smoke.yaml"
PRODUCTION = ROOT / "experiments/empirical_evaluation/manifests/production.template.yaml"


def test_contract_and_smoke_manifest_are_strict_and_complete():
    contract = load_contract()
    manifest = load_manifest(SMOKE)
    assert contract.version == "quark-tmlr-2026-08-13"
    assert contract.campaign_fixed_axes["width_vs_spatial"]["n"] == list(N_WIDTH_GRID)
    production_keys = set(yaml.safe_load(PRODUCTION.read_text(encoding="utf-8"))["pre_run"])
    smoke_keys = set(yaml.safe_load(SMOKE.read_text(encoding="utf-8"))["pre_run"])
    assert set(contract.required_keys) == production_keys
    assert set(contract.required_keys) == smoke_keys
    assert len(ATLAS_TASKS) == 36
    assert manifest.contract_digest == contract.digest
    assert manifest.digest == load_manifest(SMOKE).digest


def test_unresolved_production_template_is_rejected():
    with pytest.raises(ValueError, match="REQUIRED"):
        load_manifest(PRODUCTION)


def test_unknown_manifest_field_is_rejected(tmp_path):
    payload = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_manifest(path)


def test_all_campaigns_expand_deterministically_with_campaign_local_caches():
    manifest = load_manifest(SMOKE)
    plans = {
        campaign: plan_experiment(manifest, campaign, 0)
        for campaign in manifest.campaigns
    }
    assert plans[CampaignId.REPRESENTATION].digest == plan_experiment(
        manifest, CampaignId.REPRESENTATION, 0
    ).digest
    assert len(plans[CampaignId.FROZEN_POOL].acquisitions) == 6
    assert any("task_atlas/quark_small" in item.id for item in plans[CampaignId.REPRESENTATION].acquisitions)
    assert any(item.kind.value == "shot_gap" for item in plans[CampaignId.FIDELITY].comparisons)
    assert any(item.kind.value == "selection_regret" for item in plans[CampaignId.FROZEN_POOL].comparisons)
    for plan in plans.values():
        for acquisition in plan.acquisitions:
            axes = {item.axis for item in acquisition.prefix_limits}
            assert not {PrefixAxis.R, PrefixAxis.S, PrefixAxis.M}.issubset(axes)


def test_artifact_root_does_not_change_manifest_identity(tmp_path):
    payload = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(yaml.safe_dump(payload), encoding="utf-8")
    payload["artifact_root"] = str(tmp_path / "elsewhere")
    second.write_text(yaml.safe_dump(payload), encoding="utf-8")
    assert load_manifest(first).digest == load_manifest(second).digest



def test_negative_repetition_index_is_rejected():
    manifest = load_manifest(SMOKE)
    with pytest.raises(ValueError, match="nonnegative"):
        plan_experiment(manifest, CampaignId.REPRESENTATION, -1)
