from __future__ import annotations

from pathlib import Path

import pytest

from src.experiment import CampaignId, load_manifest, plan_experiment
from src.experiment.campaigns.campaign_iii import (
    SEARCH_R_GRID,
    SELECTION_N_GRID,
    SELECTION_S_GRID,
)
from src.experiment.campaigns.constants import (
    ARCHITECTURE_TIERS,
    ATLAS_TASKS,
    DEPENDENCE_GRID,
    D_GRID,
    GAP_GRID,
    H_GRID,
    L_GRID,
    M_GRID,
    N_WIDTH_GRID,
    POOL_GRID,
    R_GRID,
    SAMPLE_GRID,
    SHOT_ARCHITECTURES,
    TAU_PLUS_GRID,
    WINDOW_GRID,
)
from src.experiment.domain import PrefixAxis


ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "experiments/empirical_evaluation/manifests/smoke.yaml"


@pytest.fixture(scope="module")
def plans():
    manifest = load_manifest(SMOKE)
    return {
        campaign: plan_experiment(manifest, campaign, 0)
        for campaign in manifest.campaigns
    }


def test_campaign_i_expands_every_locked_surface_control_and_atlas_task(plans):
    plan = plans[CampaignId.REPRESENTATION]
    comparisons = {item.id for item in plan.comparisons}
    acquisitions = {item.id for item in plan.acquisitions}

    for tau in TAU_PLUS_GRID:
        for lag in L_GRID:
            assert f"memory_vs_lag/tau_plus={tau}/nemse/F_mem_{lag}" in comparisons
    for variant in ("heterogeneous", "homogeneous_center"):
        for branches in R_GRID:
            for modes in H_GRID:
                assert f"multiplex_vs_modes/{variant}/R={branches}/nemse/F_multi_{modes}" in comparisons
    for branches in R_GRID:
        for task in ("F_multi_4", "F_multi_8"):
            assert f"multiplex_vs_modes/homogeneous_selected/R={branches}/nemse/{task}" in comparisons

    for dimension in D_GRID:
        assert f"width_vs_spatial/d={dimension}/raw/nemse/F_sp_{dimension}" in comparisons
        for width in N_WIDTH_GRID:
            for method in ("projection", "quark"):
                assert f"width_vs_spatial/d={dimension}/{method}/n={width}/nemse/F_sp_{dimension}" in comparisons

    for dynamics in ("identity", "local_only", "full_cycle"):
        for bank in ("vertex_only", "complete"):
            stem = f"mixer_mechanism/{dynamics}/{bank}"
            assert f"{stem}/diagnostics" in comparisons
    assert "mixer_mechanism/cost_matched/bank_comparison" in comparisons

    atlas_methods = tuple(f"quark_{name}" for name in ARCHITECTURE_TIERS) + (
        "raw_history_matern",
        "classical_multiscale",
    )
    for method in atlas_methods:
        assert f"task_atlas/{method}/acquire/train" in acquisitions
        for task in ATLAS_TASKS:
            assert f"task_atlas/{method}/nemse/{task}" in comparisons


def test_campaign_ii_expands_every_shot_window_and_dependence_cell(plans):
    plan = plans[CampaignId.FIDELITY]
    comparisons = {item.id for item in plan.comparisons}
    task = "F_multi_8"

    for width, branches in SHOT_ARCHITECTURES:
        for shots in M_GRID:
            stem = f"shot_fidelity/n={width}/R={branches}/M={shots}/{task}"
            assert f"{stem}/shot_gap" in comparisons
            assert f"{stem}/nemse" in comparisons
    for tau in TAU_PLUS_GRID:
        for window in WINDOW_GRID:
            stem = f"window_fidelity/tau_plus={tau}/w={window}/{task}"
            assert f"{stem}/window_gap" in comparisons
    for half_life in (None,) + DEPENDENCE_GRID:
        label = "iid" if half_life is None else str(half_life)
        for gap in GAP_GRID:
            stem = f"dependence/surface/tau_dep={label}/g={gap}"
            assert f"{stem}/generalization_gap/F_mem_1" in comparisons
    for gap in GAP_GRID:
        for sample_count in SAMPLE_GRID:
            stem = f"dependence/sample_size/tau_dep=16/g={gap}/N={sample_count}"
            assert f"{stem}/generalization_gap/F_mem_1" in comparisons


def test_campaign_iii_expands_erm_oracle_regret_and_separate_cache_groups(plans):
    plan = plans[CampaignId.FROZEN_POOL]
    comparisons = {item.id for item in plan.comparisons}
    tasks = ("F_mem_1", "F_mem_64", "F_multi_8", "F_sp_64", "F_deg_16", "F_C6")

    def assert_selection_cell(stem: str, task: str) -> None:
        assert f"{stem}/{task}/selection_regret" in comparisons
        assert f"{stem}/{task}/selected_nemse" in comparisons
        assert f"{stem}/{task}/oracle_nemse" in comparisons

    for branches in SEARCH_R_GRID:
        for pool_size in POOL_GRID:
            for task in tasks:
                assert_selection_cell(f"frozen_pool/main/R={branches}/S={pool_size}", task)
    for sample_count in SELECTION_N_GRID:
        for pool_size in SELECTION_S_GRID:
            for task in ("F_multi_8", "F_C6"):
                assert_selection_cell(
                    f"frozen_pool/selection_pressure/N={sample_count}/S={pool_size}",
                    task,
                )
    for pool_size in POOL_GRID[:-2]:
        for task in tasks:
            assert_selection_cell(f"frozen_pool/finite_shot/S={pool_size}", task)

    acquisitions = {item.id: item for item in plan.acquisitions}
    main_axes = {item.axis for item in acquisitions["frozen_pool/main/acquire/train"].prefix_limits}
    pressure_axes = {
        item.axis
        for item in acquisitions["frozen_pool/selection_pressure/acquire/train"].prefix_limits
    }
    measured_axes = {
        item.axis
        for item in acquisitions["frozen_pool/finite_shot/acquire/train"].prefix_limits
    }
    assert main_axes == {PrefixAxis.R, PrefixAxis.S}
    assert pressure_axes == {PrefixAxis.N, PrefixAxis.S}
    assert measured_axes == {PrefixAxis.S}
