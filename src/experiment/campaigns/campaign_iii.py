"""Campaign III: frozen-pool search and resource economics."""

from __future__ import annotations

from ..domain import (
    AcquisitionKind,
    CampaignId,
    ComparisonKind,
    ComparisonSpec,
    EvaluationSpec,
    FitSpec,
    RiskRole,
    SelectionRule,
)
from ..manifest import RunManifest
from .base import PlanBuilder, limits
from .constants import POOL_GRID, R_GRID


SEARCH_R_GRID = R_GRID[:-1]
SELECTION_N_GRID = (256, 1024, 4096)
SELECTION_S_GRID = (1, 16, 64, 256)


def plan_campaign_iii(manifest: RunManifest, repetition):
    builder = PlanBuilder(manifest, CampaignId.FROZEN_POOL, repetition)
    _main_pool_search(builder)
    _selection_pressure(builder)
    _finite_shot_ranking(builder)
    return builder.build(
        title="Frozen-pool search and resource economics",
        candidate_prefix_max=256,
        search_branch_prefix_max=32,
        acquisition_cost="9*N*M*S*R",
    )


def _main_pool_search(builder: PlanBuilder) -> None:
    tasks = builder.manifest.pre_run.frozen_pool_tasks
    train, test, pairing = builder.add_paired_data("frozen_pool/main", tasks=tasks, input_dim=64)
    resources = {"n": 8, "tau_plus": 32, "features": "exact"}
    train_acq = builder.add_acquisition(
        "frozen_pool/main/acquire/train", data_id=train, split="train",
        kind=AcquisitionKind.EXACT, pairing_key=pairing,
        prefix_limits=limits(S=256, R=32), fixed_resources=resources,
        randomness_path="design/candidate_pool", program_key="quark_pool"
    )
    test_acq = builder.add_acquisition(
        "frozen_pool/main/acquire/test", data_id=test, split="test",
        kind=AcquisitionKind.EXACT, pairing_key=pairing,
        prefix_limits=limits(S=256, R=32), fixed_resources=resources,
        randomness_path="design/candidate_pool", program_key="quark_pool"
    )
    for branches in SEARCH_R_GRID:
        for pool_size in POOL_GRID:
            stem = f"frozen_pool/main/R={branches}/S={pool_size}"
            train_view = builder.add_view(
                f"{stem}/view/train", acquisition_id=train_acq, data_id=train,
                split="train", tasks=tasks, pairing_key=pairing,
                prefixes=limits(R=branches, S=pool_size)
            )
            test_view = builder.add_view(
                f"{stem}/view/test", acquisition_id=test_acq, data_id=test,
                split="test", tasks=tasks, pairing_key=pairing,
                prefixes=limits(R=branches, S=pool_size)
            )
            for task in tasks:
                _add_selection_cell(
                    builder, stem, train_view, test_view, test, pairing, task,
                    pool_size=pool_size, sample_count=1024, branches=branches, measured_m=None
                )


def _selection_pressure(builder: PlanBuilder) -> None:
    tasks = ("F_multi_8", "F_C6")
    train, test, pairing = builder.add_paired_data(
        "frozen_pool/selection_pressure", tasks=tasks, input_dim=64,
        sample_count=max(SELECTION_N_GRID)
    )
    resources = {"n": 8, "R": 16, "tau_plus": 32, "features": "exact"}
    train_acq = builder.add_acquisition(
        "frozen_pool/selection_pressure/acquire/train", data_id=train, split="train",
        kind=AcquisitionKind.EXACT, pairing_key=pairing,
        prefix_limits=limits(S=256, N=4096), fixed_resources=resources,
        randomness_path="design/candidate_pool", program_key="quark_pool"
    )
    test_acq = builder.add_acquisition(
        "frozen_pool/selection_pressure/acquire/test", data_id=test, split="test",
        kind=AcquisitionKind.EXACT, pairing_key=pairing,
        prefix_limits=limits(S=256), fixed_resources=resources,
        randomness_path="design/candidate_pool", program_key="quark_pool"
    )
    for sample_count in SELECTION_N_GRID:
        for pool_size in SELECTION_S_GRID:
            stem = f"frozen_pool/selection_pressure/N={sample_count}/S={pool_size}"
            train_view = builder.add_view(
                f"{stem}/view/train", acquisition_id=train_acq, data_id=train,
                split="train", tasks=tasks, pairing_key=pairing,
                prefixes=limits(N=sample_count, S=pool_size)
            )
            test_view = builder.add_view(
                f"{stem}/view/test", acquisition_id=test_acq, data_id=test,
                split="test", tasks=tasks, pairing_key=pairing,
                prefixes=limits(S=pool_size)
            )
            for task in tasks:
                _add_selection_cell(
                    builder, stem, train_view, test_view, test, pairing, task,
                    pool_size=pool_size, sample_count=sample_count, branches=16, measured_m=None
                )


def _finite_shot_ranking(builder: PlanBuilder) -> None:
    tasks = builder.manifest.pre_run.frozen_pool_tasks
    train, test, pairing = builder.add_paired_data("frozen_pool/finite_shot", tasks=tasks, input_dim=64)
    shots = builder.manifest.pre_run.campaign_iii_finite_shot_m
    resources = {"n": 8, "R": 16, "tau_plus": 32, "M": shots, "settings": 9}
    train_acq = builder.add_acquisition(
        "frozen_pool/finite_shot/acquire/train", data_id=train, split="train",
        kind=AcquisitionKind.GROUPED_MEASUREMENT, pairing_key=pairing,
        prefix_limits=limits(S=64), fixed_resources=resources,
        randomness_path="measurement/train/pool", program_key="quark_measured_pool"
    )
    test_acq = builder.add_acquisition(
        "frozen_pool/finite_shot/acquire/test", data_id=test, split="test",
        kind=AcquisitionKind.GROUPED_MEASUREMENT, pairing_key=pairing,
        prefix_limits=limits(S=64), fixed_resources=resources,
        randomness_path="measurement/heldout/pool", program_key="quark_measured_pool"
    )
    for pool_size in (1, 2, 4, 8, 16, 32, 64):
        stem = f"frozen_pool/finite_shot/S={pool_size}"
        train_view = builder.add_view(
            f"{stem}/view/train", acquisition_id=train_acq, data_id=train,
            split="train", tasks=tasks, pairing_key=pairing, prefixes=limits(S=pool_size)
        )
        test_view = builder.add_view(
            f"{stem}/view/test", acquisition_id=test_acq, data_id=test,
            split="test", tasks=tasks, pairing_key=pairing, prefixes=limits(S=pool_size)
        )
        for task in tasks:
            _add_selection_cell(
                builder, stem, train_view, test_view, test, pairing, task,
                pool_size=pool_size, sample_count=1024, branches=16, measured_m=shots
            )


def _add_selection_cell(
    builder: PlanBuilder,
    stem: str,
    train_view: str,
    test_view: str,
    test_data: str,
    pairing: str,
    task: str,
    *,
    pool_size: int,
    sample_count: int,
    branches: int,
    measured_m: int | None,
) -> None:
    node = f"{stem}/{task}"
    fit_id = f"{node}/fit_candidates"
    denominator = f"{pairing}/labels/{task}"
    rule = SelectionRule.POOL_ERM if pool_size > 1 else SelectionRule.FULL_SAMPLE_ERM
    builder.fits.append(
        FitSpec(
            id=fit_id, feature_view_id=train_view, task_ids=(task,),
            readout_key="fixed_rms_matern_ivanov", selection_rule=rule,
            candidate_count=pool_size, pairing_key=pairing,
            parameters={"fit_each_candidate_once": True, "candidate_tie": "smallest_index"}
        )
    )
    selected, oracle = f"{node}/eval/selected", f"{node}/eval/oracle"
    builder.evaluations.extend(
        (
            EvaluationSpec(
                id=selected, fit_id=fit_id, feature_view_id=test_view, data_id=test_data,
                task_ids=(task,), risk_role=RiskRole.TEST, denominator_key=denominator,
                mode="selected", pairing_key=pairing
            ),
            EvaluationSpec(
                id=oracle, fit_id=fit_id, feature_view_id=test_view, data_id=test_data,
                task_ids=(task,), risk_role=RiskRole.TEST, denominator_key=denominator,
                mode="oracle", pairing_key=pairing
            ),
        )
    )
    params = {
        "N": sample_count,
        "S": pool_size,
        "R": branches,
        "M": measured_m if measured_m is not None else 0,
        "quantum_cost": "9*N*M*S*R",
    }
    builder.comparisons.extend(
        (
            ComparisonSpec(
                id=f"{node}/selection_regret", kind=ComparisonKind.SELECTION_REGRET,
                evaluation_ids=(selected, oracle), denominator_key=denominator,
                pairing_key=pairing, parameters=params
            ),
            ComparisonSpec(
                id=f"{node}/selected_nemse", kind=ComparisonKind.NEMSE,
                evaluation_ids=(selected,), denominator_key=denominator,
                pairing_key=pairing, parameters=params
            ),
            ComparisonSpec(
                id=f"{node}/oracle_nemse", kind=ComparisonKind.NEMSE,
                evaluation_ids=(oracle,), denominator_key=denominator,
                pairing_key=pairing, parameters=params
            ),
        )
    )

