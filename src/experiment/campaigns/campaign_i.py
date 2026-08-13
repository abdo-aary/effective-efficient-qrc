"""Campaign I: representational phase diagrams and controlled task atlas."""

from __future__ import annotations

from ..domain import (
    AcquisitionKind,
    CampaignId,
    ComparisonKind,
    ComparisonSpec,
    SelectionRule,
)
from ..manifest import RunManifest
from .base import PlanBuilder, limits
from .constants import (
    ARCHITECTURE_TIERS,
    ATLAS_TASKS,
    D_GRID,
    H_GRID,
    L_GRID,
    N_WIDTH_GRID,
    R_GRID,
    TAU_PLUS_GRID,
)


def plan_campaign_i(manifest: RunManifest, repetition):
    builder = PlanBuilder(manifest, CampaignId.REPRESENTATION, repetition)
    _memory_vs_lag(builder)
    _multiplex_vs_modes(builder)
    _width_vs_spatial(builder)
    _mixer_mechanism(builder)
    _task_atlas(builder)
    return builder.build(
        title="Representational phase diagram and task atlas",
        maximal_cache_policy="campaign-local",
        branch_prefix_max=64,
    )


def _memory_vs_lag(builder: PlanBuilder) -> None:
    tasks = tuple(f"F_mem_{lag}" for lag in L_GRID)
    train, test, pairing = builder.add_paired_data("memory_vs_lag", tasks=tasks, input_dim=1)
    for tau in TAU_PLUS_GRID:
        stem = f"memory_vs_lag/tau_plus={tau}"
        train_acq = builder.add_acquisition(
            f"{stem}/acquire/train",
            data_id=train,
            split="train",
            kind=AcquisitionKind.EXACT,
            pairing_key=pairing,
            prefix_limits=limits(R=64),
            fixed_resources={"n": 8, "tau_plus": tau},
            randomness_path="design/branches",
        )
        test_acq = builder.add_acquisition(
            f"{stem}/acquire/test",
            data_id=test,
            split="test",
            kind=AcquisitionKind.EXACT,
            pairing_key=pairing,
            prefix_limits=limits(R=64),
            fixed_resources={"n": 8, "tau_plus": tau},
            randomness_path="design/branches",
        )
        train_view = builder.add_view(
            f"{stem}/view/train/R=16",
            acquisition_id=train_acq,
            data_id=train,
            split="train",
            tasks=tasks,
            pairing_key=pairing,
            prefixes=limits(R=16),
        )
        test_view = builder.add_view(
            f"{stem}/view/test/R=16",
            acquisition_id=test_acq,
            data_id=test,
            split="test",
            tasks=tasks,
            pairing_key=pairing,
            prefixes=limits(R=16),
        )
        builder.add_predictive_pair(stem, train_view=train_view, test_view=test_view, tasks=tasks, pairing_key=pairing)


def _multiplex_vs_modes(builder: PlanBuilder) -> None:
    tasks = tuple(f"F_multi_{modes}" for modes in H_GRID)
    train, test, pairing = builder.add_paired_data("multiplex_vs_modes", tasks=tasks, input_dim=1)
    variants = {
        "heterogeneous": ("quark", {"n": 8, "tau_plus": 32}),
        "homogeneous_center": (
            "homogeneous_memory",
            {"n": 8, "tau_plus": 32, "common_rate": "geometric_center"},
        ),
    }
    for variant, (program, resources) in variants.items():
        stem = f"multiplex_vs_modes/{variant}"
        train_acq = builder.add_acquisition(
            f"{stem}/acquire/train",
            data_id=train,
            split="train",
            kind=AcquisitionKind.EXACT,
            pairing_key=pairing,
            prefix_limits=limits(R=64),
            fixed_resources=resources,
            randomness_path="design/branches",
            program_key=program,
        )
        test_acq = builder.add_acquisition(
            f"{stem}/acquire/test",
            data_id=test,
            split="test",
            kind=AcquisitionKind.EXACT,
            pairing_key=pairing,
            prefix_limits=limits(R=64),
            fixed_resources=resources,
            randomness_path="design/branches",
            program_key=program,
        )
        for branches in R_GRID:
            train_view = builder.add_view(
                f"{stem}/view/train/R={branches}",
                acquisition_id=train_acq,
                data_id=train,
                split="train",
                tasks=tasks,
                pairing_key=pairing,
                prefixes=limits(R=branches),
            )
            test_view = builder.add_view(
                f"{stem}/view/test/R={branches}",
                acquisition_id=test_acq,
                data_id=test,
                split="test",
                tasks=tasks,
                pairing_key=pairing,
                prefixes=limits(R=branches),
            )
            builder.add_predictive_pair(
                f"{stem}/R={branches}",
                train_view=train_view,
                test_view=test_view,
                tasks=tasks,
                pairing_key=pairing,
            )

    selected_tasks = builder.manifest.pre_run.homogeneous_memory_slices
    stem = "multiplex_vs_modes/homogeneous_selected"
    resources = {"n": 8, "tau_plus_candidates": TAU_PLUS_GRID}
    train_acq = builder.add_acquisition(
        f"{stem}/acquire/train",
        data_id=train,
        split="train",
        kind=AcquisitionKind.EXACT,
        pairing_key=pairing,
        prefix_limits=limits(R=64),
        fixed_resources=resources,
        randomness_path="design/branches",
        program_key="homogeneous_memory_pool",
    )
    test_acq = builder.add_acquisition(
        f"{stem}/acquire/test",
        data_id=test,
        split="test",
        kind=AcquisitionKind.EXACT,
        pairing_key=pairing,
        prefix_limits=limits(R=64),
        fixed_resources=resources,
        randomness_path="design/branches",
        program_key="homogeneous_memory_pool",
    )
    for branches in R_GRID:
        train_view = builder.add_view(
            f"{stem}/view/train/R={branches}", acquisition_id=train_acq, data_id=train,
            split="train", tasks=selected_tasks, pairing_key=pairing, prefixes=limits(R=branches)
        )
        test_view = builder.add_view(
            f"{stem}/view/test/R={branches}", acquisition_id=test_acq, data_id=test,
            split="test", tasks=selected_tasks, pairing_key=pairing, prefixes=limits(R=branches)
        )
        builder.add_predictive_pair(
            f"{stem}/R={branches}", train_view=train_view, test_view=test_view,
            tasks=selected_tasks, pairing_key=pairing,
            selection_rule=SelectionRule.HOMOGENEOUS_ERM,
            candidate_count=len(TAU_PLUS_GRID),
        )


def _width_vs_spatial(builder: PlanBuilder) -> None:
    for dimension in D_GRID:
        task = f"F_sp_{dimension}"
        study = f"width_vs_spatial/d={dimension}"
        train, test, pairing = builder.add_paired_data(study, tasks=(task,), input_dim=dimension)
        raw_train = builder.add_acquisition(
            f"{study}/raw/acquire/train", data_id=train, split="train",
            kind=AcquisitionKind.RAW_HISTORY, pairing_key=pairing,
            fixed_resources={"w": 512}, randomness_path="data/train", program_key="raw_history"
        )
        raw_test = builder.add_acquisition(
            f"{study}/raw/acquire/test", data_id=test, split="test",
            kind=AcquisitionKind.RAW_HISTORY, pairing_key=pairing,
            fixed_resources={"w": 512}, randomness_path="data/test", program_key="raw_history"
        )
        raw_train_view = builder.add_view(
            f"{study}/raw/view/train", acquisition_id=raw_train, data_id=train,
            split="train", tasks=(task,), pairing_key=pairing
        )
        raw_test_view = builder.add_view(
            f"{study}/raw/view/test", acquisition_id=raw_test, data_id=test,
            split="test", tasks=(task,), pairing_key=pairing
        )
        builder.add_predictive_pair(
            f"{study}/raw", train_view=raw_train_view, test_view=raw_test_view,
            tasks=(task,), pairing_key=pairing
        )
        for width in N_WIDTH_GRID:
            for method, kind, program in (
                ("projection", AcquisitionKind.PROJECTED_HISTORY, "projection_only"),
                ("quark", AcquisitionKind.EXACT, "quark"),
            ):
                stem = f"{study}/{method}/n={width}"
                resources = {"n": width, "R": 16, "tau_plus": 32, "gaussian_master_width": 12}
                train_acq = builder.add_acquisition(
                    f"{stem}/acquire/train", data_id=train, split="train", kind=kind,
                    pairing_key=pairing, fixed_resources=resources,
                    randomness_path="projection", program_key=program
                )
                test_acq = builder.add_acquisition(
                    f"{stem}/acquire/test", data_id=test, split="test", kind=kind,
                    pairing_key=pairing, fixed_resources=resources,
                    randomness_path="projection", program_key=program
                )
                train_view = builder.add_view(
                    f"{stem}/view/train", acquisition_id=train_acq, data_id=train,
                    split="train", tasks=(task,), pairing_key=pairing
                )
                test_view = builder.add_view(
                    f"{stem}/view/test", acquisition_id=test_acq, data_id=test,
                    split="test", tasks=(task,), pairing_key=pairing
                )
                builder.add_predictive_pair(
                    stem, train_view=train_view, test_view=test_view, tasks=(task,), pairing_key=pairing
                )

    task = "F_sp_64"
    train = "width_vs_spatial/d=64/data/train"
    test = "width_vs_spatial/d=64/data/test"
    pairing = next(item.pairing_key for item in builder.data if item.id == train)
    allocations = ((3, 16), (6, 8), (12, 4), (3, 32), (6, 16), (12, 8), (3, 64), (6, 32), (12, 16))
    for width, branches in allocations:
        stem = f"width_vs_spatial/constant_Rn/n={width}/R={branches}"
        resources = {"n": width, "R": branches, "tau_plus": 32, "Rn": width * branches}
        train_acq = builder.add_acquisition(
            f"{stem}/acquire/train", data_id=train, split="train", kind=AcquisitionKind.EXACT,
            pairing_key=pairing, fixed_resources=resources, randomness_path="design/branches"
        )
        test_acq = builder.add_acquisition(
            f"{stem}/acquire/test", data_id=test, split="test", kind=AcquisitionKind.EXACT,
            pairing_key=pairing, fixed_resources=resources, randomness_path="design/branches"
        )
        train_view = builder.add_view(
            f"{stem}/view/train", acquisition_id=train_acq, data_id=train, split="train",
            tasks=(task,), pairing_key=pairing
        )
        test_view = builder.add_view(
            f"{stem}/view/test", acquisition_id=test_acq, data_id=test, split="test",
            tasks=(task,), pairing_key=pairing
        )
        builder.add_predictive_pair(stem, train_view=train_view, test_view=test_view, tasks=(task,), pairing_key=pairing)


def _mixer_mechanism(builder: PlanBuilder) -> None:
    conditioned = ("mix_1", "mix_2", "mix_3")
    independent = ("F_mem_8", "F_int_16", "F_C6")
    tasks = conditioned + independent
    train, test, pairing = builder.add_paired_data(
        "mixer_mechanism", tasks=tasks, input_dim=16,
        projection_conditioned_tasks=conditioned
    )
    for dynamics in ("identity", "local_only", "full_cycle"):
        for bank in ("vertex_only", "complete"):
            stem = f"mixer_mechanism/{dynamics}/{bank}"
            resources = {"n": 8, "R": 8, "tau_plus": 8, "dynamics": dynamics, "bank": bank}
            train_acq = builder.add_acquisition(
                f"{stem}/acquire/train", data_id=train, split="train", kind=AcquisitionKind.EXACT,
                pairing_key=pairing, fixed_resources=resources,
                randomness_path="design/mixer", program_key=f"mixer_{dynamics}"
            )
            test_acq = builder.add_acquisition(
                f"{stem}/acquire/test", data_id=test, split="test", kind=AcquisitionKind.EXACT,
                pairing_key=pairing, fixed_resources=resources,
                randomness_path="design/mixer", program_key=f"mixer_{dynamics}"
            )
            train_view = builder.add_view(
                f"{stem}/view/train", acquisition_id=train_acq, data_id=train, split="train",
                tasks=tasks, pairing_key=pairing, observable_bank=bank
            )
            test_view = builder.add_view(
                f"{stem}/view/test", acquisition_id=test_acq, data_id=test, split="test",
                tasks=tasks, pairing_key=pairing, observable_bank=bank
            )
            evals = builder.add_predictive_pair(
                stem, train_view=train_view, test_view=test_view, tasks=tasks, pairing_key=pairing
            )
            builder.comparisons.append(
                ComparisonSpec(
                    id=f"{stem}/diagnostics",
                    kind=ComparisonKind.DIAGNOSTIC,
                    evaluation_ids=(evals[0],),
                    denominator_key=f"{pairing}/labels/{tasks[0]}",
                    pairing_key=pairing,
                    parameters={"metrics": ("pauli_spread", "stable_rank")},
                )
            )

    task = "F_C6"
    edge_m = builder.manifest.pre_run.mixer_cost_match_m_edge
    cost_evaluations: list[str] = []
    for bank, repetitions in (("complete", edge_m), ("vertex_only", 3 * edge_m)):
        stem = f"mixer_mechanism/cost_matched/{bank}"
        resources = {"n": 8, "R": 8, "tau_plus": 8, "M": repetitions, "bank": bank}
        train_acq = builder.add_acquisition(
            f"{stem}/acquire/train", data_id=train, split="train", kind=AcquisitionKind.GROUPED_MEASUREMENT,
            pairing_key=pairing, fixed_resources=resources,
            randomness_path=f"measurement/train/{bank}", program_key="mixer_full_cycle"
        )
        test_acq = builder.add_acquisition(
            f"{stem}/acquire/test", data_id=test, split="test", kind=AcquisitionKind.GROUPED_MEASUREMENT,
            pairing_key=pairing, fixed_resources=resources,
            randomness_path=f"measurement/heldout/{bank}", program_key="mixer_full_cycle"
        )
        train_view = builder.add_view(
            f"{stem}/view/train", acquisition_id=train_acq, data_id=train, split="train",
            tasks=(task,), pairing_key=pairing, observable_bank=bank
        )
        test_view = builder.add_view(
            f"{stem}/view/test", acquisition_id=test_acq, data_id=test, split="test",
            tasks=(task,), pairing_key=pairing, observable_bank=bank
        )
        evaluations = builder.add_predictive_pair(stem, train_view=train_view, test_view=test_view, tasks=(task,), pairing_key=pairing)
        cost_evaluations.append(evaluations[0])
    builder.comparisons.append(
        ComparisonSpec(
            id="mixer_mechanism/cost_matched/bank_comparison",
            kind=ComparisonKind.DIAGNOSTIC,
            evaluation_ids=tuple(cost_evaluations),
            denominator_key=f"{pairing}/labels/{task}",
            pairing_key=pairing,
            parameters={"matched_branch_circuit_runs": 9 * edge_m},
        )
    )


def _task_atlas(builder: PlanBuilder) -> None:
    tasks = ATLAS_TASKS
    train, test, pairing = builder.add_paired_data("task_atlas", tasks=tasks, input_dim=64)
    methods = {
        **{f"quark_{name}": (AcquisitionKind.EXACT, "quark", {"n": n, "R": r, "tau_plus": tau})
           for name, (n, r, tau) in ARCHITECTURE_TIERS.items()},
        "raw_history_matern": (AcquisitionKind.RAW_HISTORY, "raw_history", {"w": 512}),
        "classical_multiscale": (
            AcquisitionKind.CLASSICAL_RESERVOIR,
            "classical_multiscale",
            {"matched_feature_dimension": "12Rn", "matched_memory_rates": True},
        ),
    }
    for method, (kind, program, resources) in methods.items():
        stem = f"task_atlas/{method}"
        train_acq = builder.add_acquisition(
            f"{stem}/acquire/train", data_id=train, split="train", kind=kind,
            pairing_key=pairing, fixed_resources=resources,
            randomness_path=f"representation/{method}", program_key=program
        )
        test_acq = builder.add_acquisition(
            f"{stem}/acquire/test", data_id=test, split="test", kind=kind,
            pairing_key=pairing, fixed_resources=resources,
            randomness_path=f"representation/{method}", program_key=program
        )
        train_view = builder.add_view(
            f"{stem}/view/train", acquisition_id=train_acq, data_id=train,
            split="train", tasks=tasks, pairing_key=pairing
        )
        test_view = builder.add_view(
            f"{stem}/view/test", acquisition_id=test_acq, data_id=test,
            split="test", tasks=tasks, pairing_key=pairing
        )
        builder.add_predictive_pair(stem, train_view=train_view, test_view=test_view, tasks=tasks, pairing_key=pairing)

