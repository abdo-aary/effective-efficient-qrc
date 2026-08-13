"""Campaign II: empirical shot, window, and dependence fidelity gaps."""

from __future__ import annotations

from ..domain import (
    AcquisitionKind,
    CampaignId,
    ComparisonKind,
    ComparisonSpec,
    EvaluationSpec,
    FitSpec,
    RiskRole,
)
from ..manifest import RunManifest
from .base import PlanBuilder, limits
from .constants import (
    DEPENDENCE_GRID,
    GAP_GRID,
    M_GRID,
    SAMPLE_GRID,
    SHOT_ARCHITECTURES,
    TAU_PLUS_GRID,
    WINDOW_GRID,
)


def plan_campaign_ii(manifest: RunManifest, repetition):
    builder = PlanBuilder(manifest, CampaignId.FIDELITY, repetition)
    _shot_fidelity(builder)
    _window_fidelity(builder)
    _dependence_fidelity(builder)
    return builder.build(
        title="Empirical fidelity decomposition",
        measurement_prefix_max=8192,
        causal_proxy_burn_in=1024,
    )


def _shot_fidelity(builder: PlanBuilder) -> None:
    tasks = builder.manifest.pre_run.fidelity_tasks
    train, test, pairing = builder.add_paired_data("shot_fidelity", tasks=tasks, input_dim=16)
    for width, branches in SHOT_ARCHITECTURES:
        stem = f"shot_fidelity/n={width}/R={branches}"
        resources = {"n": width, "R": branches, "tau_plus": 32, "settings": 9}
        measured_train = builder.add_acquisition(
            f"{stem}/measured/acquire/train", data_id=train, split="train",
            kind=AcquisitionKind.GROUPED_MEASUREMENT, pairing_key=pairing,
            prefix_limits=limits(M=max(M_GRID)), fixed_resources=resources,
            randomness_path=f"measurement/train/n={width}/R={branches}"
        )
        measured_test = builder.add_acquisition(
            f"{stem}/measured/acquire/test", data_id=test, split="test",
            kind=AcquisitionKind.GROUPED_MEASUREMENT, pairing_key=pairing,
            prefix_limits=limits(M=max(M_GRID)), fixed_resources=resources,
            randomness_path=f"measurement/heldout/n={width}/R={branches}"
        )
        exact_test = builder.add_acquisition(
            f"{stem}/exact/acquire/test", data_id=test, split="test",
            kind=AcquisitionKind.EXACT, pairing_key=pairing,
            fixed_resources=resources, randomness_path=f"design/n={width}/R={branches}"
        )
        exact_view = builder.add_view(
            f"{stem}/exact/view/test", acquisition_id=exact_test, data_id=test,
            split="test", tasks=tasks, pairing_key=pairing
        )
        for shots in M_GRID:
            measured_train_view = builder.add_view(
                f"{stem}/measured/view/train/M={shots}", acquisition_id=measured_train,
                data_id=train, split="train", tasks=tasks, pairing_key=pairing,
                prefixes=limits(M=shots)
            )
            measured_test_view = builder.add_view(
                f"{stem}/measured/view/test/M={shots}", acquisition_id=measured_test,
                data_id=test, split="test", tasks=tasks, pairing_key=pairing,
                prefixes=limits(M=shots)
            )
            for task in tasks:
                node = f"{stem}/M={shots}/{task}"
                fit_id = f"{node}/fit"
                denominator = f"{pairing}/labels/{task}"
                builder.fits.append(
                    FitSpec(
                        id=fit_id, feature_view_id=measured_train_view, task_ids=(task,),
                        readout_key="fixed_rms_matern_ivanov", pairing_key=pairing
                    )
                )
                measured_eval = f"{node}/eval/measured"
                exact_eval = f"{node}/eval/exact"
                builder.evaluations.extend(
                    (
                        EvaluationSpec(
                            id=measured_eval, fit_id=fit_id, feature_view_id=measured_test_view,
                            data_id=test, task_ids=(task,), risk_role=RiskRole.HELDOUT,
                            denominator_key=denominator, pairing_key=pairing
                        ),
                        EvaluationSpec(
                            id=exact_eval, fit_id=fit_id, feature_view_id=exact_view,
                            data_id=test, task_ids=(task,), risk_role=RiskRole.HELDOUT,
                            denominator_key=denominator, pairing_key=pairing
                        ),
                    )
                )
                builder.comparisons.extend(
                    (
                        ComparisonSpec(
                            id=f"{node}/shot_gap", kind=ComparisonKind.SHOT_GAP,
                            evaluation_ids=(measured_eval, exact_eval), denominator_key=denominator,
                            pairing_key=pairing, parameters={"M": shots, "x_axis": "M^-1/2"}
                        ),
                        ComparisonSpec(
                            id=f"{node}/nemse", kind=ComparisonKind.NEMSE,
                            evaluation_ids=(measured_eval,), denominator_key=denominator,
                            pairing_key=pairing
                        ),
                    )
                )


def _window_fidelity(builder: PlanBuilder) -> None:
    tasks = builder.manifest.pre_run.fidelity_tasks
    train, test, pairing = builder.add_paired_data("window_fidelity", tasks=tasks, input_dim=16)
    for tau in TAU_PLUS_GRID:
        stem = f"window_fidelity/tau_plus={tau}"
        resources = {"n": 8, "R": 16, "tau_plus": tau}
        finite_train = builder.add_acquisition(
            f"{stem}/finite/acquire/train", data_id=train, split="train",
            kind=AcquisitionKind.EXACT, pairing_key=pairing,
            prefix_limits=limits(w=max(WINDOW_GRID)), fixed_resources=resources,
            randomness_path="design/branches"
        )
        finite_test = builder.add_acquisition(
            f"{stem}/finite/acquire/test", data_id=test, split="test",
            kind=AcquisitionKind.EXACT, pairing_key=pairing,
            prefix_limits=limits(w=max(WINDOW_GRID)), fixed_resources=resources,
            randomness_path="design/branches"
        )
        proxy_test = builder.add_acquisition(
            f"{stem}/proxy/acquire/test", data_id=test, split="test",
            kind=AcquisitionKind.CAUSAL_PROXY, pairing_key=pairing,
            fixed_resources={**resources, "burn_in": 1024}, randomness_path="design/branches"
        )
        proxy_view = builder.add_view(
            f"{stem}/proxy/view/test", acquisition_id=proxy_test, data_id=test,
            split="test", tasks=tasks, pairing_key=pairing,
            parameters={"state_residual": f"2*exp(-1024/{tau})"}
        )
        for window in WINDOW_GRID:
            finite_train_view = builder.add_view(
                f"{stem}/finite/view/train/w={window}", acquisition_id=finite_train,
                data_id=train, split="train", tasks=tasks, pairing_key=pairing,
                prefixes=limits(w=window)
            )
            finite_test_view = builder.add_view(
                f"{stem}/finite/view/test/w={window}", acquisition_id=finite_test,
                data_id=test, split="test", tasks=tasks, pairing_key=pairing,
                prefixes=limits(w=window)
            )
            for task in tasks:
                node = f"{stem}/w={window}/{task}"
                fit_id = f"{node}/fit"
                denominator = f"{pairing}/labels/{task}"
                builder.fits.append(
                    FitSpec(
                        id=fit_id, feature_view_id=finite_train_view, task_ids=(task,),
                        readout_key="fixed_rms_matern_ivanov", pairing_key=pairing
                    )
                )
                finite_eval, proxy_eval = f"{node}/eval/finite", f"{node}/eval/proxy"
                builder.evaluations.extend(
                    (
                        EvaluationSpec(
                            id=finite_eval, fit_id=fit_id, feature_view_id=finite_test_view,
                            data_id=test, task_ids=(task,), risk_role=RiskRole.HELDOUT,
                            denominator_key=denominator, pairing_key=pairing
                        ),
                        EvaluationSpec(
                            id=proxy_eval, fit_id=fit_id, feature_view_id=proxy_view,
                            data_id=test, task_ids=(task,), risk_role=RiskRole.HELDOUT,
                            denominator_key=denominator, pairing_key=pairing
                        ),
                    )
                )
                builder.comparisons.append(
                    ComparisonSpec(
                        id=f"{node}/window_gap", kind=ComparisonKind.WINDOW_GAP,
                        evaluation_ids=(finite_eval, proxy_eval), denominator_key=denominator,
                        pairing_key=pairing,
                        parameters={"window": window, "burn_in": 1024, "report_epsilon_burn": True}
                    )
                )


def _dependence_fidelity(builder: PlanBuilder) -> None:
    task = builder.manifest.pre_run.dependence_task
    for half_life in (None,) + DEPENDENCE_GRID:
        dep_label = "iid" if half_life is None else str(half_life)
        for gap in GAP_GRID:
            study = f"dependence/surface/tau_dep={dep_label}/g={gap}"
            train, test, pairing = builder.add_paired_data(
                study, tasks=(task,), input_dim=1, sample_count=1024, gap=gap,
                dependence_half_life=half_life
            )
            _add_generalization_cell(builder, study, train, test, pairing, task, sample_count=1024)

    for gap in GAP_GRID:
        study = f"dependence/sample_size/tau_dep=16/g={gap}"
        train, test, pairing = builder.add_paired_data(
            study, tasks=(task,), input_dim=1, sample_count=max(SAMPLE_GRID), gap=gap,
            dependence_half_life=16
        )
        resources = {"n": 8, "R": 16, "tau_plus": 32, "features": "exact"}
        train_acq = builder.add_acquisition(
            f"{study}/acquire/train", data_id=train, split="train", kind=AcquisitionKind.EXACT,
            pairing_key=pairing, prefix_limits=limits(N=max(SAMPLE_GRID)),
            fixed_resources=resources, randomness_path="design/reference"
        )
        test_acq = builder.add_acquisition(
            f"{study}/acquire/test", data_id=test, split="test", kind=AcquisitionKind.EXACT,
            pairing_key=pairing, fixed_resources=resources, randomness_path="design/reference"
        )
        test_view = builder.add_view(
            f"{study}/view/test", acquisition_id=test_acq, data_id=test,
            split="test", tasks=(task,), pairing_key=pairing
        )
        for sample_count in SAMPLE_GRID:
            train_view = builder.add_view(
                f"{study}/view/train/N={sample_count}", acquisition_id=train_acq,
                data_id=train, split="train", tasks=(task,), pairing_key=pairing,
                prefixes=limits(N=sample_count)
            )
            _add_generalization_fit(
                builder, f"{study}/N={sample_count}", train_view, test_view,
                train, test, pairing, task, sample_count
            )


def _add_generalization_cell(
    builder: PlanBuilder, study: str, train: str, test: str, pairing: str, task: str, sample_count: int
) -> None:
    resources = {"n": 8, "R": 16, "tau_plus": 32, "features": "exact"}
    train_acq = builder.add_acquisition(
        f"{study}/acquire/train", data_id=train, split="train", kind=AcquisitionKind.EXACT,
        pairing_key=pairing, fixed_resources=resources, randomness_path="design/reference"
    )
    test_acq = builder.add_acquisition(
        f"{study}/acquire/test", data_id=test, split="test", kind=AcquisitionKind.EXACT,
        pairing_key=pairing, fixed_resources=resources, randomness_path="design/reference"
    )
    train_view = builder.add_view(
        f"{study}/view/train", acquisition_id=train_acq, data_id=train,
        split="train", tasks=(task,), pairing_key=pairing
    )
    test_view = builder.add_view(
        f"{study}/view/test", acquisition_id=test_acq, data_id=test,
        split="test", tasks=(task,), pairing_key=pairing
    )
    _add_generalization_fit(builder, study, train_view, test_view, train, test, pairing, task, sample_count)


def _add_generalization_fit(
    builder: PlanBuilder,
    node: str,
    train_view: str,
    test_view: str,
    train_data: str,
    test_data: str,
    pairing: str,
    task: str,
    sample_count: int,
) -> None:
    fit_id = f"{node}/fit/{task}"
    denominator = f"{pairing}/labels/{task}"
    builder.fits.append(
        FitSpec(
            id=fit_id, feature_view_id=train_view, task_ids=(task,),
            readout_key="fixed_rms_matern_ivanov", pairing_key=pairing
        )
    )
    train_eval, test_eval = f"{node}/eval/train/{task}", f"{node}/eval/test/{task}"
    builder.evaluations.extend(
        (
            EvaluationSpec(
                id=train_eval, fit_id=fit_id, feature_view_id=train_view, data_id=train_data,
                task_ids=(task,), risk_role=RiskRole.TRAIN, denominator_key=denominator,
                pairing_key=pairing
            ),
            EvaluationSpec(
                id=test_eval, fit_id=fit_id, feature_view_id=test_view, data_id=test_data,
                task_ids=(task,), risk_role=RiskRole.TEST, denominator_key=denominator,
                pairing_key=pairing
            ),
        )
    )
    builder.comparisons.append(
        ComparisonSpec(
            id=f"{node}/generalization_gap/{task}", kind=ComparisonKind.GENERALIZATION_GAP,
            evaluation_ids=(train_eval, test_eval), denominator_key=denominator,
            pairing_key=pairing, parameters={"N": sample_count}
        )
    )

