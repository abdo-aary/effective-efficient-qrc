from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data.real_tser import DEFAULT_TSER_DATA_ROOT, REAL_DATASET_KEYS, prepare_real_dataset
from src.experiment.real_world_rebuttal import (
    DEFAULT_REAL_TRIAGE_RESULTS_ROOT,
    READOUT_RETUNE_LAMBDA_GRID,
    READOUT_RETUNE_XI_GRID,
    TRIAGE_CANDIDATE_DATASETS,
    TRIAGE_CLASSICAL_METHODS,
    TRIAGE_QUARK_REGIMES,
    build_classical_triage_rows,
    build_dataset_card_rows,
    read_real_metric_rows,
    run_cached_quark_readout_retune,
    run_real_classical_baseline,
    run_real_quark,
    write_real_dataset_triage_outputs,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run staged real-world dataset triage for QuaRK.")
    parser.add_argument("--datasets", nargs="+", default=list(TRIAGE_CANDIDATE_DATASETS), choices=sorted(REAL_DATASET_KEYS))
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["prepare", "classical", "decide"],
        choices=("prepare", "classical", "decide", "quark", "all"),
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_TSER_DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_REAL_TRIAGE_RESULTS_ROOT)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-train", type=int, default=3000)
    parser.add_argument("--max-test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clip-value", type=float, default=5.0)
    parser.add_argument("--methods", nargs="+", default=list(TRIAGE_CLASSICAL_METHODS), choices=sorted(TRIAGE_CLASSICAL_METHODS))
    parser.add_argument("--feature-dim", type=int, default=315)
    parser.add_argument("--backend", choices=("auto", "cupy", "numpy"), default="cupy")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runner", default="reservoir_channel_cupy_direct")
    parser.add_argument("--method-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-promoted", type=int, default=3)
    parser.add_argument("--quark-regimes", nargs="+", default=list(TRIAGE_QUARK_REGIMES), choices=sorted(TRIAGE_QUARK_REGIMES))
    parser.add_argument("--retune-readout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-strategy", choices=("chronological", "random"), default="random")
    parser.add_argument("--tune-max-inner", type=int, default=1000)
    parser.add_argument("--tune-max-val", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    stages = expand_stages(args.stages)
    if args.dry_run:
        for action in build_dry_run_actions(args, stages=stages):
            print(action)
        return

    if "prepare" in stages:
        for dataset in args.datasets:
            out_dir = prepare_real_dataset(
                dataset,
                raw_root=args.raw_root,
                out_root=args.data_root,
                download=not args.no_download,
                force_download=args.force_download,
                overwrite=args.overwrite,
                max_train=args.max_train,
                max_test=args.max_test,
                seed=args.seed,
                clip_value=args.clip_value,
            )
            print(out_dir)

    if "classical" in stages:
        for dataset in args.datasets:
            dataset_path = args.data_root / dataset
            for method in args.methods:
                run_dir = run_real_classical_baseline(
                    method,
                    dataset_path=dataset_path,
                    out_root=args.out_root,
                    method_seed=args.method_seed,
                    feature_dim=args.feature_dim,
                    backend=args.backend,
                    device=args.device,
                )
                print(run_dir)

    aggregate_dir = None
    if "decide" in stages or "quark" in stages:
        aggregate_dir = write_real_dataset_triage_outputs(
            args.out_root,
            data_root=args.data_root,
            datasets=args.datasets,
            max_promoted=args.max_promoted,
        )
        print(aggregate_dir)

    if "quark" in stages:
        selected = selected_quark_datasets(
            out_root=args.out_root,
            data_root=args.data_root,
            datasets=args.datasets,
            max_promoted=args.max_promoted,
        )
        for dataset in selected:
            dataset_path = args.data_root / dataset
            for regime in args.quark_regimes:
                run_dir = run_real_quark(
                    dataset_path=dataset_path,
                    out_root=args.out_root,
                    method_seed=args.method_seed,
                    runner=args.runner,
                    num_workers=args.num_workers,
                    experiment_name=f"{dataset}_{regime}",
                    backend=args.backend,
                    device=args.device,
                    method_suffix=regime,
                    model_overrides=TRIAGE_QUARK_REGIMES[regime],
                )
                print(run_dir)
                if args.retune_readout:
                    retune_method = f"quark_{args.runner}_{regime}_kernel_readout_retune"
                    retune_dir = run_cached_quark_readout_retune(
                        dataset_path=dataset_path,
                        source_run=run_dir,
                        out_root=args.out_root,
                        method_seed=args.method_seed,
                        val_ratio=args.val_ratio,
                        split_strategy=args.split_strategy,
                        lambda_grid=READOUT_RETUNE_LAMBDA_GRID,
                        xi_grid=READOUT_RETUNE_XI_GRID,
                        nu_grid=np.asarray([0.5, 1.5, 2.5], dtype=float),
                        tune_max_inner=args.tune_max_inner,
                        tune_max_val=args.tune_max_val,
                        method=retune_method,
                        backend=args.backend,
                        device=args.device,
                    )
                    print(retune_dir)
        aggregate_dir = write_real_dataset_triage_outputs(
            args.out_root,
            data_root=args.data_root,
            datasets=args.datasets,
            max_promoted=args.max_promoted,
        )
        print(aggregate_dir)


def expand_stages(stages: list[str]) -> set[str]:
    if "all" in stages:
        return {"prepare", "classical", "decide", "quark"}
    return set(stages)


def selected_quark_datasets(
    *,
    out_root: Path,
    data_root: Path,
    datasets: list[str],
    max_promoted: int,
) -> list[str]:
    metric_rows = read_real_metric_rows(out_root, include_quark=True)
    dataset_rows = build_dataset_card_rows(data_root, datasets=datasets)
    decisions = build_classical_triage_rows(
        metric_rows,
        dataset_card_rows=dataset_rows,
        max_promoted=max_promoted,
    )
    return [str(row["dataset"]) for row in decisions if bool(row.get("selected_for_quark"))]


def build_dry_run_actions(args: argparse.Namespace, *, stages: set[str]) -> list[str]:
    actions: list[str] = []
    if "prepare" in stages:
        actions.append(
            "prepare datasets="
            + ",".join(args.datasets)
            + f" max_train={args.max_train} max_test={args.max_test} overwrite={args.overwrite}"
        )
    if "classical" in stages:
        actions.append(
            "run classical baselines datasets="
            + ",".join(args.datasets)
            + " methods="
            + ",".join(args.methods)
            + f" backend={args.backend} out_root={args.out_root}"
        )
    if "decide" in stages:
        actions.append(f"write triage decisions out_root={args.out_root} max_promoted={args.max_promoted}")
    if "quark" in stages:
        actions.append(
            "run QuaRK regimes datasets=<selected_by_triage> regimes="
            + ",".join(args.quark_regimes)
            + f" runner={args.runner} backend={args.backend} retune={args.retune_readout}"
        )
    return actions


if __name__ == "__main__":
    main()
