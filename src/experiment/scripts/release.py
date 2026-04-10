from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from src.settings import PROJECT_ROOT_PATH

from src.experiment.scripts.rebuttal.prepare_real_tser import main as prepare_real_tser_main
from src.experiment.scripts.rebuttal.run_real_quark_temporal_budget_comparison import (
    main as temporal_benchmark_main,
)
from src.experiment.scripts.rebuttal.run_rebuttal_result_audit import main as audit_main
from src.experiment.scripts.rebuttal.run_varma_ablation_suite import main as varma_ablation_main


PROJECT_ROOT = Path(PROJECT_ROOT_PATH)
PUBLIC_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "public_release"

PUBLIC_REAL_WORLD_DATASETS = (
    "beijing_pm25",
    "benzene_concentration",
    "copper_concentration",
    "electric_motor_temperature",
    "gas_sensor_array_acetone",
    "gas_sensor_array_ethanol",
    "hydraulic_systems",
    "iron_concentration",
    "live_fuel_moisture",
    "manganese_concentration",
)

DATASET_MANIFEST_ROWS = [
    {"dataset": "beijing_pm25", "N": 7000, "n_train": 5000, "n_test": 2000, "w": 24, "d": 9, "raw_dim": 216},
    {"dataset": "benzene_concentration", "N": 5349, "n_train": 3349, "n_test": 2000, "w": 240, "d": 8, "raw_dim": 1920},
    {"dataset": "copper_concentration", "N": 629, "n_train": 440, "n_test": 189, "w": 2542, "d": 1, "raw_dim": 2542},
    {"dataset": "electric_motor_temperature", "N": 4000, "n_train": 3000, "n_test": 1000, "w": 60, "d": 6, "raw_dim": 360},
    {"dataset": "gas_sensor_array_acetone", "N": 464, "n_train": 324, "n_test": 140, "w": 7500, "d": 1, "raw_dim": 7500},
    {"dataset": "gas_sensor_array_ethanol", "N": 464, "n_train": 324, "n_test": 140, "w": 7500, "d": 1, "raw_dim": 7500},
    {"dataset": "hydraulic_systems", "N": 1449, "n_train": 1159, "n_test": 290, "w": 60, "d": 728, "raw_dim": 43680},
    {"dataset": "iron_concentration", "N": 611, "n_train": 427, "n_test": 184, "w": 1716, "d": 1, "raw_dim": 1716},
    {"dataset": "live_fuel_moisture", "N": 4000, "n_train": 3000, "n_test": 1000, "w": 365, "d": 7, "raw_dim": 2555},
    {"dataset": "manganese_concentration", "N": 611, "n_train": 427, "n_test": 184, "w": 1716, "d": 1, "raw_dim": 1716},
]


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_dataset_manifest(real_root: Path) -> None:
    csv_path = real_root / "dataset_manifest.csv"
    md_path = real_root / "dataset_manifest.md"
    fieldnames = ["dataset", "N", "n_train", "n_test", "w", "d", "raw_dim"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(DATASET_MANIFEST_ROWS)
    md_lines = [
        "# Dataset Manifest",
        "",
        "| Dataset | N | Train | Test | w | d | raw_dim |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in DATASET_MANIFEST_ROWS:
        md_lines.append(
            f"| {row['dataset']} | {row['N']} | {row['n_train']} | {row['n_test']} | {row['w']} | {row['d']} | {row['raw_dim']} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def sync_public_release_artifacts() -> Path:
    varma_root = PUBLIC_ARTIFACT_ROOT / "varma_ablation"
    real_root = PUBLIC_ARTIFACT_ROOT / "real_world_temporal_benchmark"
    audit_root = PUBLIC_ARTIFACT_ROOT / "audit"
    varma_root.mkdir(parents=True, exist_ok=True)
    real_root.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)

    _copy(
        PROJECT_ROOT / "storage/results/rebuttal/result_audit/canonical_varma_claim_table.csv",
        varma_root / "canonical_varma_claim_table.csv",
    )
    _copy(
        PROJECT_ROOT / "storage/results/rebuttal/result_audit/canonical_varma_claim_table.md",
        varma_root / "canonical_varma_claim_table.md",
    )
    _copy(
        PROJECT_ROOT / "storage/results/rebuttal/varma_ablation/aggregate/finite_shot_ablation_table.md",
        varma_root / "finite_shot_ablation_table.md",
    )

    for name in (
        "temporal_budget_final_table.csv",
        "temporal_budget_final_table.md",
        "temporal_budget_lambda_selection.csv",
        "temporal_budget_lambda_selection.md",
        "temporal_budget_shot_sweep.csv",
        "temporal_budget_shot_sweep.md",
        "temporal_budget_selection.json",
    ):
        _copy(
            PROJECT_ROOT / "storage/results/rebuttal/result_audit/canonical_real_world" / name,
            real_root / name,
        )
    _write_dataset_manifest(real_root)

    for name in (
        "canonical_summary.json",
        "code_path_audit.csv",
        "code_path_audit.md",
        "claim_ledger.csv",
        "claim_ledger.md",
        "rebuttal_claim_checklist.csv",
        "rebuttal_claim_checklist.md",
        "aggregate_comparison.csv",
        "aggregate_comparison.md",
        "selective_recompute_report.csv",
        "selective_recompute_report.md",
    ):
        _copy(PROJECT_ROOT / "storage/results/rebuttal/result_audit" / name, audit_root / name)
    return PUBLIC_ARTIFACT_ROOT


def build_prepare_args(args: argparse.Namespace) -> list[str]:
    argv = [
        "--datasets",
        *PUBLIC_REAL_WORLD_DATASETS,
        "--out-root",
        str(args.data_root),
    ]
    if args.raw_root is not None:
        argv += ["--raw-root", str(args.raw_root)]
    if args.no_download:
        argv.append("--no-download")
    if args.dry_run:
        argv.append("--dry-run")
    return argv


def build_varma_args(args: argparse.Namespace) -> list[str]:
    argv = [
        "--stages",
        "prepare",
        "quark_architecture",
        "shots",
        "aggregate",
        "--data-root",
        str(args.data_root),
        "--out-root",
        str(args.out_root),
        "--backend",
        args.backend,
        "--device",
        str(args.device),
        "--num-workers",
        str(args.num_workers),
        "--retune-readout",
    ]
    if args.dry_run:
        argv.append("--dry-run")
    return argv


def build_benchmark_args(args: argparse.Namespace) -> list[str]:
    argv = [
        "--datasets",
        *PUBLIC_REAL_WORLD_DATASETS,
        "--stages",
        "exact",
        "esn",
        "shots",
        "final",
        "aggregate",
        "--data-root",
        str(args.data_root),
        "--out-root",
        str(args.out_root),
        "--runner",
        "reservoir_channel_cupy_direct_truncated64",
        "--temporal-baseline",
        "esn_matern_krr",
        "--reuse-existing-esn-config",
        "--seeds",
        "0",
        "1",
        "2",
        "--final-seeds",
        "0",
        "1",
        "2",
        "3",
        "4",
        "--lam-grid",
        "0.5",
        "--shot-grid",
        "3000",
        "--backend",
        args.backend,
        "--device",
        str(args.device),
        "--num-workers",
        str(args.num_workers),
        "--retune-readout",
        "--skip-existing",
    ]
    if args.dry_run:
        argv.append("--dry-run")
    return argv


def build_audit_args(args: argparse.Namespace) -> list[str]:
    argv = [
        "--stages",
        "canonical",
        "code_audit",
        "checklist",
        "rerun",
        "--backend",
        args.backend,
        "--device",
        str(args.device),
        "--num-workers",
        str(args.num_workers),
    ]
    if args.dry_run:
        argv.append("--dry-run")
    return argv


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Public release wrapper for QuaRK + ESN+Matérn reproduction.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare-real-data", help="Prepare the retained TSER datasets.")
    prepare_parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "storage/data/real/tser")
    prepare_parser.add_argument("--raw-root", type=Path, default=None)
    prepare_parser.add_argument("--no-download", action="store_true")
    prepare_parser.add_argument("--dry-run", action="store_true")

    varma_parser = subparsers.add_parser("run-varma-ablation", help="Run the retained QuaRK-only VARMA ablation.")
    varma_parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "storage/data/synthetic/varma_ablation")
    varma_parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "storage/results/rebuttal/varma_ablation")
    varma_parser.add_argument("--backend", choices=("auto", "cupy", "numpy"), default="cupy")
    varma_parser.add_argument("--device", type=int, default=0)
    varma_parser.add_argument("--num-workers", type=int, default=1)
    varma_parser.add_argument("--dry-run", action="store_true")

    benchmark_parser = subparsers.add_parser(
        "run-real-world-benchmark",
        help="Run the fixed QuaRK vs ESN+Matérn temporal benchmark and sync canonical artifacts.",
    )
    benchmark_parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "storage/data/real/tser")
    benchmark_parser.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "storage/results/rebuttal/real_world_temporal_budget",
    )
    benchmark_parser.add_argument("--backend", choices=("auto", "cupy", "numpy"), default="cupy")
    benchmark_parser.add_argument("--device", type=int, default=0)
    benchmark_parser.add_argument("--num-workers", type=int, default=1)
    benchmark_parser.add_argument("--dry-run", action="store_true")

    audit_parser = subparsers.add_parser("run-audit", help="Run the retained release-verification audit.")
    audit_parser.add_argument("--backend", choices=("auto", "cupy", "numpy"), default="cupy")
    audit_parser.add_argument("--device", type=int, default=0)
    audit_parser.add_argument("--num-workers", type=int, default=1)
    audit_parser.add_argument("--dry-run", action="store_true")

    all_parser = subparsers.add_parser("run-all", help="Run data prep, VARMA ablation, benchmark, audit, and artifact sync.")
    all_parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "storage/data/real/tser")
    all_parser.add_argument("--raw-root", type=Path, default=None)
    all_parser.add_argument("--no-download", action="store_true")
    all_parser.add_argument("--backend", choices=("auto", "cupy", "numpy"), default="cupy")
    all_parser.add_argument("--device", type=int, default=0)
    all_parser.add_argument("--num-workers", type=int, default=1)
    all_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "prepare-real-data":
        prepare_real_tser_main(build_prepare_args(args))
        return
    if args.command == "run-varma-ablation":
        varma_ablation_main(build_varma_args(args))
        if not args.dry_run:
            print(sync_public_release_artifacts())
        return
    if args.command == "run-real-world-benchmark":
        temporal_benchmark_main(build_benchmark_args(args))
        if not args.dry_run:
            print(sync_public_release_artifacts())
        return
    if args.command == "run-audit":
        audit_main(build_audit_args(args))
        if not args.dry_run:
            print(sync_public_release_artifacts())
        return

    if args.command == "run-all":
        prepare_real_tser_main(build_prepare_args(args))
        varma_ablation_args = argparse.Namespace(
            data_root=PROJECT_ROOT / "storage/data/synthetic/varma_ablation",
            out_root=PROJECT_ROOT / "storage/results/rebuttal/varma_ablation",
            backend=args.backend,
            device=args.device,
            num_workers=args.num_workers,
            dry_run=args.dry_run,
        )
        benchmark_args = argparse.Namespace(
            data_root=args.data_root,
            out_root=PROJECT_ROOT / "storage/results/rebuttal/real_world_temporal_budget",
            backend=args.backend,
            device=args.device,
            num_workers=args.num_workers,
            dry_run=args.dry_run,
        )
        audit_args = argparse.Namespace(
            backend=args.backend,
            device=args.device,
            num_workers=args.num_workers,
            dry_run=args.dry_run,
        )
        varma_ablation_main(build_varma_args(varma_args := varma_ablation_args))
        temporal_benchmark_main(build_benchmark_args(benchmark_args))
        audit_main(build_audit_args(audit_args))
        if not args.dry_run:
            print(sync_public_release_artifacts())
        return

    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
