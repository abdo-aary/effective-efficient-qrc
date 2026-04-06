from __future__ import annotations

import argparse
from pathlib import Path

from src.experiment.classical_baselines import (
    DEFAULT_DATASET_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REFERENCE_MODEL_DIR,
    aggregate_classical_baselines,
    run_classical_baseline,
)


def baseline_parser(method: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Run rebuttal baseline: {method}")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--reference-model-dir", type=Path, default=DEFAULT_REFERENCE_MODEL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--method-seed", type=int, default=0)
    parser.add_argument("--feature-dim", type=int, default=315)
    parser.add_argument("--backend", choices=("auto", "cupy", "numpy"), default="cupy")
    parser.add_argument("--device", type=int, default=0)
    return parser


def run_baseline_cli(method: str) -> Path:
    args = baseline_parser(method).parse_args()
    run_dir = run_classical_baseline(
        method,
        dataset_path=args.dataset_path,
        reference_model_dir=args.reference_model_dir,
        out_dir=args.out_dir,
        method_seed=args.method_seed,
        feature_dim=args.feature_dim,
        backend=args.backend,
        device=args.device,
    )
    print(run_dir)
    return run_dir


def run_aggregate_cli() -> Path:
    parser = argparse.ArgumentParser(description="Aggregate classical-only rebuttal baselines.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--reference-model-dir", type=Path, default=DEFAULT_REFERENCE_MODEL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--skip-quark-reference",
        action="store_true",
        help="Do not regenerate the cached QuaRK reference row before aggregation.",
    )
    args = parser.parse_args()
    aggregate_dir = aggregate_classical_baselines(
        args.out_dir,
        dataset_path=args.dataset_path,
        reference_model_dir=args.reference_model_dir,
        include_quark_reference=not args.skip_quark_reference,
    )
    print(aggregate_dir)
    return aggregate_dir
