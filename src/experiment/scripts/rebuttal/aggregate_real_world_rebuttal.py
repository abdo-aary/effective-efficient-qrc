from __future__ import annotations

import argparse
from pathlib import Path

from src.experiment.real_world_rebuttal import (
    DEFAULT_REAL_RESPONSE_DIR,
    DEFAULT_REAL_RESULTS_ROOT,
    aggregate_real_world_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate real-world TSER rebuttal runs.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_REAL_RESULTS_ROOT)
    parser.add_argument("--response-dir", type=Path, default=DEFAULT_REAL_RESPONSE_DIR)
    parser.add_argument("--metric", default="nrmse_train_y_std", choices=["mse", "rmse", "nrmse_train_y_std"])
    parser.add_argument("--classical-only", action="store_true")
    args = parser.parse_args()

    aggregate_dir = aggregate_real_world_results(
        args.out_root,
        response_dir=args.response_dir,
        metric=args.metric,
        include_quark=not args.classical_only,
    )
    print(aggregate_dir)


if __name__ == "__main__":
    main()
