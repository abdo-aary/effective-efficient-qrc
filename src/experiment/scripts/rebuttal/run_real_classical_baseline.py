from __future__ import annotations

import argparse
from pathlib import Path

from src.data.real_tser import DEFAULT_TSER_DATA_ROOT, REAL_DATASET_KEYS
from src.experiment.classical_baselines import CLASSICAL_METHODS
from src.experiment.real_world_rebuttal import DEFAULT_REAL_RESULTS_ROOT, run_real_classical_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one real-world TSER classical rebuttal baseline.")
    parser.add_argument("--dataset", required=True, choices=sorted(REAL_DATASET_KEYS))
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_TSER_DATA_ROOT)
    parser.add_argument("--method", required=True, choices=sorted(CLASSICAL_METHODS))
    parser.add_argument("--out-root", type=Path, default=DEFAULT_REAL_RESULTS_ROOT)
    parser.add_argument("--method-seed", type=int, default=0)
    parser.add_argument("--feature-dim", type=int, default=315)
    parser.add_argument("--backend", choices=("auto", "cupy", "numpy"), default="cupy")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    dataset_path = args.dataset_path or (args.data_root / args.dataset)
    run_dir = run_real_classical_baseline(
        args.method,
        dataset_path=dataset_path,
        out_root=args.out_root,
        method_seed=args.method_seed,
        feature_dim=args.feature_dim,
        backend=args.backend,
        device=args.device,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
