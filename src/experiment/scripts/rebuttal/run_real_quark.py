from __future__ import annotations

import argparse
from pathlib import Path

from src.data.real_tser import DEFAULT_TSER_DATA_ROOT, REAL_DATASET_KEYS
from src.experiment.real_world_rebuttal import DEFAULT_REAL_RESULTS_ROOT, run_real_quark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QuaRK on one prepared real-world TSER dataset.")
    parser.add_argument("--dataset", required=True, choices=sorted(REAL_DATASET_KEYS))
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_TSER_DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_REAL_RESULTS_ROOT)
    parser.add_argument("--runner", default="reservoir_channel_cupy")
    parser.add_argument("--method-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--backend", choices=("auto", "cupy", "numpy"), default="cupy")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--method-suffix", default=None)
    parser.add_argument(
        "--model-override",
        action="append",
        default=[],
        help="Additional Hydra model override. Can be passed multiple times.",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path or (args.data_root / args.dataset)
    run_dir = run_real_quark(
        dataset_path=dataset_path,
        out_root=args.out_root,
        method_seed=args.method_seed,
        runner=args.runner,
        num_workers=args.num_workers,
        experiment_name=args.experiment_name,
        backend=args.backend,
        device=args.device,
        method_suffix=args.method_suffix,
        model_overrides=args.model_override,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
