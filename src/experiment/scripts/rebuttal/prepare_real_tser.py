from __future__ import annotations

import argparse
from pathlib import Path

from src.data.real_tser import (
    DEFAULT_TSER_DATA_ROOT,
    REAL_DATASET_KEYS,
    prepare_real_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert real-world TSER datasets.")
    parser.add_argument("--datasets", nargs="+", default=list(REAL_DATASET_KEYS), choices=sorted(REAL_DATASET_KEYS))
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_TSER_DATA_ROOT)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clip-value", type=float, default=5.0)
    parser.add_argument("--hydraulic-target", default="hydraulic_accumulator_pressure")
    parser.add_argument("--hydraulic-test-ratio", type=float, default=0.2)
    parser.add_argument("--hydraulic-split-strategy", choices=("random", "chronological"), default="random")
    parser.add_argument("--include-unstable-hydraulic", action="store_true")
    parser.add_argument("--hydraulic-length-seconds", type=int, default=60)
    args = parser.parse_args()

    for dataset in args.datasets:
        out_dir = prepare_real_dataset(
            dataset,
            raw_root=args.raw_root,
            out_root=args.out_root,
            download=not args.no_download,
            force_download=args.force_download,
            overwrite=args.overwrite,
            max_train=args.max_train,
            max_test=args.max_test,
            seed=args.seed,
            clip_value=args.clip_value,
            hydraulic_target=args.hydraulic_target,
            hydraulic_test_ratio=args.hydraulic_test_ratio,
            hydraulic_split_strategy=args.hydraulic_split_strategy,
            hydraulic_stable_only=not args.include_unstable_hydraulic,
            hydraulic_length_seconds=args.hydraulic_length_seconds,
        )
        print(out_dir)


if __name__ == "__main__":
    main()
