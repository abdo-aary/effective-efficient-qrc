from __future__ import annotations

import argparse
from pathlib import Path

from src.data.real_tser import DEFAULT_TSER_DATA_ROOT, REAL_DATASET_KEYS
from src.experiment.real_world_rebuttal import (
    DEFAULT_REAL_RESULTS_ROOT,
    READOUT_RETUNE_LAMBDA_GRID,
    READOUT_RETUNE_NU_GRID,
    READOUT_RETUNE_XI_GRID,
    run_cached_quark_readout_retune,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retune QuaRK KRR readout from cached real-world features.")
    parser.add_argument("--dataset", required=True, choices=sorted(REAL_DATASET_KEYS))
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_TSER_DATA_ROOT)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_REAL_RESULTS_ROOT)
    parser.add_argument("--method-seed", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-strategy", choices=["chronological", "random"], default="chronological")
    parser.add_argument("--lambda-exp-min", type=float, default=-6.0)
    parser.add_argument("--lambda-exp-max", type=float, default=10.0)
    parser.add_argument("--lambda-exp-step", type=float, default=0.5)
    parser.add_argument("--retune-kernel", action="store_true")
    parser.add_argument("--xi-exp-min", type=float, default=-1.0)
    parser.add_argument("--xi-exp-max", type=float, default=3.0)
    parser.add_argument("--xi-exp-step", type=float, default=0.5)
    parser.add_argument("--nu-grid", default="0.5,1.5,2.5,5.0")
    parser.add_argument("--tune-max-inner", type=int, default=1500)
    parser.add_argument("--tune-max-val", type=int, default=500)
    parser.add_argument("--backend", choices=("auto", "cupy", "numpy"), default="cupy")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    dataset_path = args.dataset_path or (args.data_root / args.dataset)
    if (
        args.lambda_exp_min == -6.0
        and args.lambda_exp_max == 10.0
        and args.lambda_exp_step == 0.5
    ):
        lambda_grid = READOUT_RETUNE_LAMBDA_GRID
    else:
        import numpy as np

        lambda_grid = 10.0 ** np.arange(
            float(args.lambda_exp_min),
            float(args.lambda_exp_max) + 0.5 * float(args.lambda_exp_step),
            float(args.lambda_exp_step),
        )
    if args.retune_kernel:
        import numpy as np

        if (
            args.xi_exp_min == -1.0
            and args.xi_exp_max == 3.0
            and args.xi_exp_step == 0.5
        ):
            xi_grid = READOUT_RETUNE_XI_GRID
        else:
            xi_grid = 10.0 ** np.arange(
                float(args.xi_exp_min),
                float(args.xi_exp_max) + 0.5 * float(args.xi_exp_step),
                float(args.xi_exp_step),
            )
        if args.nu_grid.strip():
            nu_grid = np.asarray([float(part) for part in args.nu_grid.split(",") if part.strip()], dtype=float)
        else:
            nu_grid = READOUT_RETUNE_NU_GRID
    else:
        xi_grid = None
        nu_grid = READOUT_RETUNE_NU_GRID

    run_dir = run_cached_quark_readout_retune(
        dataset_path=dataset_path,
        source_run=args.source_run,
        out_root=args.out_root,
        method_seed=args.method_seed,
        val_ratio=args.val_ratio,
        split_strategy=args.split_strategy,
        lambda_grid=lambda_grid,
        xi_grid=xi_grid,
        nu_grid=nu_grid,
        tune_max_inner=args.tune_max_inner if args.retune_kernel else None,
        tune_max_val=args.tune_max_val if args.retune_kernel else None,
        backend=args.backend,
        device=args.device,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
