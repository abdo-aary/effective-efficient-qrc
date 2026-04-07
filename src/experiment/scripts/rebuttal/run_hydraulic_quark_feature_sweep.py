from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data.real_tser import DEFAULT_TSER_DATA_ROOT
from src.experiment.real_world_rebuttal import (
    DEFAULT_REAL_RESULTS_ROOT,
    READOUT_RETUNE_LAMBDA_GRID,
    READOUT_RETUNE_XI_GRID,
    run_cached_quark_readout_retune,
    run_real_quark,
)


REGIMES: dict[str, list[str]] = {
    "exact_k2_R3": [
        "model/qrc/features/retriever=exact",
    ],
    "exact_k1_R3": [
        "model/qrc/features/retriever=exact",
        "model.qrc.features.observables.locality=1",
    ],
    "exact_k2_R1_lam0p1": [
        "model/qrc/features/retriever=exact",
        "model.qrc.pubs.num_reservoirs=1",
        "model.qrc.pubs.lam_0=0.1",
    ],
    "exact_k1_R1_lam0p1": [
        "model/qrc/features/retriever=exact",
        "model.qrc.features.observables.locality=1",
        "model.qrc.pubs.num_reservoirs=1",
        "model.qrc.pubs.lam_0=0.1",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Hydraulic QuaRK feature-regime sweep.")
    parser.add_argument("--dataset", default="hydraulic_systems")
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_TSER_DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_REAL_RESULTS_ROOT)
    parser.add_argument("--regimes", nargs="+", default=list(REGIMES), choices=sorted(REGIMES))
    parser.add_argument("--runner", default="reservoir_channel_cupy_direct")
    parser.add_argument("--method-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--backend", choices=("auto", "cupy", "numpy"), default="cupy")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--retune-readout", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-strategy", choices=("chronological", "random"), default="random")
    parser.add_argument("--tune-max-inner", type=int, default=1000)
    parser.add_argument("--tune-max-val", type=int, default=300)
    args = parser.parse_args()

    dataset_path = args.dataset_path or (args.data_root / args.dataset)
    for regime in args.regimes:
        suffix = regime
        method = f"quark_{args.runner}_{suffix}"
        expected_run = args.out_root / args.dataset / "*" / method / f"seed={int(args.method_seed)}"
        if args.skip_existing and any(path.joinpath("metrics.csv").exists() for path in args.out_root.glob(str(expected_run.relative_to(args.out_root)))):
            print(f"skip existing {method}")
            continue

        run_dir = run_real_quark(
            dataset_path=dataset_path,
            out_root=args.out_root,
            method_seed=args.method_seed,
            runner=args.runner,
            num_workers=args.num_workers,
            experiment_name=f"{args.dataset}_{suffix}",
            backend=args.backend,
            device=args.device,
            method_suffix=suffix,
            model_overrides=REGIMES[regime],
        )
        print(run_dir)

        if args.retune_readout:
            retune_method = f"{method}_kernel_readout_retune"
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


if __name__ == "__main__":
    main()
