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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Hydraulic QuaRK exact-feature memory/qubit sweep."
    )
    parser.add_argument("--dataset", default="hydraulic_systems")
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_TSER_DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_REAL_RESULTS_ROOT)
    parser.add_argument("--runner", default="reservoir_channel_cupy_direct")
    parser.add_argument("--method-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--backend", choices=("auto", "cupy", "numpy"), default="cupy")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-qubits", nargs="+", type=int, default=[5, 6, 7, 8])
    parser.add_argument("--lam0-grid", nargs="+", type=float, default=[0.01, 0.05, 0.1, 0.3, 0.5, 0.8])
    parser.add_argument("--num-reservoirs", type=int, default=3)
    parser.add_argument("--locality", type=int, default=2)
    parser.add_argument("--retune-readout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-strategy", choices=("chronological", "random"), default="random")
    parser.add_argument("--tune-max-inner", type=int, default=1000)
    parser.add_argument("--tune-max-val", type=int, default=300)
    args = parser.parse_args()

    dataset_path = args.dataset_path or (args.data_root / args.dataset)
    for n in args.num_qubits:
        for lam0 in args.lam0_grid:
            suffix = (
                f"exact_k{int(args.locality)}_R{int(args.num_reservoirs)}"
                f"_n{int(n)}_lam{format_lam(lam0)}"
            )
            method = f"quark_{args.runner}_{suffix}"
            if args.skip_existing and method_exists(
                args.out_root,
                dataset=args.dataset,
                method=method,
                method_seed=args.method_seed,
            ):
                print(f"skip existing {method}")
                continue

            overrides = [
                "model/qrc/features/retriever=exact",
                f"model.qrc.cfg.num_qubits={int(n)}",
                f"model.qrc.features.observables.locality={int(args.locality)}",
                f"model.qrc.pubs.num_reservoirs={int(args.num_reservoirs)}",
                f"model.qrc.pubs.lam_0={float(lam0)}",
            ]
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
                model_overrides=overrides,
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


def format_lam(value: float) -> str:
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return text


def method_exists(out_root: Path, *, dataset: str, method: str, method_seed: int) -> bool:
    pattern = f"{dataset}/split=*/{method}/seed={int(method_seed)}/metrics.csv"
    return any(out_root.glob(pattern))


if __name__ == "__main__":
    main()
