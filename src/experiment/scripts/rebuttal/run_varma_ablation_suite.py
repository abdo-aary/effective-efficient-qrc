from __future__ import annotations

import argparse
from pathlib import Path

from src.experiment.varma_ablation import (
    DEFAULT_VARMA_ABLATION_DATA_ROOT,
    DEFAULT_VARMA_ABLATION_RESULTS_ROOT,
    VARMA_ABLATION_CLASSICAL_METHODS,
    VARMA_D_GRID,
    VARMA_SHOT_GRID,
    VARMA_W_GRID,
    aggregate_varma_ablation_results,
    architecture_dataset_spec,
    architecture_regimes,
    build_varma_dry_run_actions,
    expand_varma_ablation_stages,
    prepare_varma_ablation_dataset,
    run_varma_ablation_classical,
    run_varma_ablation_quark,
    shot_regimes,
    varma_grid_regime,
    varma_grid_specs,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run fixed-N_train VARMA rebuttal ablations.")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["prepare", "classical", "aggregate"],
        choices=(
            "prepare",
            "classical",
            "quark",
            "quark_architecture",
            "quark_varma_grid",
            "shots",
            "aggregate",
            "all",
        ),
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_VARMA_ABLATION_DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_VARMA_ABLATION_RESULTS_ROOT)
    parser.add_argument("--N-train", type=int, default=5000)
    parser.add_argument("--N-test", type=int, default=1000)
    parser.add_argument("--s", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--w-grid", nargs="+", type=int, default=list(VARMA_W_GRID))
    parser.add_argument("--d-grid", nargs="+", type=int, default=list(VARMA_D_GRID))
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(VARMA_ABLATION_CLASSICAL_METHODS),
        choices=sorted(VARMA_ABLATION_CLASSICAL_METHODS),
    )
    parser.add_argument("--feature-dim", type=int, default=315)
    parser.add_argument("--backend", choices=("auto", "cupy", "numpy"), default="cupy")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--method-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--runner", default="reservoir_channel_cupy_direct")
    parser.add_argument("--w100-runner", default="reservoir_channel_cupy_direct_truncated64")
    parser.add_argument("--use-truncated-w100", action="store_true")
    parser.add_argument("--retune-readout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tune-max-inner", type=int, default=1000)
    parser.add_argument("--tune-max-val", type=int, default=300)
    parser.add_argument("--shot-grid", nargs="+", type=int, default=list(VARMA_SHOT_GRID))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    stages = expand_varma_ablation_stages(args.stages)
    arch_spec = architecture_dataset_spec(
        N_train=args.N_train,
        N_test=args.N_test,
        s=args.s,
        seed=args.seed,
        split_seed=args.split_seed,
    )
    grid_specs = varma_grid_specs(
        w_grid=args.w_grid,
        d_grid=args.d_grid,
        N_train=args.N_train,
        N_test=args.N_test,
        s=args.s,
        seed=args.seed,
        split_seed=args.split_seed,
    )
    # Ensure the fixed architecture anchor exists even if a custom grid omits w=25,d=3.
    all_specs = list({(spec.w, spec.d): spec for spec in [arch_spec, *grid_specs]}.values())

    if args.dry_run:
        for action in build_varma_dry_run_actions(
            stages=stages,
            specs=all_specs,
            methods=args.methods,
            out_root=args.out_root,
            backend=args.backend,
            retune_readout=args.retune_readout,
        ):
            print(action)
        return

    prepared: dict[tuple[int, int], Path] = {}
    if "prepare" in stages:
        for spec in all_specs:
            out_dir = prepare_varma_ablation_dataset(spec, data_root=args.data_root, overwrite=args.overwrite)
            prepared[(spec.w, spec.d)] = out_dir
            print(out_dir)

    def dataset_dir_for(spec):
        if (spec.w, spec.d) in prepared:
            return prepared[(spec.w, spec.d)]
        return prepare_varma_ablation_dataset(spec, data_root=args.data_root, overwrite=False)

    if "classical" in stages:
        for run_dir in run_varma_ablation_classical(
            dataset_dir=dataset_dir_for(arch_spec),
            out_root=args.out_root,
            ablation="architecture",
            methods=args.methods,
            feature_dim=args.feature_dim,
            method_seed=args.method_seed,
            backend=args.backend,
            device=args.device,
        ):
            print(run_dir)
        for spec in grid_specs:
            for run_dir in run_varma_ablation_classical(
                dataset_dir=dataset_dir_for(spec),
                out_root=args.out_root,
                ablation="varma_wd_scaling",
                methods=args.methods,
                feature_dim=args.feature_dim,
                method_seed=args.method_seed,
                backend=args.backend,
                device=args.device,
            ):
                print(run_dir)

    if "quark_architecture" in stages:
        dataset_dir = dataset_dir_for(arch_spec)
        for regime in architecture_regimes():
            active_regime = regime
            if args.runner != regime.runner:
                active_regime = type(regime)(
                    name=regime.name,
                    overrides=regime.overrides,
                    runner=args.runner,
                    feature_dim=regime.feature_dim,
                )
            for run_dir in run_varma_ablation_quark(
                dataset_dir=dataset_dir,
                out_root=args.out_root,
                ablation="architecture",
                regime=active_regime,
                method_seed=args.method_seed,
                backend=args.backend,
                device=args.device,
                num_workers=args.num_workers,
                retune_readout=args.retune_readout,
                tune_max_inner=args.tune_max_inner,
                tune_max_val=args.tune_max_val,
            ):
                print(run_dir)

    if "quark_varma_grid" in stages:
        for spec in grid_specs:
            runner = args.w100_runner if args.use_truncated_w100 and int(spec.w) == 100 else args.runner
            regime = varma_grid_regime(runner=runner)
            method_name = regime.name
            if runner != args.runner:
                method_name = f"{method_name}_{runner}"
                regime = type(regime)(
                    name=method_name,
                    overrides=regime.overrides,
                    runner=runner,
                    feature_dim=regime.feature_dim,
                )
            for run_dir in run_varma_ablation_quark(
                dataset_dir=dataset_dir_for(spec),
                out_root=args.out_root,
                ablation="varma_wd_scaling",
                regime=regime,
                method_seed=args.method_seed,
                backend=args.backend,
                device=args.device,
                num_workers=args.num_workers,
                retune_readout=args.retune_readout,
                tune_max_inner=args.tune_max_inner,
                tune_max_val=args.tune_max_val,
            ):
                print(run_dir)

    if "shots" in stages:
        dataset_dir = dataset_dir_for(arch_spec)
        for regime in shot_regimes(args.shot_grid):
            active_regime = regime
            if args.runner != regime.runner:
                active_regime = type(regime)(
                    name=regime.name,
                    overrides=regime.overrides,
                    runner=args.runner,
                    feature_dim=regime.feature_dim,
                )
            for run_dir in run_varma_ablation_quark(
                dataset_dir=dataset_dir,
                out_root=args.out_root,
                ablation="finite_shots",
                regime=active_regime,
                method_seed=args.method_seed,
                backend=args.backend,
                device=args.device,
                num_workers=args.num_workers,
                retune_readout=False,
            ):
                print(run_dir)

    if "aggregate" in stages:
        aggregate_dir = aggregate_varma_ablation_results(args.out_root)
        print(aggregate_dir)


if __name__ == "__main__":
    main()
