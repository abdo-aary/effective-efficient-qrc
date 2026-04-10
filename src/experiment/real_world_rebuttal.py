"""Real-world TSER rebuttal experiment utilities."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import Matern as SkMatern

from src.compute.backend import asnumpy, resolve_backend
from src.data.factory import load_windows_dataset
from src.experiment.temporal_baselines import (
    METRICS_COLUMNS,
    RIDGE_ALPHA_GRID,
    REG_GRID,
    BenchmarkData,
    SplitData,
    TEMPORAL_BASELINE_METHODS,
    build_markdown_table,
    fit_esn_matern_krr,
    fit_esn_matern_krr_from_saved_params,
    fit_esn_ridge,
    json_safe,
    make_metric_rows,
    method_run_dir,
    mean_squared_error,
    matern_nu_grid_for_backend,
    standardize_flattened_windows,
    write_baseline_plot,
    write_dict_csv,
    write_metrics_csv,
    write_run_artifacts,
)
from src.experiment.experiment import Experiment
from src.models.gpu_krr import GPU_MATERN_NU_GRID
from src.models.gpu_krr import krr_lambda_sweep as gpu_krr_lambda_sweep
from src.models.gpu_krr import matern_kernel_matrix as gpu_matern_kernel_matrix
from src.models.gpu_krr import solve_krr as gpu_solve_krr
from src.settings import PROJECT_ROOT_PATH


PROJECT_ROOT = Path(PROJECT_ROOT_PATH)
DEFAULT_REAL_DATA_ROOT = PROJECT_ROOT / "storage/data/real/tser"
DEFAULT_REAL_RESULTS_ROOT = PROJECT_ROOT / "storage/results/rebuttal/real_world"
DEFAULT_REAL_TRIAGE_RESULTS_ROOT = PROJECT_ROOT / "storage/results/rebuttal/real_world_triage"
DEFAULT_REAL_RESPONSE_DIR = PROJECT_ROOT / "docs/rebuttal/responses"
QUARK_READOUT_RETUNE_METHOD = "quark_reservoir_channel_readout_retune"
QUARK_KERNEL_READOUT_RETUNE_METHOD = "quark_reservoir_channel_kernel_readout_retune"
READOUT_RETUNE_LAMBDA_GRID = 10.0 ** np.arange(-6.0, 10.5, 0.5)
READOUT_RETUNE_XI_GRID = 10.0 ** np.arange(-1.0, 3.5, 0.5)
READOUT_RETUNE_NU_GRID = np.asarray([0.5, 1.5, 2.5, 5.0], dtype=float)


def run_real_classical_baseline(
    method: str,
    *,
    dataset_path: str | Path,
    out_root: str | Path = DEFAULT_REAL_RESULTS_ROOT,
    method_seed: int = 0,
    feature_dim: int = 315,
    reuse_esn_source_run: str | Path | None = None,
    backend: str = "auto",
    device: int | None = 0,
) -> Path:
    method = str(method)
    if method not in TEMPORAL_BASELINE_METHODS:
        raise ValueError(f"Unknown temporal baseline method={method!r}.")

    t0 = time.perf_counter()
    dataset_path = resolve_real_dataset_path(dataset_path)
    dataset = load_real_benchmark_data(dataset_path)
    split = load_real_split(dataset_path)
    run_root = real_dataset_output_root(out_root, dataset.dataset_id, split.split_source)
    run_dir = method_run_dir(run_root, method, method_seed=method_seed, feature_dim=feature_dim)
    run_dir.mkdir(parents=True, exist_ok=True)
    _, raw_scaler = standardize_flattened_windows(dataset.X, split.train_idx)

    if method == "esn":
        result = fit_esn_ridge(
            dataset.X,
            dataset.y2d,
            split,
            dataset.task_names,
            method_seed=method_seed,
            feature_dim=feature_dim,
            backend=backend,
            device=device,
        )
        final_feature_dim = feature_dim
    elif method == "esn_matern_krr":
        if reuse_esn_source_run is None:
            result = fit_esn_matern_krr(
                dataset.X,
                dataset.y2d,
                split,
                dataset.task_names,
                method_seed=method_seed,
                feature_dim=feature_dim,
                backend=backend,
                device=device,
            )
        else:
            source_run = Path(reuse_esn_source_run)
            source_params = json.loads((source_run / "best_params.json").read_text(encoding="utf-8"))
            result = fit_esn_matern_krr_from_saved_params(
                dataset.X,
                dataset.y2d,
                split,
                dataset.task_names,
                method_seed=method_seed,
                feature_dim=feature_dim,
                saved_best_params=source_params,
                backend=backend,
                device=device,
            )
        final_feature_dim = feature_dim
    else:  # pragma: no cover
        raise AssertionError(method)

    rows = make_real_metric_rows(
        result=result,
        dataset=dataset,
        split=split,
        method=method,
        method_seed=method_seed,
        feature_dim=final_feature_dim,
        artifact_dir=run_dir,
    )
    write_run_artifacts(
        run_dir=run_dir,
        rows=rows,
        result=result,
        run_config={
            "method": method,
            "dataset_path": str(Path(dataset_path)),
            "out_root": str(Path(out_root)),
            "method_seed": int(method_seed),
            "feature_dim": int(feature_dim),
            "reuse_esn_source_run": None if reuse_esn_source_run is None else str(Path(reuse_esn_source_run)),
            "backend": str(backend),
            "device": None if device is None else int(device),
            "raw_dim": int(dataset.raw_dim),
            "n_train": int(split.train_idx.size),
            "n_test": int(split.test_idx.size),
            "split_source": split.split_source,
            "raw_scaler_mean_shape": list(raw_scaler.mean_.shape),
            "ridge_alpha_grid": RIDGE_ALPHA_GRID.tolist(),
            "reg_grid": REG_GRID.tolist(),
        },
        summary_extra={
            "duration_seconds": float(time.perf_counter() - t0),
            "dataset": dataset.dataset_id,
            "task_names": dataset.task_names,
            "primary_metric": "test_nrmse_train_y_std",
        },
    )
    return run_dir


def run_real_quark(
    *,
    dataset_path: str | Path,
    out_root: str | Path = DEFAULT_REAL_RESULTS_ROOT,
    method_seed: int = 0,
    runner: str = "reservoir_channel",
    num_workers: int = 1,
    experiment_name: str | None = None,
    backend: str = "auto",
    device: int | None = 0,
    method_suffix: str | None = None,
    model_overrides: Sequence[str] = (),
) -> Path:
    """Run QuaRK on one prepared real-world TSER dataset using the official split."""

    t0 = time.perf_counter()
    dataset_path = resolve_real_dataset_path(dataset_path)
    dataset = load_real_benchmark_data(dataset_path)
    split = load_real_split(dataset_path)
    method = f"quark_{runner}{normalize_method_suffix(method_suffix)}"
    run_root = real_dataset_output_root(out_root, dataset.dataset_id, split.split_source)
    run_dir = run_root / method / f"seed={int(method_seed)}"
    model_dir = run_dir / "model"
    artifacts_dir = run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = compose_real_quark_config(
        dataset_path=dataset_path,
        runner=runner,
        experiment_name=experiment_name,
        N=int(dataset.X.shape[0]),
        w=int(dataset.X.shape[1]),
        d=int(dataset.X.shape[2]),
        seed=int(method_seed),
        backend=backend,
        device=device,
        model_overrides=model_overrides,
    )
    exp = Experiment.from_paths(
        dataset_path=dataset_path,
        model_cfg=cfg.model,
        instantiate_functionals=False,
    )
    exp.model.train_idx_ = split.train_idx
    exp.model.test_idx_ = split.test_idx
    exp.fit(num_workers=int(num_workers))
    exp.save_model(model_dir)

    result = make_quark_result(exp, dataset, split, method=method, method_seed=method_seed)
    rows = make_real_metric_rows(
        result=result,
        dataset=dataset,
        split=split,
        method=method,
        method_seed=method_seed,
        feature_dim=int(exp.model.Phi_full_.shape[1]),
        artifact_dir=run_dir,
    )
    write_run_artifacts(
        run_dir=run_dir,
        rows=rows,
        result=result,
        run_config={
            "method": method,
            "dataset_path": str(Path(dataset_path)),
            "out_root": str(Path(out_root)),
            "runner": runner,
            "method_seed": int(method_seed),
            "n_train": int(split.train_idx.size),
            "n_test": int(split.test_idx.size),
            "split_source": split.split_source,
            "model_dir": str(model_dir),
            "backend": str(backend),
            "device": None if device is None else int(device),
            "method_suffix": method_suffix,
            "model_overrides": list(model_overrides),
            "hydra_model_config": OmegaConf.to_container(cfg.model, resolve=True),
        },
        summary_extra={
            "duration_seconds": float(time.perf_counter() - t0),
            "dataset": dataset.dataset_id,
            "task_names": dataset.task_names,
            "primary_metric": "test_nrmse_train_y_std",
        },
    )
    # Save the resolved model config separately for reproducibility.
    with (run_dir / "model_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(json_safe(OmegaConf.to_container(cfg.model, resolve=True)), f, sort_keys=False)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_cached_quark_readout_retune(
    *,
    dataset_path: str | Path,
    source_run: str | Path,
    out_root: str | Path = DEFAULT_REAL_RESULTS_ROOT,
    method_seed: int = 0,
    val_ratio: float = 0.2,
    split_strategy: str = "chronological",
    lambda_grid: Sequence[float] = READOUT_RETUNE_LAMBDA_GRID,
    xi_grid: Sequence[float] | None = None,
    nu_grid: Sequence[float] = READOUT_RETUNE_NU_GRID,
    tune_max_inner: int | None = None,
    tune_max_val: int | None = None,
    method: str | None = None,
    backend: str = "auto",
    device: int | None = 0,
) -> Path:
    """Retune KRR readout hyperparameters from cached QuaRK features."""

    t0 = time.perf_counter()
    dataset_path = resolve_real_dataset_path(dataset_path)
    dataset = load_real_benchmark_data(dataset_path)
    split = load_real_split(dataset_path)

    source_run = Path(source_run)
    model_dir = resolve_cached_quark_model_dir(source_run)
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    with np.load(model_dir / "arrays.npz", allow_pickle=False) as z:
        Phi_full = np.asarray(z["Phi_full"], dtype=float)
        cached_train_idx = np.asarray(z["train_idx"], dtype=int).reshape(-1)
        cached_test_idx = np.asarray(z["test_idx"], dtype=int).reshape(-1)
        scaler_mean = np.asarray(z["scaler_mean"], dtype=float) if "scaler_mean" in z.files else None
        scaler_scale = np.asarray(z["scaler_scale"], dtype=float) if "scaler_scale" in z.files else None

    if Phi_full.shape[0] != dataset.X.shape[0]:
        raise ValueError(f"Cached Phi_full has N={Phi_full.shape[0]}, dataset has N={dataset.X.shape[0]}.")
    if not np.array_equal(cached_train_idx, split.train_idx) or not np.array_equal(cached_test_idx, split.test_idx):
        raise ValueError("Cached QuaRK split does not match the official real-world split.")

    Phi_work = Phi_full
    if bool(meta.get("standardize", False)):
        if scaler_mean is None or scaler_scale is None:
            raise ValueError("Cached model is standardized but arrays.npz is missing scaler_mean/scaler_scale.")
        scale = np.where(scaler_scale == 0.0, 1.0, scaler_scale)
        Phi_work = (Phi_full - scaler_mean.reshape(1, -1)) / scale.reshape(1, -1)

    best_params = meta.get("best_params_")
    if isinstance(best_params, list):
        params_by_task = [dict(p) for p in best_params]
    elif isinstance(best_params, dict):
        params_by_task = [dict(best_params) for _ in dataset.task_names]
    else:
        raise ValueError(f"Cached model meta.json has invalid best_params_={best_params!r}.")

    if len(params_by_task) != len(dataset.task_names):
        if len(params_by_task) == 1:
            params_by_task = [dict(params_by_task[0]) for _ in dataset.task_names]
        else:
            raise ValueError(
                f"Cached best_params_ has {len(params_by_task)} entries, expected {len(dataset.task_names)}."
            )

    reg_grid = np.asarray(lambda_grid, dtype=float).reshape(-1)
    if reg_grid.size == 0 or np.any(~np.isfinite(reg_grid)) or np.any(reg_grid <= 0.0):
        raise ValueError("lambda_grid must be non-empty with finite positive entries.")
    if xi_grid is None:
        retune_kernel = False
        xi_arr: np.ndarray | None = None
    else:
        retune_kernel = True
        xi_arr = np.asarray(xi_grid, dtype=float).reshape(-1)
        if xi_arr.size == 0 or np.any(~np.isfinite(xi_arr)) or np.any(xi_arr <= 0.0):
            raise ValueError("xi_grid must be non-empty with finite positive entries.")
    nu_arr = np.asarray(nu_grid, dtype=float).reshape(-1)
    if nu_arr.size == 0 or np.any(~np.isfinite(nu_arr)) or np.any(nu_arr <= 0.0):
        raise ValueError("nu_grid must be non-empty with finite positive entries.")
    if retune_kernel:
        nu_arr = matern_nu_grid_for_backend(backend, nu_arr)
    if method is None:
        method = QUARK_KERNEL_READOUT_RETUNE_METHOD if retune_kernel else QUARK_READOUT_RETUNE_METHOD

    inner_train_idx, val_idx = split_cached_train_validation(
        split.train_idx,
        val_ratio=val_ratio,
        split_strategy=split_strategy,
        seed=method_seed,
    )
    tune_inner_idx = cap_tuning_indices(
        inner_train_idx,
        max_size=tune_max_inner,
        strategy=split_strategy,
        seed=method_seed,
    )
    tune_val_idx = cap_tuning_indices(
        val_idx,
        max_size=tune_max_val,
        strategy=split_strategy,
        seed=method_seed + 1,
    )

    y_train_pred = np.empty((dataset.y2d.shape[0], split.train_idx.size), dtype=float)
    y_test_pred = np.empty((dataset.y2d.shape[0], split.test_idx.size), dtype=float)
    curve_rows: list[dict[str, Any]] = []
    normalized_params: list[dict[str, Any]] = []

    for task_i, task_name in enumerate(dataset.task_names):
        params = dict(params_by_task[task_i])
        candidate_xi = xi_arr if xi_arr is not None else np.asarray([float(params["xi"])], dtype=float)
        candidate_nu = nu_arr if retune_kernel else np.asarray([float(params["nu"])], dtype=float)

        Phi_inner = Phi_work[tune_inner_idx]
        Phi_val = Phi_work[tune_val_idx]
        y_inner = dataset.y2d[task_i, tune_inner_idx].reshape(-1)
        y_val = dataset.y2d[task_i, tune_val_idx].reshape(-1)

        best: dict[str, Any] | None = None
        for xi in candidate_xi:
            for nu in candidate_nu:
                if cached_retune_can_use_gpu(backend, float(nu)):
                    K_inner = gpu_matern_kernel_matrix(
                        Phi_inner,
                        Phi_inner,
                        xi=float(xi),
                        nu=float(nu),
                        backend=backend,
                        device=device,
                    )
                    K_val_inner = gpu_matern_kernel_matrix(
                        Phi_val,
                        Phi_inner,
                        xi=float(xi),
                        nu=float(nu),
                        backend=backend,
                        device=device,
                    )
                    sweep = gpu_krr_lambda_sweep(
                        K_inner,
                        K_val_inner,
                        y_inner,
                        y_val,
                        reg_grid,
                        backend=backend,
                    )
                else:
                    kernel = build_fixed_matern_kernel(xi=float(xi), nu=float(nu))
                    K_inner = kernel(Phi_inner, Phi_inner)
                    K_val_inner = kernel(Phi_val, Phi_inner)
                    sweep = eigensolve_krr_lambda_sweep(
                        K_train=K_inner,
                        K_eval_train=K_val_inner,
                        y_train=y_inner,
                        y_eval=y_val,
                        lambda_grid=reg_grid,
                    )
                best_idx = int(np.argmin(sweep["eval_mse"]))
                score = float(sweep["eval_mse"][best_idx])
                for i, lam in enumerate(reg_grid):
                    curve_rows.append(
                        {
                            "task": task_name,
                            "xi": float(xi),
                            "nu": float(nu),
                            "lambda": float(lam),
                            "train_inner_mse": float(sweep["train_mse"][i]),
                            "val_mse": float(sweep["eval_mse"][i]),
                            "selected": 0,
                        }
                    )
                if best is None or score < float(best["score"]):
                    best = {
                        "xi": float(xi),
                        "nu": float(nu),
                        "lambda": float(reg_grid[best_idx]),
                        "score": score,
                    }

        if best is None:  # pragma: no cover - protected by grid validation
            raise RuntimeError("No cached readout hyperparameter candidate was evaluated.")
        xi = float(best["xi"])
        nu = float(best["nu"])
        selected_lambda = float(best["lambda"])

        for row in curve_rows:
            if (
                row["task"] == task_name
                and float(row["xi"]) == xi
                and float(row["nu"]) == nu
                and float(row["lambda"]) == selected_lambda
            ):
                row["selected"] = 1

        Phi_train = Phi_work[split.train_idx]
        Phi_test = Phi_work[split.test_idx]
        y_train = dataset.y2d[task_i, split.train_idx].reshape(-1)
        if cached_retune_can_use_gpu(backend, nu):
            K_train = gpu_matern_kernel_matrix(
                Phi_train,
                Phi_train,
                xi=xi,
                nu=nu,
                backend=backend,
                device=device,
            )
            alpha = gpu_solve_krr(K_train, y_train, reg=selected_lambda, backend=backend)
            y_train_pred[task_i] = asnumpy(K_train @ alpha)
            K_test = gpu_matern_kernel_matrix(
                Phi_test,
                Phi_train,
                xi=xi,
                nu=nu,
                backend=backend,
                device=device,
            )
            y_test_pred[task_i] = asnumpy(K_test @ alpha)
        else:
            kernel = build_fixed_matern_kernel(xi=xi, nu=nu)
            K_train = kernel(Phi_train, Phi_train)
            alpha = np.linalg.solve(
                K_train + selected_lambda * np.eye(K_train.shape[0], dtype=K_train.dtype),
                y_train,
            )
            y_train_pred[task_i] = K_train @ alpha
            y_test_pred[task_i] = kernel(Phi_test, Phi_train) @ alpha

        normalized_params.append(
            {
                "task": task_name,
                "selected_alpha": None,
                "selected_lambda": selected_lambda,
                "selected_gamma": None,
                "selected_xi": xi,
                "selected_nu": nu,
                "selected_spectral_radius": None,
                "selected_input_scale": None,
                "selected_leak_rate": None,
                "val_score": float(best["score"]),
                "source_lambda": params.get("reg"),
                "source_xi": params.get("xi"),
                "source_nu": params.get("nu"),
            }
        )

    run_root = real_dataset_output_root(out_root, dataset.dataset_id, split.split_source)
    run_dir = run_root / method / f"seed={int(method_seed)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "method": method,
        "method_seed": int(method_seed),
        "task_names": list(dataset.task_names),
        "feature_dim": int(Phi_full.shape[1]),
        "raw_dim": int(dataset.raw_dim),
        "best_params": normalized_params,
        "train_mse": np.asarray(
            [
                mean_squared_error(dataset.y2d[i, split.train_idx], y_train_pred[i])
                for i in range(dataset.y2d.shape[0])
            ],
            dtype=float,
        ),
        "test_mse": np.asarray(
            [
                mean_squared_error(dataset.y2d[i, split.test_idx], y_test_pred[i])
                for i in range(dataset.y2d.shape[0])
            ],
            dtype=float,
        ),
        "y_train_true": dataset.y2d[:, split.train_idx],
        "y_train_pred": y_train_pred,
        "y_test_true": dataset.y2d[:, split.test_idx],
        "y_test_pred": y_test_pred,
    }
    rows = make_real_metric_rows(
        result=result,
        dataset=dataset,
        split=split,
        method=method,
        method_seed=method_seed,
        feature_dim=int(Phi_full.shape[1]),
        artifact_dir=run_dir,
    )
    write_run_artifacts(
        run_dir=run_dir,
        rows=rows,
        result=result,
        run_config={
            "method": method,
            "dataset_path": str(dataset_path),
            "source_run": str(source_run),
            "source_model_dir": str(model_dir),
            "out_root": str(Path(out_root)),
            "method_seed": int(method_seed),
            "val_ratio": float(val_ratio),
            "split_strategy": split_strategy,
            "lambda_grid": reg_grid.tolist(),
            "inner_train_size": int(inner_train_idx.size),
            "validation_size": int(val_idx.size),
            "tune_inner_size": int(tune_inner_idx.size),
            "tune_validation_size": int(tune_val_idx.size),
            "xi_grid": None if xi_arr is None else xi_arr.tolist(),
            "nu_grid": nu_arr.tolist(),
            "retune_kernel": bool(retune_kernel),
            "feature_dim": int(Phi_full.shape[1]),
            "source_best_params": best_params,
            "standardize_from_cached_model": bool(meta.get("standardize", False)),
            "backend": str(backend),
            "device": None if device is None else int(device),
            "gpu_matern_nu_grid": list(GPU_MATERN_NU_GRID),
        },
        summary_extra={
            "duration_seconds": float(time.perf_counter() - t0),
            "dataset": dataset.dataset_id,
            "task_names": dataset.task_names,
            "primary_metric": "test_nrmse_train_y_std",
            "curve_path": str(run_dir / "readout_retune_curve.csv"),
        },
    )
    write_readout_retune_curve(run_dir / "readout_retune_curve.csv", curve_rows)
    write_readout_retune_curve_plot(run_dir / "readout_retune_curve.pdf", curve_rows)
    np.savez_compressed(
        run_dir / "readout_retune_indices.npz",
        inner_train_idx=inner_train_idx,
        val_idx=val_idx,
        test_idx=split.test_idx,
    )
    return run_dir


def cached_retune_can_use_gpu(backend: str, nu: float) -> bool:
    if resolve_backend(backend) != "cupy":
        return False
    if any(np.isclose(float(nu), allowed) for allowed in GPU_MATERN_NU_GRID):
        return True
    if str(backend) == "auto":
        return False
    raise ValueError(f"GPU Matérn supports nu in {GPU_MATERN_NU_GRID}, got {nu}.")


def aggregate_real_world_results(
    out_root: str | Path = DEFAULT_REAL_RESULTS_ROOT,
    *,
    response_dir: str | Path = DEFAULT_REAL_RESPONSE_DIR,
    metric: str = "nrmse_train_y_std",
    include_quark: bool = True,
) -> Path:
    out_root = Path(out_root)
    aggregate_dir = out_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    rows = read_real_metric_rows(out_root, include_quark=include_quark)

    if not rows:
        raise FileNotFoundError(f"No metrics.csv files found under {out_root}.")

    long_path = aggregate_dir / "real_world_long.csv"
    write_metrics_csv(long_path, rows)
    wide_rows = build_real_wide_rows(rows, metric=metric)
    wide_path = aggregate_dir / "real_world_wide.csv"
    write_dict_csv(wide_path, wide_rows)
    table_path = aggregate_dir / "real_world_table.md"
    table_md = build_real_markdown_table(wide_rows, metric=metric)
    table_path.write_text(table_md, encoding="utf-8")
    write_baseline_plot(
        [
            {
                "method": row["method"],
                "task": row["dataset"],
                "test_mse": row.get(f"test_{metric}", ""),
            }
            for row in wide_rows
        ],
        aggregate_dir / "real_world_plot.pdf",
    )
    dataset_cards = build_dataset_cards(out_root)
    (aggregate_dir / "dataset_cards.md").write_text(dataset_cards, encoding="utf-8")
    update_real_world_response_drafts(
        response_dir=Path(response_dir),
        aggregate_dir=aggregate_dir,
        table_markdown=table_md,
        dataset_cards=dataset_cards,
        metric=metric,
    )
    return aggregate_dir


def read_real_metric_rows(
    out_root: str | Path,
    *,
    include_quark: bool = True,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for metrics_path in sorted(Path(out_root).glob("**/metrics.csv")):
        if "aggregate" in metrics_path.parts:
            continue
        with metrics_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not include_quark and str(row.get("method", "")).startswith("quark_"):
                    continue
                rows.append({col: row.get(col, "") for col in METRICS_COLUMNS})
    return rows


def load_real_benchmark_data(dataset_path: str | Path) -> BenchmarkData:
    dataset_path = resolve_real_dataset_path(dataset_path)
    ds, artifact = load_windows_dataset(dataset_path, instantiate_functionals=False)
    X = np.asarray(ds.X, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"Expected window tensor with shape (N,w,d), got {X.shape}.")
    N, w, d = X.shape
    y = np.asarray(ds.y, dtype=float)
    if y.ndim == 1:
        y2d = y.reshape(1, -1)
    elif y.ndim == 2 and y.shape[1] == N:
        y2d = y
    elif y.ndim == 2 and y.shape[0] == N:
        y2d = y.T
    else:
        raise ValueError(f"Cannot align y={y.shape} with X={X.shape}.")
    meta = dict(ds.meta)
    dataset_id = str(meta.get("dataset_id") or artifact.root.name)
    task_names = list(meta.get("task_names") or [str(meta.get("target_name") or dataset_id)])
    if len(task_names) != y2d.shape[0]:
        task_names = [f"{dataset_id}_target_{i}" for i in range(y2d.shape[0])]
    return BenchmarkData(
        X=X,
        y2d=y2d,
        task_names=task_names,
        dataset_id=dataset_id,
        raw_dim=int(w * d),
        artifact_root=artifact.root,
        meta=meta,
    )


def load_real_split(dataset_path: str | Path) -> SplitData:
    dataset_path = resolve_real_dataset_path(dataset_path)
    ds, artifact = load_windows_dataset(dataset_path, instantiate_functionals=False)
    split_path = artifact.root / "split.npz"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing real-world split sidecar: {split_path}")
    with np.load(split_path) as z:
        train_idx = np.asarray(z["train_idx"], dtype=int).reshape(-1)
        test_idx = np.asarray(z["test_idx"], dtype=int).reshape(-1)
    N = int(np.asarray(ds.X).shape[0])
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("Official split must have non-empty train/test indices.")
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError("Official train/test split indices overlap.")
    if train_idx.max(initial=-1) >= N or test_idx.max(initial=-1) >= N:
        raise ValueError(f"Official split index out of range for N={N}.")
    split_source = str(dict(ds.meta).get("split_source") or "official_tser")
    return SplitData(train_idx=train_idx, test_idx=test_idx, split_source=split_source)


def split_cached_train_validation(
    train_idx: np.ndarray,
    *,
    val_ratio: float,
    split_strategy: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_idx = np.asarray(train_idx, dtype=int).reshape(-1)
    if train_idx.size < 3:
        raise ValueError("Need at least 3 training samples for a cached readout train/validation split.")
    val_ratio = float(val_ratio)
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}.")

    n_val = max(1, int(round(val_ratio * train_idx.size)))
    if n_val >= train_idx.size:
        n_val = train_idx.size - 1

    if split_strategy == "chronological":
        order = np.argsort(train_idx)
        val_local = order[-n_val:]
        inner_local = order[:-n_val]
    elif split_strategy == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(train_idx.size)
        val_local = perm[:n_val]
        inner_local = perm[n_val:]
    else:
        raise ValueError(f"Unknown split_strategy={split_strategy!r}; expected 'chronological' or 'random'.")

    inner_train_idx = np.asarray(train_idx[inner_local], dtype=int).reshape(-1)
    val_idx = np.asarray(train_idx[val_local], dtype=int).reshape(-1)
    if np.intersect1d(inner_train_idx, val_idx).size:
        raise RuntimeError("Internal cached readout train/validation split overlaps.")
    return inner_train_idx, val_idx


def cap_tuning_indices(
    idx: np.ndarray,
    *,
    max_size: int | None,
    strategy: str,
    seed: int,
) -> np.ndarray:
    idx = np.asarray(idx, dtype=int).reshape(-1)
    if max_size is None:
        return idx
    max_size = int(max_size)
    if max_size <= 0:
        raise ValueError(f"max_size must be positive or None, got {max_size}.")
    if idx.size <= max_size:
        return idx
    if strategy == "chronological":
        return np.sort(idx)[-max_size:]
    if strategy == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(idx, size=max_size, replace=False))
    raise ValueError(f"Unknown strategy={strategy!r}; expected 'chronological' or 'random'.")


def build_fixed_matern_kernel(*, xi: float, nu: float):
    return ConstantKernel(1.0, constant_value_bounds="fixed") * SkMatern(
        length_scale=float(xi),
        length_scale_bounds="fixed",
        nu=float(nu),
    )


def eigensolve_krr_lambda_sweep(
    *,
    K_train: np.ndarray,
    K_eval_train: np.ndarray,
    y_train: np.ndarray,
    y_eval: np.ndarray,
    lambda_grid: Sequence[float],
) -> dict[str, np.ndarray]:
    """Evaluate a KRR lambda sweep from a single eigendecomposition of K_train."""

    K_train = np.asarray(K_train, dtype=float)
    K_eval_train = np.asarray(K_eval_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    y_eval = np.asarray(y_eval, dtype=float).reshape(-1)
    reg_grid = np.asarray(lambda_grid, dtype=float).reshape(-1)

    if K_train.ndim != 2 or K_train.shape[0] != K_train.shape[1]:
        raise ValueError(f"K_train must be square, got {K_train.shape}.")
    if K_train.shape[0] != y_train.size:
        raise ValueError(f"K_train/y_train mismatch: {K_train.shape} vs {y_train.shape}.")
    if K_eval_train.shape != (y_eval.size, y_train.size):
        raise ValueError(
            f"K_eval_train must have shape {(y_eval.size, y_train.size)}, got {K_eval_train.shape}."
        )
    if reg_grid.size == 0 or np.any(~np.isfinite(reg_grid)) or np.any(reg_grid <= 0.0):
        raise ValueError("lambda_grid must be non-empty with finite positive entries.")

    eigvals, eigvecs = np.linalg.eigh(K_train)
    projected_y = eigvecs.T @ y_train
    train_mse = np.empty(reg_grid.size, dtype=float)
    eval_mse = np.empty(reg_grid.size, dtype=float)
    alpha_grid = np.empty((reg_grid.size, y_train.size), dtype=float)

    for i, reg in enumerate(reg_grid):
        alpha = eigvecs @ (projected_y / (eigvals + float(reg)))
        alpha_grid[i] = alpha
        train_pred = K_train @ alpha
        eval_pred = K_eval_train @ alpha
        train_mse[i] = mean_squared_error(y_train, train_pred)
        eval_mse[i] = mean_squared_error(y_eval, eval_pred)

    return {
        "lambda_grid": reg_grid,
        "alpha_grid": alpha_grid,
        "train_mse": train_mse,
        "eval_mse": eval_mse,
    }


def write_readout_retune_curve(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["task", "xi", "nu", "lambda", "train_inner_mse", "val_mse", "selected"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_readout_retune_curve_plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment guard
        path.with_suffix(".plot_error.txt").write_text(str(exc), encoding="utf-8")
        return

    tasks = sorted({str(row["task"]) for row in rows})
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for task in tasks:
        task_rows = [row for row in rows if str(row["task"]) == task]
        selected = [row for row in task_rows if int(row.get("selected", 0)) == 1]
        if selected:
            selected_xi = float(selected[0].get("xi", np.nan))
            selected_nu = float(selected[0].get("nu", np.nan))
            task_rows = [
                row
                for row in task_rows
                if float(row.get("xi", selected_xi)) == selected_xi
                and float(row.get("nu", selected_nu)) == selected_nu
            ]
            label = f"{task} (xi={selected_xi:g}, nu={selected_nu:g})"
        else:
            label = task
        lam = np.asarray([float(row["lambda"]) for row in task_rows], dtype=float)
        val_mse = np.asarray([float(row["val_mse"]) for row in task_rows], dtype=float)
        order = np.argsort(lam)
        ax.plot(lam[order], val_mse[order], marker="o", markersize=3, linewidth=1.2, label=label)
        if selected:
            ax.scatter(
                [float(selected[0]["lambda"])],
                [float(selected[0]["val_mse"])],
                s=50,
                marker="x",
                color=ax.lines[-1].get_color(),
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("KRR lambda")
    ax.set_ylabel("Validation MSE")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def resolve_cached_quark_model_dir(source_run: str | Path) -> Path:
    source = Path(source_run)
    if (source / "arrays.npz").exists() and (source / "meta.json").exists():
        return source
    model_dir = source / "model"
    if (model_dir / "arrays.npz").exists() and (model_dir / "meta.json").exists():
        return model_dir
    raise FileNotFoundError(f"Could not find cached QuaRK model arrays/meta under {source}.")


def resolve_real_dataset_path(dataset_path: str | Path) -> Path:
    path = Path(dataset_path)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Real-world dataset path does not exist: {path}")
    candidates = sorted(p for p in path.glob("*.npz") if p.name != "split.npz")
    if len(candidates) == 1:
        return candidates[0]
    latest_path = path / "latest_dataset.txt"
    if latest_path.exists():
        latest_name = latest_path.read_text(encoding="utf-8").strip()
        latest_candidate = path / latest_name
        if latest_candidate in candidates and latest_candidate.exists():
            return latest_candidate
        raise ValueError(
            f"{latest_path} points to {latest_name!r}, but available dataset files are {candidates}."
        )
    if candidates:
        # Backward-compatible fallback for directories created before the latest
        # pointer existed. Prefer the newest artifact, which is normally the
        # full dataset generated after a capped smoke run.
        return max(candidates, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(f"No real dataset .npz found under {path}.")


def real_dataset_output_root(out_root: str | Path, dataset_id: str, split_source: str) -> Path:
    return Path(out_root) / dataset_id / f"split={split_source}"


def make_real_metric_rows(
    *,
    result: Mapping[str, Any],
    dataset: BenchmarkData,
    split: SplitData,
    method: str,
    method_seed: int,
    feature_dim: int,
    artifact_dir: Path,
) -> list[dict[str, Any]]:
    rows = make_metric_rows(
        result=result,
        dataset=dataset,
        split=split,
        method=method,
        method_seed=method_seed,
        feature_dim=feature_dim,
        artifact_dir=artifact_dir,
    )
    y_train_true = np.asarray(result["y_train_true"], dtype=float)
    y_test_true = np.asarray(result["y_test_true"], dtype=float)
    y_train_pred = np.asarray(result["y_train_pred"], dtype=float)
    y_test_pred = np.asarray(result["y_test_pred"], dtype=float)
    train_std = np.std(y_train_true, axis=1)
    train_std = np.where(train_std > 0.0, train_std, 1.0)
    best_params = list(result["best_params"])
    for task_i, task_name in enumerate(result["task_names"]):
        params = dict(best_params[task_i])
        for split_name, y_true, y_pred in (
            ("train", y_train_true[task_i], y_train_pred[task_i]),
            ("test", y_test_true[task_i], y_test_pred[task_i]),
        ):
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            for metric_name, value in (
                ("rmse", rmse),
                ("nrmse_train_y_std", rmse / float(train_std[task_i])),
            ):
                row = {col: "" for col in METRICS_COLUMNS}
                row.update(
                    {
                        "experiment_id": f"{dataset.dataset_id}__{split.split_source}__{method}",
                        "method": method,
                        "dataset": dataset.dataset_id,
                        "task": task_name,
                        "split": split_name,
                        "metric": metric_name,
                        "value": value,
                        "n_train": int(split.train_idx.size),
                        "n_test": int(split.test_idx.size),
                        "feature_dim": int(feature_dim),
                        "raw_dim": int(dataset.raw_dim),
                        "method_seed": int(method_seed),
                        "split_source": split.split_source,
                        "val_seed": int(method_seed),
                        "artifact_dir": str(artifact_dir),
                    }
                )
                for key in (
                    "selected_alpha",
                    "selected_lambda",
                    "selected_gamma",
                    "selected_xi",
                    "selected_nu",
                    "selected_spectral_radius",
                    "selected_input_scale",
                    "selected_leak_rate",
                ):
                    value = params.get(key)
                    row[key] = "" if value is None else value
                rows.append(row)
    return rows


def make_quark_result(
    exp: Experiment,
    dataset: BenchmarkData,
    split: SplitData,
    *,
    method: str,
    method_seed: int,
) -> dict[str, Any]:
    model = exp.model
    Phi = np.asarray(model.Phi_full_, dtype=float)
    Phi_train = Phi[split.train_idx]
    if model.scaler_ is not None:
        Phi_train = model.scaler_.transform(Phi_train)
    y_train_pred = np.asarray(model.predict_from_features(Phi[split.train_idx]), dtype=float)
    y_test_pred = np.asarray(model.y_pred_test_, dtype=float)
    if y_train_pred.ndim == 1:
        y_train_pred = y_train_pred.reshape(1, -1)
    if y_test_pred.ndim == 1:
        y_test_pred = y_test_pred.reshape(1, -1)
    best_params = model.best_params_ if isinstance(model.best_params_, list) else [model.best_params_]
    normalized_params = []
    for task_name, params in zip(dataset.task_names, best_params):
        params = dict(params or {})
        normalized_params.append(
            {
                "task": task_name,
                "selected_alpha": None,
                "selected_lambda": params.get("reg"),
                "selected_gamma": None,
                "selected_xi": params.get("xi"),
                "selected_nu": params.get("nu"),
                "selected_spectral_radius": None,
                "selected_input_scale": None,
                "selected_leak_rate": None,
                "val_score": params.get("score"),
            }
        )
    return {
        "method": method,
        "method_seed": int(method_seed),
        "task_names": list(dataset.task_names),
        "feature_dim": int(Phi.shape[1]),
        "raw_dim": int(dataset.raw_dim),
        "best_params": normalized_params,
        "train_mse": np.asarray(
            [mean_squared_error(dataset.y2d[i, split.train_idx], y_train_pred[i]) for i in range(dataset.y2d.shape[0])],
            dtype=float,
        ),
        "test_mse": np.asarray(
            [mean_squared_error(dataset.y2d[i, split.test_idx], y_test_pred[i]) for i in range(dataset.y2d.shape[0])],
            dtype=float,
        ),
        "y_train_true": dataset.y2d[:, split.train_idx],
        "y_train_pred": y_train_pred,
        "y_test_true": dataset.y2d[:, split.test_idx],
        "y_test_pred": y_test_pred,
    }


def compose_real_quark_config(
    *,
    dataset_path: str | Path,
    runner: str,
    experiment_name: str | None,
    N: int | None = None,
    w: int | None = None,
    d: int | None = None,
    seed: int = 0,
    backend: str = "auto",
    device: int | None = 0,
    model_overrides: Sequence[str] = (),
):
    config_dir = PROJECT_ROOT / "src/experiment/conf"
    dataset_path_text = str(Path(dataset_path)).replace("\\", "\\\\").replace("'", "\\'")
    overrides = [
        f"dataset_path='{dataset_path_text}'",
        f"model/qrc/runner={runner}",
        f"model.readout_backend={backend}",
        f"model.readout_device={0 if device is None else int(device)}",
        "instantiate_functionals=false",
        f"seed={int(seed)}",
    ]
    if N is not None:
        overrides.append(f"data.sampling.N={int(N)}")
    if w is not None:
        overrides.append(f"data.sampling.w={int(w)}")
    if d is not None:
        overrides.append(f"data.sampling.d={int(d)}")
    if experiment_name:
        overrides.append(f"experiment.name={experiment_name}")
    overrides.extend(str(override) for override in model_overrides)
    if str(backend) == "cupy":
        overrides.append("model.tuning.nu_grid=[0.5,1.5,2.5]")
        overrides.append("model.qrc.pubs.projection_backend=cupy")
        overrides.append(f"model.qrc.pubs.projection_device={0 if device is None else int(device)}")
        overrides.append("model.qrc.features.retriever.backend=cupy")
    else:
        overrides.append("model.qrc.pubs.projection_backend=numpy")
        overrides.append("model.qrc.features.retriever.backend=numpy")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="reg_sweep_experiment", overrides=overrides)


def normalize_method_suffix(method_suffix: str | None) -> str:
    if method_suffix is None or not str(method_suffix).strip():
        return ""
    suffix = str(method_suffix).strip()
    if not suffix.startswith("_"):
        suffix = "_" + suffix
    return suffix


def build_real_wide_rows(rows: Sequence[Mapping[str, str]], *, metric: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("metric") != metric:
            continue
        key = (row.get("dataset", ""), row.get("method", ""), row.get("method_seed", ""))
        out = grouped.setdefault(key, dict(row))
        out[f"{row.get('split')}_{metric}"] = row.get("value", "")
    return [
        {
            "dataset": row.get("dataset", ""),
            "method": row.get("method", ""),
            "method_seed": row.get("method_seed", ""),
            f"train_{metric}": row.get(f"train_{metric}", ""),
            f"test_{metric}": row.get(f"test_{metric}", ""),
            "n_train": row.get("n_train", ""),
            "n_test": row.get("n_test", ""),
            "feature_dim": row.get("feature_dim", ""),
            "raw_dim": row.get("raw_dim", ""),
            "artifact_dir": row.get("artifact_dir", ""),
        }
        for row in sorted(grouped.values(), key=lambda r: (r.get("dataset", ""), r.get("method", "")))
    ]


def build_real_markdown_table(wide_rows: Sequence[Mapping[str, Any]], *, metric: str) -> str:
    datasets = sorted({str(r["dataset"]) for r in wide_rows})
    methods = sorted({str(r["method"]) for r in wide_rows})
    lookup = {(str(r["method"]), str(r["dataset"])): format_metric(r.get(f"test_{metric}", "")) for r in wide_rows}
    lines = [f"| Method | " + " | ".join(datasets) + " |"]
    lines.append("|---|" + "|".join(["---:" for _ in datasets]) + "|")
    for method in methods:
        cells = [lookup.get((method, dataset), "") for dataset in datasets]
        lines.append("| " + method + " | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def format_metric(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(val):
        return ""
    return f"{val:.4g}"


def build_dataset_card_rows(
    data_root: str | Path = DEFAULT_REAL_DATA_ROOT,
    *,
    datasets: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    root = Path(data_root)
    if not root.exists():
        return []
    dataset_filter = {str(dataset) for dataset in datasets} if datasets else None
    rows: list[dict[str, Any]] = []
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if dataset_filter is not None and dataset_dir.name not in dataset_filter:
            continue
        try:
            dataset_path = resolve_real_dataset_path(dataset_dir)
        except (FileNotFoundError, ValueError):
            continue
        meta_path = dataset_path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        preprocess = dict(meta.get("preprocess") or {})
        w = int(meta.get("w") or 0)
        d = int(meta.get("d") or 0)
        rows.append(
            {
                "dataset": str(meta.get("dataset_id") or dataset_dir.name),
                "dataset_name": str(meta.get("dataset_name") or dataset_dir.name),
                "target_name": str(meta.get("target_name") or ""),
                "split_source": str(meta.get("split_source") or ""),
                "source": str(meta.get("source") or meta.get("download_url") or ""),
                "description": str(meta.get("description") or ""),
                "N": int(meta.get("N") or 0),
                "n_train": int(meta.get("n_train") or 0),
                "n_test": int(meta.get("n_test") or 0),
                "w": w,
                "d": d,
                "raw_dim": int(w * d),
                "missing_or_imputed": int(preprocess.get("nan_count_train", 0))
                + int(preprocess.get("nan_count_test", 0)),
                "artifact": str(dataset_path),
            }
        )
    return rows


def build_dataset_cards(
    out_root: str | Path,
    *,
    data_root: str | Path = DEFAULT_REAL_DATA_ROOT,
    datasets: Sequence[str] | None = None,
) -> str:
    _ = out_root
    rows = build_dataset_card_rows(data_root, datasets=datasets)
    cards = ["# Real-World Dataset Cards", ""]
    if not rows:
        cards.append("_No prepared datasets found._")
        cards.append("")
        return "\n".join(cards)
    cards.extend(
        [
            "| Dataset | N | Train | Test | w | d | raw_dim | Missing/Imputed | Target | Split | Source |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in rows:
        cards.append(
            "| {dataset} | {N} | {n_train} | {n_test} | {w} | {d} | {raw_dim} | "
            "{missing_or_imputed} | {target_name} | {split_source} | {source} |".format(**row)
        )
    cards.append("")
    return "\n".join(cards)


def build_classical_triage_rows(
    metric_rows: Sequence[Mapping[str, str]],
    *,
    dataset_card_rows: Sequence[Mapping[str, Any]] = (),
    metric: str = "nrmse_train_y_std",
    max_promoted: int = 3,
) -> list[dict[str, Any]]:
    card_by_dataset = {str(row["dataset"]): row for row in dataset_card_rows}
    values: dict[str, dict[str, float]] = {}
    for row in metric_rows:
        if row.get("split") != "test" or row.get("metric") != metric:
            continue
        method = str(row.get("method", ""))
        if method.startswith("quark_"):
            continue
        try:
            value = float(row.get("value", ""))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.setdefault(str(row.get("dataset", "")), {})[method] = value

    decisions: list[dict[str, Any]] = []
    for dataset, method_values in sorted(values.items()):
        if dataset_card_rows and dataset not in card_by_dataset:
            continue
        if not method_values:
            continue
        best_method, best_value = min(method_values.items(), key=lambda item: item[1])
        raw_ridge = method_values.get("raw_ridge", np.nan)
        raw_matern = method_values.get("raw_matern_krr", np.nan)
        matched = method_values.get("matched_random_features_matern_krr", np.nan)
        nonlinear = [
            value
            for method, value in method_values.items()
            if method in {"raw_matern_krr", "matched_random_features_matern_krr", "rff_ridge", "esn"}
        ]
        nonlinear_disagreement = float(max(nonlinear) - min(nonlinear)) if nonlinear else np.nan
        raw_ridge_weak = bool(np.isfinite(raw_ridge) and raw_ridge > 0.55)

        if best_value <= 0.30:
            decision = "reject_classical_crushed"
            candidate = False
        elif best_value <= 0.55:
            decision = "watchlist"
            candidate = False
        else:
            decision = "promote"
            candidate = True
        if best_value > 0.30 and raw_ridge_weak and np.isfinite(nonlinear_disagreement) and nonlinear_disagreement >= 0.20:
            decision = "promote_nonlinear_disagreement"
            candidate = True

        raw_dim = int(card_by_dataset.get(dataset, {}).get("raw_dim", 0) or 0)
        score = float(np.log1p(max(raw_dim, 0)) * best_value)
        decisions.append(
            {
                "dataset": dataset,
                "decision": decision,
                "selected_for_quark": False,
                "best_classical_method": best_method,
                "best_classical_test_nrmse": float(best_value),
                "raw_ridge_test_nrmse": raw_ridge,
                "raw_matern_test_nrmse": raw_matern,
                "matched_random_features_test_nrmse": matched,
                "nonlinear_disagreement": nonlinear_disagreement,
                "raw_dim": raw_dim,
                "difficulty_score": score,
                "candidate": candidate,
            }
        )

    promoted = [row for row in decisions if bool(row["candidate"])]
    promoted = sorted(promoted, key=lambda row: (float(row["difficulty_score"]), int(row["raw_dim"])), reverse=True)
    selected = {row["dataset"] for row in promoted[: int(max_promoted)]}
    for row in decisions:
        row["selected_for_quark"] = row["dataset"] in selected
        if row["candidate"] and not row["selected_for_quark"]:
            row["decision"] = str(row["decision"]) + "_deferred_top3_cap"
    return decisions


def build_quark_triage_rows(
    metric_rows: Sequence[Mapping[str, str]],
    classical_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str = "nrmse_train_y_std",
) -> list[dict[str, Any]]:
    classical_by_dataset = {str(row["dataset"]): row for row in classical_rows}
    quark_values: dict[str, dict[str, float]] = {}
    for row in metric_rows:
        if row.get("split") != "test" or row.get("metric") != metric:
            continue
        method = str(row.get("method", ""))
        if not method.startswith("quark_"):
            continue
        try:
            value = float(row.get("value", ""))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            quark_values.setdefault(str(row.get("dataset", "")), {})[method] = value

    rows: list[dict[str, Any]] = []
    for dataset, method_values in sorted(quark_values.items()):
        if classical_rows and dataset not in classical_by_dataset:
            continue
        if not method_values:
            continue
        best_quark_method, best_quark = min(method_values.items(), key=lambda item: item[1])
        classical = classical_by_dataset.get(dataset, {})
        best_classical = float(classical.get("best_classical_test_nrmse", np.nan))
        matched = float(classical.get("matched_random_features_test_nrmse", np.nan))
        raw_matern = float(classical.get("raw_matern_test_nrmse", np.nan))
        status = "needs_classical_context"
        if np.isfinite(best_classical):
            close_to_best = best_quark <= best_classical * 1.05 or best_quark <= best_classical + 0.05
            beats_matched = np.isfinite(matched) and best_quark <= matched - 0.05
            if close_to_best or beats_matched:
                status = "rebuttal_positive"
            elif np.isfinite(raw_matern) and best_quark > raw_matern * 1.20:
                has_n10 = any("_n10" in method for method in method_values)
                status = "drop_after_n10" if has_n10 else "diagnostic_only"
            else:
                status = "watchlist"
        rows.append(
            {
                "dataset": dataset,
                "status": status,
                "best_quark_method": best_quark_method,
                "best_quark_test_nrmse": best_quark,
                "best_classical_method": classical.get("best_classical_method", ""),
                "best_classical_test_nrmse": best_classical,
                "matched_random_features_test_nrmse": matched,
                "raw_matern_test_nrmse": raw_matern,
            }
        )
    return rows


def write_real_dataset_triage_outputs(
    out_root: str | Path = DEFAULT_REAL_TRIAGE_RESULTS_ROOT,
    *,
    data_root: str | Path = DEFAULT_REAL_DATA_ROOT,
    datasets: Sequence[str] | None = None,
    metric: str = "nrmse_train_y_std",
    max_promoted: int = 3,
) -> Path:
    out_root = Path(out_root)
    aggregate_dir = out_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = read_real_metric_rows(out_root, include_quark=True)
    dataset_rows = build_dataset_card_rows(data_root, datasets=datasets)
    classical_rows = build_classical_triage_rows(
        metric_rows,
        dataset_card_rows=dataset_rows,
        metric=metric,
        max_promoted=max_promoted,
    )
    quark_rows = build_quark_triage_rows(metric_rows, classical_rows, metric=metric)

    (aggregate_dir / "dataset_cards.md").write_text(
        build_dataset_cards(out_root, data_root=data_root, datasets=datasets),
        encoding="utf-8",
    )
    (aggregate_dir / "classical_triage_table.md").write_text(
        build_triage_markdown_table(classical_rows, title="Classical Triage Decisions"),
        encoding="utf-8",
    )
    (aggregate_dir / "quark_triage_table.md").write_text(
        build_triage_markdown_table(quark_rows, title="QuaRK Triage Results"),
        encoding="utf-8",
    )
    write_dict_csv(aggregate_dir / "triage_decisions.csv", classical_rows)
    write_dict_csv(aggregate_dir / "quark_triage.csv", quark_rows)
    return aggregate_dir


def build_triage_markdown_table(rows: Sequence[Mapping[str, Any]], *, title: str) -> str:
    if not rows:
        return f"# {title}\n\n_No rows available._\n"
    keys = list(rows[0].keys())
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("|" + "|".join("---" for _ in keys) + "|")
    for row in rows:
        lines.append("| " + " | ".join(format_triage_cell(row.get(key, "")) for key in keys) + " |")
    lines.append("")
    return "\n".join(lines)


def format_triage_cell(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return format_metric(value)
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    return str(value)


def update_real_world_response_drafts(
    *,
    response_dir: Path,
    aggregate_dir: Path,
    table_markdown: str,
    dataset_cards: str,
    metric: str,
) -> None:
    response_dir.mkdir(parents=True, exist_ok=True)
    global_text = f"""# Real-World Rebuttal Experiment Map

The real-world package evaluates scalar-label time-series extrinsic regression datasets where labels are external functionals of observed trajectories. The primary table reports test `{metric}`; raw MSE/RMSE are available in the long CSV.

Aggregate artifacts:

- Long CSV: `{aggregate_dir / 'real_world_long.csv'}`
- Wide CSV: `{aggregate_dir / 'real_world_wide.csv'}`
- Markdown table: `{aggregate_dir / 'real_world_table.md'}`
- Plot: `{aggregate_dir / 'real_world_plot.pdf'}`
- Dataset cards: `{aggregate_dir / 'dataset_cards.md'}`

{table_markdown}

{dataset_cards}
"""
    (response_dir / "real_world_global.md").write_text(global_text, encoding="utf-8")
    update_marker_block(
        response_dir / "global_view.md",
        marker="real_world_rebuttal",
        content=f"""## Real-world TSER rebuttal package

Primary metric: test `{metric}`.

Aggregate root: `{aggregate_dir}`

{table_markdown}
""",
    )
    for name, lead in {
        "MSDS_real_world.md": "We added real-world time-series extrinsic regression tasks to test whether the method is useful beyond the original synthetic benchmark.",
        "kc2J_real_world.md": "We added real-world comparative experiments with classical baselines on scalar-label time-series regression tasks.",
        "vKFL_real_world.md": "We added real-world trajectory-to-scalar regression datasets, while keeping the pipeline split-controlled and artifact-backed.",
    }.items():
        text = f"""# {name.removesuffix('.md')} Draft

{lead} These datasets are chosen because the target is an external scalar associated with the observed time-series trajectory, which is closer to the functional-learning setting of QuaRK than ordinary one-step forecasting.

Primary metric: test `{metric}`.

{table_markdown}

Artifacts are stored under `{aggregate_dir}`.
"""
        (response_dir / name).write_text(text, encoding="utf-8")


def update_marker_block(path: Path, *, marker: str, content: str) -> None:
    start = f"<!-- BEGIN {marker} -->"
    end = f"<!-- END {marker} -->"
    block = f"{start}\n{content.rstrip()}\n{end}\n"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if start in text and end in text:
            before = text.split(start, 1)[0]
            after = text.split(end, 1)[1]
            path.write_text(before + block + after.lstrip("\n"), encoding="utf-8")
            return
        sep = "" if text.endswith("\n") else "\n"
        path.write_text(text + sep + "\n" + block, encoding="utf-8")
        return
    path.write_text("# Global Rebuttal Experiment Map\n\n" + block, encoding="utf-8")


def real_dataset_dirs(data_root: str | Path = DEFAULT_REAL_DATA_ROOT, datasets: Sequence[str] | None = None) -> list[Path]:
    root = Path(data_root)
    if datasets:
        return [root / dataset for dataset in datasets]
    return sorted(path for path in root.iterdir() if path.is_dir())
