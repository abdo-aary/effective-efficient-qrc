"""Classical-only rebuttal baselines for the QuaRK KDD response.

This module intentionally does not import or call the QRC featurizer.  The only
QuaRK artifact it reads is the cached model output saved in ``arrays.npz`` so the
reference row can be evaluated on the same train/test split without rerunning
quantum circuits.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import eigs
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import Matern as SkMatern
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.compute.backend import asnumpy, import_cupy, resolve_backend
from src.data.factory import load_windows_dataset
from src.experiment.exp_utils import _as_LN
from src.models.gpu_krr import GPU_MATERN_NU_GRID, krr_lambda_sweep as gpu_krr_lambda_sweep
from src.models.gpu_krr import krr_validation_mse as gpu_krr_validation_mse
from src.models.gpu_krr import matern_kernel_matrix, ridge_fit_predict, solve_krr as gpu_solve_krr
from src.settings import PROJECT_ROOT_PATH


PROJECT_ROOT = Path(PROJECT_ROOT_PATH)
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "storage/data/synthetic/varma/N=10000__w=25__d=3__s=100"
)
DEFAULT_REFERENCE_MODEL_DIR = (
    PROJECT_ROOT
    / "storage/results/reg_sweep/c-shadows/full/N=10000__w=25__d=3__s=100"
    / "2026-02-02_14-02-13/model"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "storage/results/rebuttal/classical_only"
    / "varma_e2_three__N=10000__w=25__d=3__s=100"
    / "split=paper_cshadows_2026-02-02_14-02-13"
)

DATASET_ID = "varma_e2_three__N=10000__w=25__d=3__s=100"
SPLIT_SOURCE = "paper_cshadows_2026-02-02_14-02-13"
TASK_NAMES_E2_THREE = ("one_step_forecast", "exp_fading_linear", "volterra")

REG_GRID = np.asarray(
    [
        1e-12,
        1e-11,
        1e-10,
        1e-9,
        1e-8,
        1e-7,
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        5e-3,
        1e-2,
        5e-2,
        1e-1,
        5e-1,
        1e0,
        5e0,
        1e1,
        5e1,
        1e2,
        5e2,
        1e3,
        5e3,
        1e4,
        5e4,
        1e5,
    ],
    dtype=float,
)
RIDGE_ALPHA_GRID = np.asarray([1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4], dtype=float)
RFF_GAMMA_GRID = np.asarray([1e-3, 1e-2, 1e-1, 1e0, 1e1], dtype=float)
MATERN_NU_GRID = np.asarray([0.5, 1.5, 2.5, 5.0], dtype=float)
MATERN_XI_BOUNDS = (1e-3, 1e3)
MATERN_XI_MAXITER = 80
MATERN_TUNING_REG = 1e-6
MATERN_TUNE_MAX_TRAIN = 1024
MATERN_TUNE_MAX_VAL = 512

ESN_SPECTRAL_RADIUS_GRID = np.asarray([0.5, 0.9], dtype=float)
ESN_INPUT_SCALE_GRID = np.asarray([0.2, 1.0], dtype=float)
ESN_LEAK_RATE_GRID = np.asarray([0.5, 1.0], dtype=float)
ESN_DENSITY = 0.05

METRICS_COLUMNS = [
    "experiment_id",
    "method",
    "dataset",
    "task",
    "split",
    "metric",
    "value",
    "n_train",
    "n_test",
    "feature_dim",
    "raw_dim",
    "method_seed",
    "split_source",
    "val_seed",
    "selected_alpha",
    "selected_lambda",
    "selected_gamma",
    "selected_xi",
    "selected_nu",
    "selected_spectral_radius",
    "selected_input_scale",
    "selected_leak_rate",
    "artifact_dir",
]

CLASSICAL_METHODS = {
    "raw_matern_krr",
    "raw_ridge",
    "matched_random_features_matern_krr",
    "rff_ridge",
    "esn",
}
FEATURE_DIM_METHODS = {"matched_random_features_matern_krr", "rff_ridge", "esn"}


@dataclass(frozen=True)
class BenchmarkData:
    X: np.ndarray
    y2d: np.ndarray
    task_names: list[str]
    dataset_id: str
    raw_dim: int
    artifact_root: Path
    meta: dict[str, Any]


@dataclass(frozen=True)
class SplitData:
    train_idx: np.ndarray
    test_idx: np.ndarray
    split_source: str


def run_classical_baseline(
    method: str,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    reference_model_dir: str | Path = DEFAULT_REFERENCE_MODEL_DIR,
    out_dir: str | Path = DEFAULT_OUTPUT_ROOT,
    method_seed: int = 0,
    feature_dim: int = 315,
    backend: str = "auto",
    device: int | None = 0,
) -> Path:
    """Run one classical-only rebuttal baseline and write standard artifacts.

    Parameters match the rebuttal plan.  ``reference_model_dir`` is used only to
    load the paper train/test split.
    """

    method = str(method)
    if method not in CLASSICAL_METHODS:
        raise ValueError(f"Unknown classical baseline method={method!r}.")

    t0 = time.perf_counter()
    dataset = load_benchmark_data(dataset_path)
    split = load_reference_split(reference_model_dir)
    run_dir = method_run_dir(out_dir, method, method_seed=method_seed, feature_dim=feature_dim)
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_std, raw_scaler = standardize_flattened_windows(dataset.X, split.train_idx)

    if method == "raw_matern_krr":
        result = fit_matern_krr_features(
            raw_std,
            dataset.y2d,
            split,
            dataset.task_names,
            method="raw_matern_krr",
            method_seed=method_seed,
            backend=backend,
            device=device,
        )
        final_feature_dim = dataset.raw_dim
    elif method == "raw_ridge":
        result = fit_ridge_features(
            raw_std,
            dataset.y2d,
            split,
            dataset.task_names,
            method="raw_ridge",
            method_seed=method_seed,
            backend=backend,
            device=device,
        )
        final_feature_dim = dataset.raw_dim
    elif method == "matched_random_features_matern_krr":
        features, feature_meta = make_matched_random_features(
            raw_std, raw_dim=dataset.raw_dim, feature_dim=feature_dim, seed=method_seed
        )
        features, _ = standardize_features(features, split.train_idx)
        result = fit_matern_krr_features(
            features,
            dataset.y2d,
            split,
            dataset.task_names,
            method="matched_random_features_matern_krr",
            method_seed=method_seed,
            extra_summary=feature_meta,
            backend=backend,
            device=device,
        )
        final_feature_dim = feature_dim
    elif method == "rff_ridge":
        result = fit_rff_ridge(
            raw_std,
            dataset.y2d,
            split,
            dataset.task_names,
            method_seed=method_seed,
            feature_dim=feature_dim,
            backend=backend,
            device=device,
        )
        final_feature_dim = feature_dim
    elif method == "esn":
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
    else:  # pragma: no cover - guarded above
        raise AssertionError(method)

    rows = make_metric_rows(
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
            "reference_model_dir": str(Path(reference_model_dir)),
            "out_dir": str(Path(out_dir)),
            "method_seed": int(method_seed),
            "feature_dim": int(feature_dim),
            "backend": str(backend),
            "device": None if device is None else int(device),
            "raw_dim": int(dataset.raw_dim),
            "n_train": int(split.train_idx.size),
            "n_test": int(split.test_idx.size),
            "split_source": split.split_source,
            "raw_scaler_mean_shape": list(raw_scaler.mean_.shape),
            "ridge_alpha_grid": RIDGE_ALPHA_GRID.tolist(),
            "reg_grid": REG_GRID.tolist(),
            "matern_tune_max_train": int(MATERN_TUNE_MAX_TRAIN),
            "matern_tune_max_val": int(MATERN_TUNE_MAX_VAL),
        },
        summary_extra={
            "duration_seconds": float(time.perf_counter() - t0),
            "dataset": dataset.dataset_id,
            "task_names": dataset.task_names,
        },
    )

    return run_dir


def run_quark_reference(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    reference_model_dir: str | Path = DEFAULT_REFERENCE_MODEL_DIR,
    out_dir: str | Path = DEFAULT_OUTPUT_ROOT,
    method_seed: int = 0,
) -> Path:
    """Evaluate the cached QuaRK feature map on the paper split.

    This is a read-only reference run: it loads ``Phi_full`` and the saved split,
    selects only the KRR regularization on an inner validation split, then refits
    dual weights on the full outer train split.
    """

    t0 = time.perf_counter()
    dataset = load_benchmark_data(dataset_path)
    split = load_reference_split(reference_model_dir)
    ref = load_reference_arrays(reference_model_dir)
    meta = load_reference_meta(reference_model_dir)
    Phi = np.asarray(ref["Phi_full"], dtype=float)
    scaler_mean = np.asarray(ref["scaler_mean"], dtype=float)
    scaler_scale = np.asarray(ref["scaler_scale"], dtype=float)
    scaler_scale = np.where(scaler_scale == 0.0, 1.0, scaler_scale)
    Phi = (Phi - scaler_mean) / scaler_scale

    best_params = meta.get("best_params_", [])
    if len(best_params) < dataset.y2d.shape[0]:
        raise ValueError("Reference meta.json does not contain one best_params_ entry per task.")

    result = fit_cached_quark_krr(
        Phi=Phi,
        y2d=dataset.y2d,
        split=split,
        task_names=dataset.task_names,
        best_params=best_params,
        method_seed=method_seed,
    )

    run_dir = method_run_dir(out_dir, "quark_reference", method_seed=method_seed, feature_dim=Phi.shape[1])
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = make_metric_rows(
        result=result,
        dataset=dataset,
        split=split,
        method="quark_reference",
        method_seed=method_seed,
        feature_dim=int(Phi.shape[1]),
        artifact_dir=run_dir,
    )
    write_run_artifacts(
        run_dir=run_dir,
        rows=rows,
        result=result,
        run_config={
            "method": "quark_reference",
            "dataset_path": str(Path(dataset_path)),
            "reference_model_dir": str(Path(reference_model_dir)),
            "out_dir": str(Path(out_dir)),
            "method_seed": int(method_seed),
            "feature_dim": int(Phi.shape[1]),
            "raw_dim": int(dataset.raw_dim),
            "n_train": int(split.train_idx.size),
            "n_test": int(split.test_idx.size),
            "split_source": split.split_source,
            "reg_grid": REG_GRID.tolist(),
            "matern_tune_max_train": int(MATERN_TUNE_MAX_TRAIN),
            "matern_tune_max_val": int(MATERN_TUNE_MAX_VAL),
            "note": "Uses cached Phi_full from arrays.npz; no quantum circuit execution.",
        },
        summary_extra={
            "duration_seconds": float(time.perf_counter() - t0),
            "dataset": dataset.dataset_id,
            "task_names": dataset.task_names,
        },
    )
    return run_dir


def aggregate_classical_baselines(
    out_dir: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    reference_model_dir: str | Path = DEFAULT_REFERENCE_MODEL_DIR,
    include_quark_reference: bool = True,
) -> Path:
    """Aggregate classical-only rebuttal metrics into CSV, Markdown, and a PDF plot."""

    out_dir = Path(out_dir)
    if include_quark_reference:
        quark_dir = method_run_dir(out_dir, "quark_reference", method_seed=0, feature_dim=315)
        if not (quark_dir / "metrics.csv").exists():
            run_quark_reference(dataset_path, reference_model_dir, out_dir)

    aggregate_dir = out_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for metrics_path in sorted(out_dir.glob("**/metrics.csv")):
        if "aggregate" in metrics_path.parts:
            continue
        with metrics_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({col: row.get(col, "") for col in METRICS_COLUMNS})

    if not rows:
        raise FileNotFoundError(f"No metrics.csv files found under {out_dir}")

    long_path = aggregate_dir / "classical_baselines_long.csv"
    write_metrics_csv(long_path, rows)

    wide_rows = build_wide_rows(rows)
    wide_path = aggregate_dir / "classical_baselines_wide.csv"
    write_dict_csv(wide_path, wide_rows)

    table_path = aggregate_dir / "classical_baselines_table.md"
    table_path.write_text(build_markdown_table(wide_rows), encoding="utf-8")

    plot_path = aggregate_dir / "classical_baselines_plot.pdf"
    write_baseline_plot(wide_rows, plot_path)

    update_rebuttal_response_drafts(
        aggregate_dir=aggregate_dir,
        table_markdown=table_path.read_text(encoding="utf-8"),
    )
    return aggregate_dir


def load_benchmark_data(dataset_path: str | Path) -> BenchmarkData:
    ds, artifact = load_windows_dataset(dataset_path, instantiate_functionals=False)
    X = np.asarray(ds.X, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"Expected window tensor with shape (N,w,d), got {X.shape}.")
    N, w, d = X.shape
    y2d = _as_LN(np.asarray(ds.y, dtype=float), N)
    task_names = canonical_task_names(y2d.shape[0], ds.meta, ds.label_functionals)
    return BenchmarkData(
        X=X,
        y2d=y2d,
        task_names=task_names,
        dataset_id=dataset_id_from_meta(ds.meta),
        raw_dim=int(w * d),
        artifact_root=artifact.root,
        meta=dict(ds.meta),
    )


def load_reference_split(reference_model_dir: str | Path) -> SplitData:
    arrays = load_reference_arrays(reference_model_dir)
    train_idx = np.asarray(arrays["train_idx"], dtype=int).reshape(-1)
    test_idx = np.asarray(arrays["test_idx"], dtype=int).reshape(-1)
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("Reference train/test split must be non-empty.")
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError("Reference train_idx and test_idx overlap.")
    return SplitData(
        train_idx=train_idx,
        test_idx=test_idx,
        split_source=split_source_from_path(reference_model_dir),
    )


def load_reference_arrays(reference_model_dir: str | Path) -> dict[str, np.ndarray]:
    path = Path(reference_model_dir) / "arrays.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing reference arrays: {path}")
    with np.load(path) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def load_reference_meta(reference_model_dir: str | Path) -> dict[str, Any]:
    path = Path(reference_model_dir) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing reference meta.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_task_names(
    n_tasks: int,
    meta: Mapping[str, Any] | None = None,
    label_functionals: Sequence[Any] | None = None,
) -> list[str]:
    if label_functionals:
        names = [type(fn).__name__.replace("Functional", "") for fn in label_functionals]
        if len(names) == n_tasks:
            return [camel_to_snake(name).replace("voltera", "volterra") for name in names]
    meta = dict(meta or {})
    if meta.get("functionals_kind") == "e2_three" and n_tasks == 3:
        return list(TASK_NAMES_E2_THREE)
    return [f"task_{i}" for i in range(n_tasks)]


def dataset_id_from_meta(meta: Mapping[str, Any] | None) -> str:
    meta = dict(meta or {})
    if {
        "process_kind",
        "functionals_kind",
        "N",
        "w",
        "d",
        "s",
    }.issubset(meta):
        return (
            f"{meta['process_kind']}_{meta['functionals_kind']}__"
            f"N={meta['N']}__w={meta['w']}__d={meta['d']}__s={meta['s']}"
        )
    return DATASET_ID


def split_source_from_path(reference_model_dir: str | Path) -> str:
    path = Path(reference_model_dir)
    if path == DEFAULT_REFERENCE_MODEL_DIR:
        return SPLIT_SOURCE
    # For paths shaped ".../c-shadows/.../<timestamp>/model", keep a compact trace.
    parts = path.parts
    if len(parts) >= 2 and path.name == "model":
        return f"reference_{path.parent.name}"
    return str(path)


def method_run_dir(
    out_dir: str | Path,
    method: str,
    *,
    method_seed: int,
    feature_dim: int,
) -> Path:
    root = Path(out_dir)
    if method in FEATURE_DIM_METHODS:
        return root / method / f"feature_dim={int(feature_dim)}" / f"seed={int(method_seed)}"
    return root / method / f"seed={int(method_seed)}"


def standardize_flattened_windows(X: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    flat = np.asarray(X, dtype=float).reshape(X.shape[0], -1)
    return standardize_features(flat, train_idx)


def standardize_features(X: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    scaler.fit(np.asarray(X)[np.asarray(train_idx, dtype=int)])
    return scaler.transform(X), scaler


def make_inner_split(
    train_idx: np.ndarray,
    *,
    val_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    train_idx = np.asarray(train_idx, dtype=int).reshape(-1)
    if train_idx.size < 3:
        raise ValueError("Need at least 3 outer-train samples for an inner validation split.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_idx.size)
    n_val = max(1, int(round(val_ratio * train_idx.size)))
    if n_val >= train_idx.size:
        n_val = train_idx.size - 1
    val_local = perm[:n_val]
    tr_local = perm[n_val:]
    return train_idx[tr_local], train_idx[val_local]


def cap_train_val_indices(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    seed: int,
    max_train: int,
    max_val: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically cap expensive kernel-tuning splits.

    The final fit still uses the full outer train split; this cap only keeps
    Matérn hyperparameter and lambda selection feasible for rebuttal timing.
    """

    rng = np.random.default_rng(seed)
    train_idx = np.asarray(train_idx, dtype=int).reshape(-1)
    val_idx = np.asarray(val_idx, dtype=int).reshape(-1)
    if train_idx.size > int(max_train):
        train_idx = train_idx[rng.choice(train_idx.size, size=int(max_train), replace=False)]
    if val_idx.size > int(max_val):
        val_idx = val_idx[rng.choice(val_idx.size, size=int(max_val), replace=False)]
    return np.sort(train_idx), np.sort(val_idx)


def fit_ridge_features(
    features: np.ndarray,
    y2d: np.ndarray,
    split: SplitData,
    task_names: Sequence[str],
    *,
    method: str,
    method_seed: int,
    backend: str = "numpy",
    device: int | None = None,
) -> dict[str, Any]:
    inner_train_idx, val_idx = make_inner_split(split.train_idx, seed=method_seed)
    y_train_pred = np.empty((y2d.shape[0], split.train_idx.size), dtype=float)
    y_test_pred = np.empty((y2d.shape[0], split.test_idx.size), dtype=float)
    best_params: list[dict[str, Any]] = []

    for task_i, task_name in enumerate(task_names):
        best_alpha, best_val = tune_ridge_alpha(
            features,
            y2d[task_i],
            inner_train_idx,
            val_idx,
            alphas=RIDGE_ALPHA_GRID,
            backend=backend,
            device=device,
        )
        if resolve_backend(backend) == "cupy":
            y_train_pred[task_i] = ridge_fit_predict(
                features[split.train_idx],
                y2d[task_i, split.train_idx],
                features[split.train_idx],
                alpha=best_alpha,
                backend=backend,
                device=device,
            )
            y_test_pred[task_i] = ridge_fit_predict(
                features[split.train_idx],
                y2d[task_i, split.train_idx],
                features[split.test_idx],
                alpha=best_alpha,
                backend=backend,
                device=device,
            )
        else:
            model = Ridge(alpha=best_alpha, fit_intercept=True)
            model.fit(features[split.train_idx], y2d[task_i, split.train_idx])
            y_train_pred[task_i] = model.predict(features[split.train_idx])
            y_test_pred[task_i] = model.predict(features[split.test_idx])
        best_params.append(
            {
                "task": task_name,
                "selected_alpha": float(best_alpha),
                "selected_lambda": None,
                "selected_gamma": None,
                "selected_xi": None,
                "selected_nu": None,
                "selected_spectral_radius": None,
                "selected_input_scale": None,
                "selected_leak_rate": None,
                "val_mse": float(best_val),
            }
        )

    return make_result(
        method=method,
        method_seed=method_seed,
        task_names=task_names,
        y2d=y2d,
        split=split,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
        best_params=best_params,
        feature_dim=int(features.shape[1]),
        raw_dim=None,
    )


def tune_ridge_alpha(
    features: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    alphas: Sequence[float],
    backend: str = "numpy",
    device: int | None = None,
) -> tuple[float, float]:
    best_alpha = float("nan")
    best_mse = float("inf")
    for alpha in alphas:
        if resolve_backend(backend) == "cupy":
            pred = ridge_fit_predict(
                features[train_idx],
                y[train_idx],
                features[val_idx],
                alpha=float(alpha),
                backend=backend,
                device=device,
            )
        else:
            model = Ridge(alpha=float(alpha), fit_intercept=True)
            model.fit(features[train_idx], y[train_idx])
            pred = model.predict(features[val_idx])
        mse = mean_squared_error(y[val_idx], pred)
        if mse < best_mse:
            best_mse = mse
            best_alpha = float(alpha)
    return best_alpha, best_mse


def make_matched_random_features(
    raw_features: np.ndarray,
    *,
    raw_dim: int,
    feature_dim: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    W = rng.normal(loc=0.0, scale=1.0 / math.sqrt(raw_dim), size=(raw_features.shape[1], feature_dim))
    b = rng.uniform(-math.pi, math.pi, size=feature_dim)
    Phi = np.tanh(raw_features @ W + b)
    return Phi, {"random_feature_map": "tanh(XW+b)", "weight_std": 1.0 / math.sqrt(raw_dim)}


def fit_rff_ridge(
    raw_features: np.ndarray,
    y2d: np.ndarray,
    split: SplitData,
    task_names: Sequence[str],
    *,
    method_seed: int,
    feature_dim: int,
    backend: str = "numpy",
    device: int | None = None,
) -> dict[str, Any]:
    inner_train_idx, val_idx = make_inner_split(split.train_idx, seed=method_seed)
    y_train_pred = np.empty((y2d.shape[0], split.train_idx.size), dtype=float)
    y_test_pred = np.empty((y2d.shape[0], split.test_idx.size), dtype=float)
    best_params: list[dict[str, Any]] = []

    for task_i, task_name in enumerate(task_names):
        best: dict[str, Any] | None = None
        for gamma in RFF_GAMMA_GRID:
            features = make_rff_features(
                raw_features,
                gamma=float(gamma),
                feature_dim=feature_dim,
                seed=method_seed,
                backend=backend,
                device=device,
            )
            features, _ = standardize_features(features, split.train_idx)
            alpha, val_mse = tune_ridge_alpha(
                features,
                y2d[task_i],
                inner_train_idx,
                val_idx,
                alphas=RIDGE_ALPHA_GRID,
                backend=backend,
                device=device,
            )
            if best is None or val_mse < best["val_mse"]:
                best = {"gamma": float(gamma), "alpha": float(alpha), "val_mse": float(val_mse)}

        assert best is not None
        features = make_rff_features(
            raw_features,
            gamma=best["gamma"],
            feature_dim=feature_dim,
            seed=method_seed,
            backend=backend,
            device=device,
        )
        features, _ = standardize_features(features, split.train_idx)
        if resolve_backend(backend) == "cupy":
            y_train_pred[task_i] = ridge_fit_predict(
                features[split.train_idx],
                y2d[task_i, split.train_idx],
                features[split.train_idx],
                alpha=float(best["alpha"]),
                backend=backend,
                device=device,
            )
            y_test_pred[task_i] = ridge_fit_predict(
                features[split.train_idx],
                y2d[task_i, split.train_idx],
                features[split.test_idx],
                alpha=float(best["alpha"]),
                backend=backend,
                device=device,
            )
        else:
            model = Ridge(alpha=float(best["alpha"]), fit_intercept=True)
            model.fit(features[split.train_idx], y2d[task_i, split.train_idx])
            y_train_pred[task_i] = model.predict(features[split.train_idx])
            y_test_pred[task_i] = model.predict(features[split.test_idx])
        best_params.append(
            {
                "task": task_name,
                "selected_alpha": float(best["alpha"]),
                "selected_lambda": None,
                "selected_gamma": float(best["gamma"]),
                "selected_xi": None,
                "selected_nu": None,
                "selected_spectral_radius": None,
                "selected_input_scale": None,
                "selected_leak_rate": None,
                "val_mse": float(best["val_mse"]),
            }
        )

    return make_result(
        method="rff_ridge",
        method_seed=method_seed,
        task_names=task_names,
        y2d=y2d,
        split=split,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
        best_params=best_params,
        feature_dim=int(feature_dim),
        raw_dim=None,
    )


def make_rff_features(
    raw_features: np.ndarray,
    *,
    gamma: float,
    feature_dim: int,
    seed: int,
    backend: str = "numpy",
    device: int | None = None,
) -> np.ndarray:
    if resolve_backend(backend) == "cupy":
        cp = import_cupy()
        rng = np.random.default_rng(seed)
        W = rng.normal(
            loc=0.0,
            scale=np.sqrt(2.0 * float(gamma)),
            size=(raw_features.shape[1], int(feature_dim)),
        )
        b = rng.uniform(0.0, 2.0 * math.pi, size=int(feature_dim))
        Xg = cp.asarray(raw_features)
        Wg = cp.asarray(W)
        bg = cp.asarray(b)
        return asnumpy(np.sqrt(2.0 / int(feature_dim)) * cp.cos(Xg @ Wg + bg))
    sampler = RBFSampler(gamma=float(gamma), n_components=int(feature_dim), random_state=int(seed))
    return sampler.fit_transform(raw_features)


def fit_matern_krr_features(
    features: np.ndarray,
    y2d: np.ndarray,
    split: SplitData,
    task_names: Sequence[str],
    *,
    method: str,
    method_seed: int,
    extra_summary: Mapping[str, Any] | None = None,
    backend: str = "numpy",
    device: int | None = None,
) -> dict[str, Any]:
    inner_train_idx, val_idx = make_inner_split(split.train_idx, seed=method_seed)
    y_train_pred = np.empty((y2d.shape[0], split.train_idx.size), dtype=float)
    y_test_pred = np.empty((y2d.shape[0], split.test_idx.size), dtype=float)
    best_params: list[dict[str, Any]] = []

    for task_i, task_name in enumerate(task_names):
        tune_train_idx, tune_val_idx = cap_train_val_indices(
            inner_train_idx,
            val_idx,
            seed=method_seed + task_i,
            max_train=MATERN_TUNE_MAX_TRAIN,
            max_val=MATERN_TUNE_MAX_VAL,
        )
        effective_nu_grid = matern_nu_grid_for_backend(backend, MATERN_NU_GRID)
        xi, nu, matern_val = tune_matern_xi_nu_on_split(
            features,
            y2d[task_i],
            tune_train_idx,
            tune_val_idx,
            nu_grid=effective_nu_grid,
            xi_bounds=MATERN_XI_BOUNDS,
            xi_maxiter=MATERN_XI_MAXITER,
            reg=MATERN_TUNING_REG,
            backend=backend,
            device=device,
        )
        lam, lam_val = tune_krr_lambda_on_split(
            features,
            y2d[task_i],
            tune_train_idx,
            tune_val_idx,
            xi=xi,
            nu=nu,
            reg_grid=REG_GRID,
            backend=backend,
            device=device,
        )
        if can_use_gpu_matern(backend, nu):
            Ktt_gpu = matern_kernel_matrix(
                features[split.train_idx],
                features[split.train_idx],
                xi=xi,
                nu=nu,
                backend=backend,
                device=device,
            )
            alpha = asnumpy(gpu_solve_krr(Ktt_gpu, y2d[task_i, split.train_idx], reg=float(lam), backend=backend))
            y_train_pred[task_i] = asnumpy(Ktt_gpu @ import_cupy().asarray(alpha))
            Ktest_gpu = matern_kernel_matrix(
                features[split.test_idx],
                features[split.train_idx],
                xi=xi,
                nu=nu,
                backend=backend,
                device=device,
            )
            y_test_pred[task_i] = asnumpy(Ktest_gpu @ import_cupy().asarray(alpha))
        else:
            ker = build_matern_kernel(xi=xi, nu=nu)
            Ktt = ker(features[split.train_idx], features[split.train_idx])
            alpha = solve_linear_system(Ktt + float(lam) * np.eye(Ktt.shape[0]), y2d[task_i, split.train_idx])
            y_train_pred[task_i] = Ktt @ alpha
            Ktest = ker(features[split.test_idx], features[split.train_idx])
            y_test_pred[task_i] = Ktest @ alpha
        best_params.append(
            {
                "task": task_name,
                "selected_alpha": None,
                "selected_lambda": float(lam),
                "selected_gamma": None,
                "selected_xi": float(xi),
                "selected_nu": float(nu),
                "selected_spectral_radius": None,
                "selected_input_scale": None,
                "selected_leak_rate": None,
                "matern_val_mse": float(matern_val),
                "lambda_val_mse": float(lam_val),
                "tuning_train_size": int(tune_train_idx.size),
                "tuning_val_size": int(tune_val_idx.size),
            }
        )

    result = make_result(
        method=method,
        method_seed=method_seed,
        task_names=task_names,
        y2d=y2d,
        split=split,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
        best_params=best_params,
        feature_dim=int(features.shape[1]),
        raw_dim=None,
    )
    if extra_summary:
        result["extra_summary"] = dict(extra_summary)
    return result


def can_use_gpu_matern(backend: str, nu: float) -> bool:
    if resolve_backend(backend) != "cupy":
        return False
    if any(np.isclose(float(nu), allowed) for allowed in GPU_MATERN_NU_GRID):
        return True
    if str(backend) == "auto":
        return False
    raise ValueError(f"GPU Matérn supports nu in {GPU_MATERN_NU_GRID}, got {nu}.")


def matern_nu_grid_for_backend(backend: str, nu_grid: Sequence[float]) -> np.ndarray:
    """Return a Matérn smoothness grid compatible with the selected backend."""

    nu_arr = np.asarray(nu_grid, dtype=float).reshape(-1)
    if resolve_backend(backend) != "cupy":
        return nu_arr
    allowed = np.asarray(GPU_MATERN_NU_GRID, dtype=float)
    keep = np.asarray([np.any(np.isclose(nu, allowed)) for nu in nu_arr], dtype=bool)
    filtered = nu_arr[keep]
    if filtered.size == 0:
        raise ValueError(
            f"GPU Matérn supports nu in {GPU_MATERN_NU_GRID}, but nu_grid={nu_arr.tolist()} has no overlap."
        )
    return filtered


def tune_matern_xi_nu_on_split(
    features: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    nu_grid: Sequence[float],
    xi_bounds: tuple[float, float],
    xi_maxiter: int,
    reg: float,
    backend: str = "numpy",
    device: int | None = None,
) -> tuple[float, float, float]:
    if xi_bounds[0] <= 0 or xi_bounds[1] <= 0 or xi_bounds[0] >= xi_bounds[1]:
        raise ValueError(f"Invalid xi_bounds={xi_bounds!r}")
    log_lo, log_hi = float(np.log(xi_bounds[0])), float(np.log(xi_bounds[1]))
    best_xi = float("nan")
    best_nu = float("nan")
    best_mse = float("inf")
    for nu in nu_grid:
        nu = float(nu)
        res = minimize_scalar(
            lambda log_xi: (
                gpu_krr_validation_mse(
                    features[train_idx],
                    y[train_idx],
                    features[val_idx],
                    y[val_idx],
                    xi=float(np.exp(log_xi)),
                    nu=nu,
                    reg=reg,
                    backend=backend,
                    device=device,
                )
                if can_use_gpu_matern(backend, nu)
                else krr_validation_mse(
                    features,
                    y,
                    train_idx,
                    val_idx,
                    xi=float(np.exp(log_xi)),
                    nu=nu,
                    reg=reg,
                )
            ),
            bounds=(log_lo, log_hi),
            method="bounded",
            options={"maxiter": int(xi_maxiter)},
        )
        if float(res.fun) < best_mse:
            best_mse = float(res.fun)
            best_xi = float(np.exp(res.x))
            best_nu = nu
    return best_xi, best_nu, best_mse


def tune_krr_lambda_on_split(
    features: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    xi: float,
    nu: float,
    reg_grid: Sequence[float],
    backend: str = "numpy",
    device: int | None = None,
) -> tuple[float, float]:
    if can_use_gpu_matern(backend, nu):
        Ktt = matern_kernel_matrix(features[train_idx], features[train_idx], xi=xi, nu=nu, backend=backend, device=device)
        Kvt = matern_kernel_matrix(features[val_idx], features[train_idx], xi=xi, nu=nu, backend=backend, device=device)
        sweep = gpu_krr_lambda_sweep(
            Ktt,
            Kvt,
            y[train_idx],
            y[val_idx],
            reg_grid,
            backend=backend,
        )
        best_idx = int(np.argmin(sweep["eval_mse"]))
        return float(np.asarray(reg_grid, dtype=float)[best_idx]), float(sweep["eval_mse"][best_idx])
    ker = build_matern_kernel(xi=xi, nu=nu)
    Ktt = ker(features[train_idx], features[train_idx])
    Kvt = ker(features[val_idx], features[train_idx])
    I = np.eye(Ktt.shape[0])
    best_lam = float("nan")
    best_mse = float("inf")
    for lam in reg_grid:
        alpha = solve_linear_system(Ktt + float(lam) * I, y[train_idx])
        pred = Kvt @ alpha
        mse = mean_squared_error(y[val_idx], pred)
        if mse < best_mse:
            best_mse = mse
            best_lam = float(lam)
    return best_lam, best_mse


def krr_validation_mse(
    features: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    xi: float,
    nu: float,
    reg: float,
) -> float:
    ker = build_matern_kernel(xi=xi, nu=nu)
    Ktt = ker(features[train_idx], features[train_idx])
    Kvt = ker(features[val_idx], features[train_idx])
    alpha = solve_linear_system(Ktt + float(reg) * np.eye(Ktt.shape[0]), y[train_idx])
    pred = Kvt @ alpha
    return mean_squared_error(y[val_idx], pred)


def build_matern_kernel(*, xi: float, nu: float):
    return ConstantKernel(1.0, constant_value_bounds="fixed") * SkMatern(
        length_scale=float(xi),
        length_scale_bounds="fixed",
        nu=float(nu),
    )


def fit_cached_quark_krr(
    *,
    Phi: np.ndarray,
    y2d: np.ndarray,
    split: SplitData,
    task_names: Sequence[str],
    best_params: Sequence[Mapping[str, Any]],
    method_seed: int,
) -> dict[str, Any]:
    inner_train_idx, val_idx = make_inner_split(split.train_idx, seed=method_seed)
    y_train_pred = np.empty((y2d.shape[0], split.train_idx.size), dtype=float)
    y_test_pred = np.empty((y2d.shape[0], split.test_idx.size), dtype=float)
    out_params: list[dict[str, Any]] = []
    for task_i, task_name in enumerate(task_names):
        task_params = dict(best_params[task_i])
        xi = float(task_params["xi"])
        nu = float(task_params["nu"])
        tune_train_idx, tune_val_idx = cap_train_val_indices(
            inner_train_idx,
            val_idx,
            seed=method_seed + task_i,
            max_train=MATERN_TUNE_MAX_TRAIN,
            max_val=MATERN_TUNE_MAX_VAL,
        )
        lam, lam_val = tune_krr_lambda_on_split(
            Phi,
            y2d[task_i],
            tune_train_idx,
            tune_val_idx,
            xi=xi,
            nu=nu,
            reg_grid=REG_GRID,
        )
        ker = build_matern_kernel(xi=xi, nu=nu)
        Ktt = ker(Phi[split.train_idx], Phi[split.train_idx])
        alpha = solve_linear_system(Ktt + float(lam) * np.eye(Ktt.shape[0]), y2d[task_i, split.train_idx])
        y_train_pred[task_i] = Ktt @ alpha
        y_test_pred[task_i] = ker(Phi[split.test_idx], Phi[split.train_idx]) @ alpha
        out_params.append(
            {
                "task": task_name,
                "selected_alpha": None,
                "selected_lambda": float(lam),
                "selected_gamma": None,
                "selected_xi": float(xi),
                "selected_nu": float(nu),
                "selected_spectral_radius": None,
                "selected_input_scale": None,
                "selected_leak_rate": None,
                "lambda_val_mse": float(lam_val),
                "tuning_train_size": int(tune_train_idx.size),
                "tuning_val_size": int(tune_val_idx.size),
            }
        )
    return make_result(
        method="quark_reference",
        method_seed=method_seed,
        task_names=task_names,
        y2d=y2d,
        split=split,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
        best_params=out_params,
        feature_dim=int(Phi.shape[1]),
        raw_dim=None,
    )


def fit_esn_ridge(
    X: np.ndarray,
    y2d: np.ndarray,
    split: SplitData,
    task_names: Sequence[str],
    *,
    method_seed: int,
    feature_dim: int,
    backend: str = "numpy",
    device: int | None = None,
) -> dict[str, Any]:
    inner_train_idx, val_idx = make_inner_split(split.train_idx, seed=method_seed)
    combos = list(
        itertools.product(ESN_SPECTRAL_RADIUS_GRID, ESN_INPUT_SCALE_GRID, ESN_LEAK_RATE_GRID)
    )
    best_by_task: list[dict[str, Any] | None] = [None for _ in task_names]

    for spectral_radius, input_scale, leak_rate in combos:
        features = make_esn_features(
            X,
            train_idx=split.train_idx,
            feature_dim=feature_dim,
            seed=method_seed,
            spectral_radius=float(spectral_radius),
            input_scale=float(input_scale),
            leak_rate=float(leak_rate),
            backend=backend,
            device=device,
        )
        features, _ = standardize_features(features, split.train_idx)
        for task_i in range(y2d.shape[0]):
            alpha, val_mse = tune_ridge_alpha(
                features,
                y2d[task_i],
                inner_train_idx,
                val_idx,
                alphas=RIDGE_ALPHA_GRID,
                backend=backend,
                device=device,
            )
            current = best_by_task[task_i]
            if current is None or val_mse < current["val_mse"]:
                best_by_task[task_i] = {
                    "spectral_radius": float(spectral_radius),
                    "input_scale": float(input_scale),
                    "leak_rate": float(leak_rate),
                    "alpha": float(alpha),
                    "val_mse": float(val_mse),
                }

    y_train_pred = np.empty((y2d.shape[0], split.train_idx.size), dtype=float)
    y_test_pred = np.empty((y2d.shape[0], split.test_idx.size), dtype=float)
    best_params: list[dict[str, Any]] = []
    for task_i, task_name in enumerate(task_names):
        best = best_by_task[task_i]
        assert best is not None
        features = make_esn_features(
            X,
            train_idx=split.train_idx,
            feature_dim=feature_dim,
            seed=method_seed,
            spectral_radius=float(best["spectral_radius"]),
            input_scale=float(best["input_scale"]),
            leak_rate=float(best["leak_rate"]),
            backend=backend,
            device=device,
        )
        features, _ = standardize_features(features, split.train_idx)
        if resolve_backend(backend) == "cupy":
            y_train_pred[task_i] = ridge_fit_predict(
                features[split.train_idx],
                y2d[task_i, split.train_idx],
                features[split.train_idx],
                alpha=float(best["alpha"]),
                backend=backend,
                device=device,
            )
            y_test_pred[task_i] = ridge_fit_predict(
                features[split.train_idx],
                y2d[task_i, split.train_idx],
                features[split.test_idx],
                alpha=float(best["alpha"]),
                backend=backend,
                device=device,
            )
        else:
            model = Ridge(alpha=float(best["alpha"]), fit_intercept=True)
            model.fit(features[split.train_idx], y2d[task_i, split.train_idx])
            y_train_pred[task_i] = model.predict(features[split.train_idx])
            y_test_pred[task_i] = model.predict(features[split.test_idx])
        best_params.append(
            {
                "task": task_name,
                "selected_alpha": float(best["alpha"]),
                "selected_lambda": None,
                "selected_gamma": None,
                "selected_xi": None,
                "selected_nu": None,
                "selected_spectral_radius": float(best["spectral_radius"]),
                "selected_input_scale": float(best["input_scale"]),
                "selected_leak_rate": float(best["leak_rate"]),
                "val_mse": float(best["val_mse"]),
            }
        )

    return make_result(
        method="esn",
        method_seed=method_seed,
        task_names=task_names,
        y2d=y2d,
        split=split,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
        best_params=best_params,
        feature_dim=int(feature_dim),
        raw_dim=None,
    )


def make_esn_features(
    X: np.ndarray,
    *,
    train_idx: np.ndarray,
    feature_dim: int,
    seed: int,
    spectral_radius: float,
    input_scale: float,
    leak_rate: float,
    backend: str = "numpy",
    device: int | None = None,
) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    N, window_length, input_dim = X.shape
    X_train_flat = X[np.asarray(train_idx, dtype=int)].reshape(-1, input_dim)
    mean = X_train_flat.mean(axis=0)
    std = X_train_flat.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    X_std = (X - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)

    rng = np.random.default_rng(seed)
    W_in = rng.normal(
        loc=0.0,
        scale=float(input_scale) / math.sqrt(input_dim),
        size=(int(feature_dim), input_dim),
    )
    bias = rng.uniform(-0.1, 0.1, size=int(feature_dim))
    W_res = make_sparse_reservoir(
        feature_dim=int(feature_dim),
        spectral_radius=float(spectral_radius),
        density=ESN_DENSITY,
        rng=rng,
    )

    if resolve_backend(backend) == "cupy":
        cp = import_cupy()
        Xg = cp.asarray(X_std)
        W_in_g = cp.asarray(W_in)
        bias_g = cp.asarray(bias)
        W_res_g = cp.asarray(W_res.toarray())
        state_g = cp.zeros((N, int(feature_dim)), dtype=cp.float64)
        for t in range(window_length):
            recurrent = state_g @ W_res_g.T
            candidate = cp.tanh(Xg[:, t, :] @ W_in_g.T + recurrent + bias_g)
            state_g = (1.0 - float(leak_rate)) * state_g + float(leak_rate) * candidate
        return asnumpy(state_g)

    state = np.zeros((N, int(feature_dim)), dtype=float)
    for t in range(window_length):
        recurrent = (W_res @ state.T).T
        candidate = np.tanh(X_std[:, t, :] @ W_in.T + recurrent + bias)
        state = (1.0 - float(leak_rate)) * state + float(leak_rate) * candidate
    return state


def make_sparse_reservoir(
    *,
    feature_dim: int,
    spectral_radius: float,
    density: float,
    rng: np.random.Generator,
) -> sparse.csr_matrix:
    mask = rng.random((feature_dim, feature_dim)) < float(density)
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        rows = np.asarray([0])
        cols = np.asarray([0])
    values = rng.normal(size=rows.size)
    W = sparse.csr_matrix((values, (rows, cols)), shape=(feature_dim, feature_dim), dtype=float)
    radius = estimate_spectral_radius(W)
    if radius > 0:
        W = W * (float(spectral_radius) / radius)
    return W


def estimate_spectral_radius(W: sparse.csr_matrix) -> float:
    if W.shape[0] <= 8:
        vals = np.linalg.eigvals(W.toarray())
        return float(np.max(np.abs(vals))) if vals.size else 0.0
    try:
        vals = eigs(W, k=1, which="LM", return_eigenvectors=False)
        return float(np.max(np.abs(vals))) if vals.size else 0.0
    except Exception:
        vals = np.linalg.eigvals(W.toarray())
        return float(np.max(np.abs(vals))) if vals.size else 0.0


def make_result(
    *,
    method: str,
    method_seed: int,
    task_names: Sequence[str],
    y2d: np.ndarray,
    split: SplitData,
    y_train_pred: np.ndarray,
    y_test_pred: np.ndarray,
    best_params: Sequence[Mapping[str, Any]],
    feature_dim: int,
    raw_dim: int | None,
) -> dict[str, Any]:
    y_train_true = y2d[:, split.train_idx]
    y_test_true = y2d[:, split.test_idx]
    return {
        "method": method,
        "method_seed": int(method_seed),
        "task_names": list(task_names),
        "feature_dim": int(feature_dim),
        "raw_dim": raw_dim,
        "best_params": [dict(p) for p in best_params],
        "train_mse": np.asarray(
            [mean_squared_error(y_train_true[i], y_train_pred[i]) for i in range(y2d.shape[0])],
            dtype=float,
        ),
        "test_mse": np.asarray(
            [mean_squared_error(y_test_true[i], y_test_pred[i]) for i in range(y2d.shape[0])],
            dtype=float,
        ),
        "y_train_true": y_train_true,
        "y_train_pred": y_train_pred,
        "y_test_true": y_test_true,
        "y_test_pred": y_test_pred,
    }


def make_metric_rows(
    *,
    result: Mapping[str, Any],
    dataset: BenchmarkData,
    split: SplitData,
    method: str,
    method_seed: int,
    feature_dim: int,
    artifact_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best_params = list(result["best_params"])
    for task_i, task_name in enumerate(result["task_names"]):
        params = dict(best_params[task_i])
        for split_name, values in (("train", result["train_mse"]), ("test", result["test_mse"])):
            row = {col: "" for col in METRICS_COLUMNS}
            row.update(
                {
                    "experiment_id": f"{dataset.dataset_id}__{split.split_source}__{method}",
                    "method": method,
                    "dataset": dataset.dataset_id,
                    "task": task_name,
                    "split": split_name,
                    "metric": "mse",
                    "value": float(values[task_i]),
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
                row[key] = blank_if_none(params.get(key))
            rows.append(row)
    return rows


def write_run_artifacts(
    *,
    run_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    result: Mapping[str, Any],
    run_config: Mapping[str, Any],
    summary_extra: Mapping[str, Any] | None = None,
) -> None:
    write_metrics_csv(run_dir / "metrics.csv", rows)
    (run_dir / "best_params.json").write_text(
        json.dumps(result["best_params"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (run_dir / "run_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(json_safe(run_config), f, sort_keys=False)
    np.savez_compressed(
        run_dir / "predictions.npz",
        task_names=np.asarray(result["task_names"], dtype=str),
        y_train_true=np.asarray(result["y_train_true"], dtype=float),
        y_train_pred=np.asarray(result["y_train_pred"], dtype=float),
        y_test_true=np.asarray(result["y_test_true"], dtype=float),
        y_test_pred=np.asarray(result["y_test_pred"], dtype=float),
        train_mse=np.asarray(result["train_mse"], dtype=float),
        test_mse=np.asarray(result["test_mse"], dtype=float),
    )
    summary = {
        "method": result["method"],
        "method_seed": int(result["method_seed"]),
        "feature_dim": int(result["feature_dim"]),
        "train_mse": np.asarray(result["train_mse"], dtype=float).tolist(),
        "test_mse": np.asarray(result["test_mse"], dtype=float).tolist(),
        "best_params_path": str(run_dir / "best_params.json"),
        "metrics_path": str(run_dir / "metrics.csv"),
        "predictions_path": str(run_dir / "predictions.npz"),
    }
    if "extra_summary" in result:
        summary["extra_summary"] = json_safe(result["extra_summary"])
    if summary_extra:
        summary.update(json_safe(summary_extra))
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: serialize_csv_value(row.get(col, "")) for col in METRICS_COLUMNS})


def write_dict_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: serialize_csv_value(row.get(k, "")) for k in fieldnames})


def build_wide_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("metric") != "mse":
            continue
        key = (row.get("method", ""), row.get("task", ""), row.get("method_seed", ""))
        out = grouped.setdefault(key, dict(row))
        out[f"{row.get('split')}_mse"] = row.get("value", "")

    wide: list[dict[str, Any]] = []
    for (_, _, _), row in sorted(grouped.items()):
        wide.append(
            {
                "method": row.get("method", ""),
                "task": row.get("task", ""),
                "method_seed": row.get("method_seed", ""),
                "train_mse": row.get("train_mse", ""),
                "test_mse": row.get("test_mse", ""),
                "n_train": row.get("n_train", ""),
                "n_test": row.get("n_test", ""),
                "feature_dim": row.get("feature_dim", ""),
                "raw_dim": row.get("raw_dim", ""),
                "selected_alpha": row.get("selected_alpha", ""),
                "selected_lambda": row.get("selected_lambda", ""),
                "selected_gamma": row.get("selected_gamma", ""),
                "selected_xi": row.get("selected_xi", ""),
                "selected_nu": row.get("selected_nu", ""),
                "selected_spectral_radius": row.get("selected_spectral_radius", ""),
                "selected_input_scale": row.get("selected_input_scale", ""),
                "selected_leak_rate": row.get("selected_leak_rate", ""),
                "artifact_dir": row.get("artifact_dir", ""),
            }
        )
    return wide


def build_markdown_table(wide_rows: Sequence[Mapping[str, Any]]) -> str:
    methods = sorted({str(r["method"]) for r in wide_rows})
    tasks = sorted({str(r["task"]) for r in wide_rows})
    lookup = {
        (str(r["method"]), str(r["task"])): format_float(r.get("test_mse", ""))
        for r in wide_rows
    }
    lines = ["| Method | " + " | ".join(tasks) + " |"]
    lines.append("|---|" + "|".join(["---:" for _ in tasks]) + "|")
    for method in methods:
        cells = [lookup.get((method, task), "") for task in tasks]
        lines.append("| " + method + " | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def write_baseline_plot(wide_rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment guard
        path.with_suffix(".plot_error.txt").write_text(str(exc), encoding="utf-8")
        return

    tasks = sorted({str(r["task"]) for r in wide_rows})
    methods = sorted({str(r["method"]) for r in wide_rows})
    lookup = {}
    for row in wide_rows:
        try:
            lookup[(str(row["method"]), str(row["task"]))] = float(row.get("test_mse", np.nan))
        except (TypeError, ValueError):
            lookup[(str(row["method"]), str(row["task"]))] = np.nan

    x = np.arange(len(tasks))
    width = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(max(7.0, 1.6 * len(tasks) + 0.9 * len(methods)), 4.5))
    for i, method in enumerate(methods):
        values = [lookup.get((method, task), np.nan) for task in tasks]
        ax.bar(x + (i - (len(methods) - 1) / 2.0) * width, values, width=width, label=method)
    ax.set_ylabel("Test MSE")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=20, ha="right")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def update_rebuttal_response_drafts(*, aggregate_dir: Path, table_markdown: str) -> None:
    canonical_docs_dir = PROJECT_ROOT / "docs/rebuttal/responses"
    legacy_docs_dir = PROJECT_ROOT / "docs/rebutal/responses"
    docs_dir = canonical_docs_dir if canonical_docs_dir.exists() else legacy_docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    table_path = aggregate_dir / "classical_baselines_table.md"
    long_path = aggregate_dir / "classical_baselines_long.csv"
    wide_path = aggregate_dir / "classical_baselines_wide.csv"

    global_text = f"""# Global Rebuttal Experiment Map

## Classical-only baseline package

All five classical-only rebuttal baselines use the fixed paper benchmark
`N=10000, w=25, d=3, s=100`, the same three synthetic beta-mixing VARMA
functionals, and the saved QuaRK paper split from
`storage/results/reg_sweep/c-shadows/full/N=10000__w=25__d=3__s=100/2026-02-02_14-02-13/model/arrays.npz`.

| Experiment | Reviewer concern addressed | Artifact root |
|---|---|---|
| `raw_ridge` | Simple linear classical baseline | `{aggregate_dir.parent / 'raw_ridge'}` |
| `raw_matern_krr` | Kernel-readout-only baseline on raw windows | `{aggregate_dir.parent / 'raw_matern_krr'}` |
| `matched_random_features_matern_krr` | Matched non-quantum feature-map ablation with same Matérn readout | `{aggregate_dir.parent / 'matched_random_features_matern_krr'}` |
| `rff_ridge` | Standard random Fourier feature baseline | `{aggregate_dir.parent / 'rff_ridge'}` |
| `esn` | Classical reservoir computing baseline | `{aggregate_dir.parent / 'esn'}` |
| `quark_reference` | Cached QuaRK reference, no circuit rerun | `{aggregate_dir.parent / 'quark_reference'}` |

Aggregate artifacts:

- Long CSV: `{long_path}`
- Wide CSV: `{wide_path}`
- Markdown table: `{table_path}`
- Plot: `{aggregate_dir / 'classical_baselines_plot.pdf'}`

## Current aggregate table

{table_markdown}
"""
    (docs_dir / "global_view.md").write_text(global_text, encoding="utf-8")

    msds_text = f"""# MSDS Response Draft

We agree that the original empirical section did not sufficiently isolate the
quantum reservoir contribution from the downstream readout. To address this, we
added a controlled classical-only baseline package on the exact same benchmark,
using the same `N=10000, w=25, d=3, s=100` synthetic beta-mixing VARMA windows,
the same three target functionals, and the same saved train/test split as the
paper QuaRK run.

The added comparisons are: linear ridge on raw flattened windows, Matérn KRR on
raw flattened windows, a matched classical random nonlinear map with `D=315`
features followed by the same Matérn KRR readout, RBF random Fourier features
with ridge readout, and an ESN reservoir baseline with `D=315` states. The
`quark_reference` row is recomputed only from cached QuaRK features and the
saved split, so no quantum circuits are rerun for this comparison.

{table_markdown}

Artifacts are stored under `{aggregate_dir}`. The most direct answer to the
ablation concern is the comparison between `quark_reference` and
`matched_random_features_matern_krr`, since these use matched feature dimension
and the same Matérn readout while replacing the quantum reservoir with a
classical random nonlinear map.
"""
    (docs_dir / "MSDS.md").write_text(msds_text, encoding="utf-8")

    kc2j_text = f"""# kc2J Response Draft

We added classical comparative baselines on the same fixed benchmark used in
the paper: `N=10000, w=25, d=3, s=100`, three beta-mixing VARMA functionals, and
the exact same saved train/test split. This avoids confounding the comparison
with a change in dataset or split.

The new package includes raw-window linear ridge, raw-window Matérn KRR,
matched classical random nonlinear features with the same Matérn readout, RBF
random Fourier features with ridge readout, and an ESN reservoir baseline. We
also include a `quark_reference` row evaluated from cached QuaRK features only.

{table_markdown}

The raw-window Matérn KRR row addresses whether the kernel readout alone is
sufficient on the original inputs, while the matched random-feature row directly
tests whether replacing the quantum reservoir by a classical random nonlinear
map of the same dimension can explain the observed performance.
"""
    (docs_dir / "kc2J.md").write_text(kc2j_text, encoding="utf-8")

    vkfl_text = f"""# vKFL Response Draft

Within the rebuttal window, we prioritized controlled baselines and ablations on
the fixed benchmark rather than changing the dataset at the same time as the
model comparison. All added classical-only experiments use the paper's saved
`N=10000, w=25, d=3, s=100` split, and the QuaRK reference row is evaluated from
cached features without rerunning quantum circuits.

{table_markdown}

These results strengthen the empirical comparison against standard classical
alternatives. We view real-world time-series experiments as the next extension,
but kept the rebuttal package focused on the most direct controlled comparison
requested by the reviewers.
"""
    (docs_dir / "vKFL.md").write_text(vkfl_text, encoding="utf-8")


def solve_linear_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def camel_to_snake(name: str) -> str:
    out: list[str] = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def blank_if_none(value: Any) -> Any:
    return "" if value is None else value


def serialize_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(float(value)):
            return ""
        return f"{float(value):.17g}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    return str(value)


def format_float(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(val):
        return ""
    return f"{val:.4g}"


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value
