"""Finite, chronology-preserving Matérn KRR used by the empirical protocol."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

import numpy as np
from scipy.special import gamma as gamma_function, kv

from src.experiment.metrics import mse, training_normalized_rmse
from src.experiment.protocol import ChronologicalSplit


def _xp_for(backend: str):
    if backend == "numpy":
        return np
    if backend != "cupy":
        raise ValueError("readout backend must be 'numpy' or 'cupy'.")
    try:
        import cupy as cp
    except ImportError as exc:
        raise ImportError("The CuPy readout backend requires CuPy.") from exc
    return cp


def _asnumpy(value: Any) -> np.ndarray:
    return value.get() if hasattr(value, "get") else np.asarray(value)


def _synchronize(xp: Any) -> None:
    if xp is not np:
        xp.cuda.Stream.null.synchronize()


@dataclass(frozen=True)
class FeatureStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    constant_mask: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "FeatureStandardizer":
        X = np.asarray(features, dtype=np.float64)
        mean = np.mean(X, axis=0)
        scale = np.std(X, axis=0, ddof=0)
        mask = scale <= 1e-15
        safe_scale = scale.copy()
        safe_scale[mask] = 1.0
        return cls(mean=mean, scale=safe_scale, constant_mask=mask)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (np.asarray(features, dtype=np.float64) - self.mean) / self.scale


def pairwise_sq_dists(X: Any, Y: Any, *, xp: Any) -> Any:
    Xb = xp.asarray(X, dtype=xp.float64)
    Yb = xp.asarray(Y, dtype=xp.float64)
    d2 = (
        xp.sum(Xb * Xb, axis=1)[:, None]
        + xp.sum(Yb * Yb, axis=1)[None, :]
        - 2.0 * (Xb @ Yb.T)
    )
    return xp.maximum(d2, 0.0)


def matern_from_sq_dists(d2: Any, *, nu: float, xi: float, xp: Any) -> Any:
    """Evaluate the normalized Matérn profile, including exact generic nu."""

    nu = float(nu)
    xi = float(xi)
    if nu <= 0.0 or xi <= 0.0:
        raise ValueError("Matérn nu and xi must be positive.")
    r = xp.sqrt(d2) / xi
    if np.isclose(nu, 0.5):
        return xp.exp(-r)
    if np.isclose(nu, 1.5):
        z = np.sqrt(3.0) * r
        return (1.0 + z) * xp.exp(-z)
    if np.isclose(nu, 2.5):
        z = np.sqrt(5.0) * r
        return (1.0 + z + z * z / 3.0) * xp.exp(-z)

    host_r = np.sqrt(_asnumpy(d2)) / xi
    z = np.sqrt(2.0 * nu) * host_r
    result = np.ones_like(z, dtype=np.float64)
    positive = z > 0.0
    zp = z[positive]
    result[positive] = (
        (2.0 ** (1.0 - nu) / float(gamma_function(nu)))
        * (zp**nu)
        * kv(nu, zp)
    )
    if not np.all(np.isfinite(result)):
        raise FloatingPointError(f"Nonfinite Matérn values for nu={nu}, xi={xi}.")
    return xp.asarray(result) if xp is not np else result


def _uses_closed_form(nu: float) -> bool:
    return any(np.isclose(float(nu), value) for value in (0.5, 1.5, 2.5))


def _matern_with_host_cache(
    d2: Any,
    host_d2: np.ndarray | None,
    *,
    nu: float,
    xi: float,
    xp: Any,
) -> Any:
    if _uses_closed_form(nu):
        return matern_from_sq_dists(d2, nu=nu, xi=xi, xp=xp)
    if host_d2 is None:
        host_d2 = _asnumpy(d2)
    host_kernel = matern_from_sq_dists(host_d2, nu=nu, xi=xi, xp=np)
    if xp is np:
        return host_kernel
    return xp.asarray(host_kernel, dtype=xp.float64)


def _solve_shifted(gram: Any, targets: Any, regularization: float, *, xp: Any) -> Any:
    n = int(gram.shape[0])
    system = gram + float(regularization) * xp.eye(n, dtype=xp.float64)
    chol = xp.linalg.cholesky(system)
    return xp.linalg.solve(chol.T, xp.linalg.solve(chol, targets))


def _rkhs_norm_backend(alpha: Any, gram: Any, *, xp: Any) -> float:
    value = float(_asnumpy(alpha.T @ gram @ alpha))
    scale = max(1.0, abs(value))
    if value < -1e-12 * scale:
        raise FloatingPointError(f"Negative RKHS squared norm: {value}.")
    return float(np.sqrt(max(0.0, value)))


def _condition_backend(gram: Any, regularization: float, *, xp: Any) -> tuple[float, float, str]:
    eigenvalues = _asnumpy(xp.linalg.eigvalsh(0.5 * (gram + gram.T)))
    return _condition_from_eigenvalues(eigenvalues, regularization)


def _condition_from_eigenvalues(
    eigenvalues: np.ndarray, regularization: float
) -> tuple[float, float, str]:
    largest = float(np.max(eigenvalues))
    smallest = float(np.min(eigenvalues))
    singular = largest <= 0.0 or smallest < 1e-14 * largest
    base = float("inf") if singular else largest / smallest
    shifted = (largest + regularization) / (max(smallest, 0.0) + regularization)
    return base, float(shifted), "numerical_singularity" if singular else "ok"


@dataclass
class GridReadoutResult:
    candidate_rows: list[dict[str, Any]]
    selected: dict[str, dict[str, Any]]
    predictions: np.ndarray
    alphas: np.ndarray
    metric_rows: list[dict[str, Any]]
    scaler: FeatureStandardizer
    elapsed_seconds: float
    diagnostics: dict[str, Any]


class FiniteGridMaternKRR:
    """Predeclared finite-grid KRR with validation-only selection."""

    def __init__(
        self,
        *,
        nu_grid: Sequence[float],
        xi_grid: Sequence[float],
        lambda_grid: Sequence[float],
        backend: str = "cupy",
        tie_tolerance: float = 1e-12,
    ) -> None:
        self.nu_grid = tuple(sorted(float(value) for value in nu_grid))
        self.xi_grid = tuple(sorted(float(value) for value in xi_grid))
        self.lambda_grid = tuple(sorted(float(value) for value in lambda_grid))
        self.backend = str(backend)
        self.tie_tolerance = float(tie_tolerance)
        if not self.nu_grid or not self.xi_grid or not self.lambda_grid:
            raise ValueError("Every finite-grid axis must be nonempty.")
        if min(self.nu_grid + self.xi_grid + self.lambda_grid) <= 0.0:
            raise ValueError("All Matérn grid values must be positive.")

    @property
    def candidate_count(self) -> int:
        return len(self.nu_grid) * len(self.xi_grid) * len(self.lambda_grid)

    def fit_select_refit(
        self,
        *,
        features: np.ndarray,
        targets: np.ndarray,
        task_names: Sequence[str],
        split: ChronologicalSplit,
    ) -> GridReadoutResult:
        started = perf_counter()
        X = np.asarray(features, dtype=np.float64)
        Y = np.asarray(targets, dtype=np.float64)
        names = tuple(str(name) for name in task_names)
        if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
            raise ValueError("features and targets must have shapes (N,p) and (N,L).")
        if Y.shape[1] != len(names) or split.test_stop > X.shape[0]:
            raise ValueError("Task names or split do not match the supplied arrays.")
        xp = _xp_for(self.backend)

        inner, validation = split.inner_train, split.validation
        scaler_selection = FeatureStandardizer.fit(X[inner])
        X_inner = scaler_selection.transform(X[inner])
        X_validation = scaler_selection.transform(X[validation])
        Y_inner = xp.asarray(Y[inner], dtype=xp.float64)
        _synchronize(xp)
        distance_started = perf_counter()
        d2_inner = pairwise_sq_dists(X_inner, X_inner, xp=xp)
        d2_validation = pairwise_sq_dists(X_validation, X_inner, xp=xp)
        _synchronize(xp)
        distance_seconds = perf_counter() - distance_started

        requires_host_kernel = any(
            not _uses_closed_form(nu) for nu in self.nu_grid
        )
        d2_inner_host = _asnumpy(d2_inner) if requires_host_kernel else None
        d2_validation_host = (
            _asnumpy(d2_validation) if requires_host_kernel else None
        )

        candidate_rows: list[dict[str, Any]] = []
        kernel_diagnostics: list[dict[str, Any]] = []
        solve_diagnostics: list[dict[str, Any]] = []
        for nu in self.nu_grid:
            for xi in self.xi_grid:
                _synchronize(xp)
                kernel_started = perf_counter()
                K_inner = _matern_with_host_cache(
                    d2_inner, d2_inner_host, nu=nu, xi=xi, xp=xp
                )
                K_validation = _matern_with_host_cache(
                    d2_validation,
                    d2_validation_host,
                    nu=nu,
                    xi=xi,
                    xp=xp,
                )
                _synchronize(xp)
                kernel_seconds = perf_counter() - kernel_started
                kernel_diagnostics.append(
                    {"nu": nu, "xi": xi, "seconds": kernel_seconds}
                )
                for regularization in self.lambda_grid:
                    _synchronize(xp)
                    solve_started = perf_counter()
                    try:
                        alpha = _solve_shifted(
                            K_inner, Y_inner, regularization, xp=xp
                        )
                        prediction = _asnumpy(K_validation @ alpha)
                        values = np.mean((prediction - Y[validation]) ** 2, axis=0)
                        status = "ok"
                    except Exception as exc:
                        values = np.full(len(names), np.nan)
                        status = f"invalid:{type(exc).__name__}"
                    _synchronize(xp)
                    solve_seconds = perf_counter() - solve_started
                    solve_diagnostics.append(
                        {
                            "nu": nu,
                            "xi": xi,
                            "lambda_K": regularization,
                            "seconds": solve_seconds,
                            "status": status,
                        }
                    )
                    for task_index, task_name in enumerate(names):
                        candidate_rows.append(
                            {
                                "task": task_name,
                                "nu": nu,
                                "xi": xi,
                                "lambda_K": regularization,
                                "eta": regularization / len(inner),
                                "validation_mse": float(values[task_index]),
                                "status": status,
                            }
                        )
                del K_inner, K_validation

        selected: dict[str, dict[str, Any]] = {}
        for task_name in names:
            valid = [
                row
                for row in candidate_rows
                if row["task"] == task_name
                and row["status"] == "ok"
                and np.isfinite(row["validation_mse"])
            ]
            if not valid:
                raise RuntimeError(f"All KRR candidates failed for task {task_name!r}.")
            minimum = min(float(row["validation_mse"]) for row in valid)
            tied = [
                row
                for row in valid
                if float(row["validation_mse"]) <= minimum + self.tie_tolerance
            ]
            choice = min(
                tied,
                key=lambda row: (row["nu"], row["xi"], row["lambda_K"]),
            )
            selected[task_name] = dict(choice)

        outer, test = split.outer_train, split.test
        scaler_final = FeatureStandardizer.fit(X[outer])
        X_outer = scaler_final.transform(X[outer])
        X_test = scaler_final.transform(X[test])
        d2_outer = pairwise_sq_dists(X_outer, X_outer, xp=xp)
        d2_test = pairwise_sq_dists(X_test, X_outer, xp=xp)
        selected_uses_host = any(
            not _uses_closed_form(float(choice["nu"]))
            for choice in selected.values()
        )
        d2_outer_host = _asnumpy(d2_outer) if selected_uses_host else None
        d2_test_host = _asnumpy(d2_test) if selected_uses_host else None
        predictions = np.full_like(Y, np.nan, dtype=float)
        alphas = np.full((len(outer), len(names)), np.nan, dtype=float)
        metric_by_task: dict[str, dict[str, Any]] = {}

        refit_groups: dict[tuple[float, float, float], list[int]] = {}
        for task_index, task_name in enumerate(names):
            choice = selected[task_name]
            group_key = (
                float(choice["nu"]),
                float(choice["xi"]),
                float(choice["lambda_K"]),
            )
            refit_groups.setdefault(group_key, []).append(task_index)

        kernel_cache: dict[tuple[float, float], tuple[Any, Any]] = {}
        for (nu, xi, regularization), task_indices in refit_groups.items():
            kernel_key = (nu, xi)
            if kernel_key not in kernel_cache:
                kernel_cache[kernel_key] = (
                    _matern_with_host_cache(
                        d2_outer, d2_outer_host, nu=nu, xi=xi, xp=xp
                    ),
                    _matern_with_host_cache(
                        d2_test, d2_test_host, nu=nu, xi=xi, xp=xp
                    ),
                )
            K_outer, K_test = kernel_cache[kernel_key]
            group_targets = xp.asarray(
                Y[outer][:, task_indices], dtype=xp.float64
            )
            alpha_group = _solve_shifted(
                K_outer, group_targets, regularization, xp=xp
            )
            pred_outer_group = _asnumpy(K_outer @ alpha_group)
            pred_test_group = _asnumpy(K_test @ alpha_group)
            alpha_group_host = _asnumpy(alpha_group)
            for group_column, task_index in enumerate(task_indices):
                task_name = names[task_index]
                choice = selected[task_name]
                pred_outer = pred_outer_group[:, group_column]
                pred_test = pred_test_group[:, group_column]
                alpha = alpha_group[:, group_column]
                predictions[outer, task_index] = pred_outer
                predictions[test, task_index] = pred_test
                alphas[:, task_index] = alpha_group_host[:, group_column]
                metric_by_task[task_name] = {
                    "task": task_name,
                    "train_mse": mse(Y[outer, task_index], pred_outer),
                    "validation_mse": float(choice["validation_mse"]),
                    "test_mse": mse(Y[test, task_index], pred_test),
                    "test_nrmse": training_normalized_rmse(
                        Y[test, task_index], pred_test, Y[outer, task_index]
                    ),
                    "rkhs_norm": _rkhs_norm_backend(alpha, K_outer, xp=xp),
                    "nu": nu,
                    "xi": xi,
                    "lambda_K": regularization,
                    "eta": regularization / len(outer),
                    "inner_train_size": len(inner),
                    "outer_train_size": len(outer),
                    "test_size": len(test),
                }
        metric_rows = [metric_by_task[name] for name in names]
        _synchronize(xp)
        return GridReadoutResult(
            candidate_rows=candidate_rows,
            selected=selected,
            predictions=predictions,
            alphas=alphas,
            metric_rows=metric_rows,
            scaler=scaler_final,
            elapsed_seconds=perf_counter() - started,
            diagnostics={
                "refit_group_count": len(refit_groups),
                "distance_seconds": distance_seconds,
                "kernel_rows": kernel_diagnostics,
                "solve_rows": solve_diagnostics,
            },
        )


def regularization_path(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    task_names: Sequence[str],
    split: ChronologicalSplit,
    lambda_grid: Sequence[float],
    nu: float = 2.5,
    xi: float = 1.0,
    backend: str = "cupy",
) -> list[dict[str, Any]]:
    X = np.asarray(features, dtype=np.float64)
    Y = np.asarray(targets, dtype=np.float64)
    names = tuple(str(name) for name in task_names)
    xp = _xp_for(backend)
    scaler = FeatureStandardizer.fit(X[split.inner_train])
    X_train = scaler.transform(X[split.inner_train])
    X_val = scaler.transform(X[split.validation])
    X_test = scaler.transform(X[split.test])
    d2_train = pairwise_sq_dists(X_train, X_train, xp=xp)
    K_train = matern_from_sq_dists(d2_train, nu=nu, xi=xi, xp=xp)
    K_val = matern_from_sq_dists(
        pairwise_sq_dists(X_val, X_train, xp=xp), nu=nu, xi=xi, xp=xp
    )
    K_test = matern_from_sq_dists(
        pairwise_sq_dists(X_test, X_train, xp=xp), nu=nu, xi=xi, xp=xp
    )
    eigenvalues = _asnumpy(xp.linalg.eigvalsh(0.5 * (K_train + K_train.T)))
    y_train = xp.asarray(Y[split.inner_train], dtype=xp.float64)
    rows: list[dict[str, Any]] = []
    for regularization in sorted(float(value) for value in lambda_grid):
        try:
            alpha = _solve_shifted(K_train, y_train, regularization, xp=xp)
            pred_train = _asnumpy(K_train @ alpha)
            pred_val = _asnumpy(K_val @ alpha)
            pred_test = _asnumpy(K_test @ alpha)
            base_condition, shifted_condition, condition_status = _condition_from_eigenvalues(
                eigenvalues, regularization
            )
            for task_index, name in enumerate(names):
                rows.append(
                    {
                        "task": name,
                        "nu": float(nu),
                        "xi": float(xi),
                        "lambda_K": regularization,
                        "eta": regularization / len(split.inner_train),
                        "train_mse": mse(
                            Y[split.inner_train, task_index], pred_train[:, task_index]
                        ),
                        "validation_mse": mse(
                            Y[split.validation, task_index], pred_val[:, task_index]
                        ),
                        "test_mse": mse(Y[split.test, task_index], pred_test[:, task_index]),
                        "rkhs_norm": _rkhs_norm_backend(alpha[:, task_index], K_train, xp=xp),
                        "condition_K": base_condition,
                        "condition_regularized": shifted_condition,
                        "condition_status": condition_status,
                        "status": "ok",
                    }
                )
        except Exception as exc:
            for name in names:
                rows.append(
                    {
                        "task": name,
                        "nu": float(nu),
                        "xi": float(xi),
                        "lambda_K": regularization,
                        "status": f"invalid:{type(exc).__name__}",
                    }
                )
    _synchronize(xp)
    return rows
