"""GPU-friendly Matérn kernel ridge utilities.

The CuPy path supports closed-form half-integer Matérn kernels used in the fast
rebuttal sweeps: ``nu in {0.5, 1.5, 2.5}``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.compute.backend import asnumpy, device_context, import_cupy, resolve_backend


GPU_MATERN_NU_GRID = (0.5, 1.5, 2.5)


def pairwise_sq_dists(X: Any, Y: Any, *, backend: str = "cupy", dtype: str = "float64", device: int | None = None) -> Any:
    """Compute pairwise squared Euclidean distances on NumPy or CuPy."""

    resolved = resolve_backend(backend)
    xp = np if resolved == "numpy" else import_cupy()
    with device_context(device) if resolved == "cupy" else _null_device():
        Xb = xp.asarray(X, dtype=xp.dtype(dtype))
        Yb = xp.asarray(Y, dtype=xp.dtype(dtype))
        x2 = xp.sum(Xb * Xb, axis=1)[:, None]
        y2 = xp.sum(Yb * Yb, axis=1)[None, :]
        d2 = x2 + y2 - 2.0 * (Xb @ Yb.T)
        return xp.maximum(d2, 0.0)


def matern_kernel_matrix(
    X: Any,
    Y: Any,
    *,
    xi: float,
    nu: float,
    backend: str = "cupy",
    dtype: str = "float64",
    device: int | None = None,
) -> Any:
    """Build a Matérn Gram matrix for half-integer smoothness values."""

    nu = float(nu)
    if not any(np.isclose(nu, allowed) for allowed in GPU_MATERN_NU_GRID):
        raise ValueError(f"GPU Matérn supports nu in {GPU_MATERN_NU_GRID}, got {nu}.")
    if float(xi) <= 0.0:
        raise ValueError(f"xi must be positive, got {xi}.")

    resolved = resolve_backend(backend)
    xp = np if resolved == "numpy" else import_cupy()
    d2 = pairwise_sq_dists(X, Y, backend=resolved, dtype=dtype, device=device)
    r = xp.sqrt(d2) / float(xi)

    if np.isclose(nu, 0.5):
        K = xp.exp(-r)
    elif np.isclose(nu, 1.5):
        z = np.sqrt(3.0) * r
        K = (1.0 + z) * xp.exp(-z)
    else:
        z = np.sqrt(5.0) * r
        K = (1.0 + z + (z * z) / 3.0) * xp.exp(-z)
    return K.astype(xp.dtype(dtype), copy=False)


def solve_krr(K_train: Any, y_train: Any, *, reg: float, backend: str = "cupy", dtype: str = "float64") -> Any:
    """Solve ``(K + reg I) alpha = y`` on the selected backend."""

    resolved = resolve_backend(backend)
    xp = np if resolved == "numpy" else import_cupy()
    K = xp.asarray(K_train, dtype=xp.dtype(dtype))
    y = xp.asarray(y_train, dtype=xp.dtype(dtype)).reshape(-1)
    A = K + float(reg) * xp.eye(K.shape[0], dtype=K.dtype)
    return xp.linalg.solve(A, y)


def krr_validation_mse(
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    *,
    xi: float,
    nu: float,
    reg: float,
    backend: str = "cupy",
    dtype: str = "float64",
    device: int | None = None,
) -> float:
    """Compute validation MSE for a Matérn KRR candidate."""

    resolved = resolve_backend(backend)
    xp = np if resolved == "numpy" else import_cupy()
    with device_context(device) if resolved == "cupy" else _null_device():
        Ktt = matern_kernel_matrix(X_train, X_train, xi=xi, nu=nu, backend=resolved, dtype=dtype, device=device)
        Kvt = matern_kernel_matrix(X_val, X_train, xi=xi, nu=nu, backend=resolved, dtype=dtype, device=device)
        alpha = solve_krr(Ktt, y_train, reg=reg, backend=resolved, dtype=dtype)
        yhat = Kvt @ alpha
        yv = xp.asarray(y_val, dtype=xp.dtype(dtype)).reshape(-1)
        return float(asnumpy(xp.mean((yhat - yv) ** 2)))


def krr_lambda_sweep(
    K_train: Any,
    K_eval_train: Any,
    y_train: Any,
    y_eval: Any,
    lambda_grid: Sequence[float],
    *,
    backend: str = "cupy",
    dtype: str = "float64",
) -> dict[str, np.ndarray]:
    """Evaluate a KRR lambda sweep using one eigen-decomposition."""

    resolved = resolve_backend(backend)
    xp = np if resolved == "numpy" else import_cupy()
    K = xp.asarray(K_train, dtype=xp.dtype(dtype))
    Kev = xp.asarray(K_eval_train, dtype=xp.dtype(dtype))
    y = xp.asarray(y_train, dtype=xp.dtype(dtype)).reshape(-1)
    yv = xp.asarray(y_eval, dtype=xp.dtype(dtype)).reshape(-1)
    reg_grid = np.asarray(lambda_grid, dtype=float).reshape(-1)
    eigvals, eigvecs = xp.linalg.eigh(K)
    projected_y = eigvecs.T @ y
    train_mse = []
    eval_mse = []
    alpha_grid = []
    for reg in reg_grid:
        alpha = eigvecs @ (projected_y / (eigvals + float(reg)))
        alpha_grid.append(asnumpy(alpha))
        train_mse.append(float(asnumpy(xp.mean((K @ alpha - y) ** 2))))
        eval_mse.append(float(asnumpy(xp.mean((Kev @ alpha - yv) ** 2))))
    return {
        "lambda_grid": reg_grid,
        "alpha_grid": np.stack(alpha_grid, axis=0),
        "train_mse": np.asarray(train_mse, dtype=float),
        "eval_mse": np.asarray(eval_mse, dtype=float),
    }


def fit_predict_matern_krr(
    X_train: Any,
    y_train: Any,
    X_eval: Any,
    *,
    xi: float,
    nu: float,
    reg: float,
    backend: str = "cupy",
    dtype: str = "float64",
    device: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Matérn KRR and return ``(alpha, y_eval_pred)`` as NumPy arrays."""

    resolved = resolve_backend(backend)
    with device_context(device) if resolved == "cupy" else _null_device():
        Ktt = matern_kernel_matrix(X_train, X_train, xi=xi, nu=nu, backend=resolved, dtype=dtype, device=device)
        Kxt = matern_kernel_matrix(X_eval, X_train, xi=xi, nu=nu, backend=resolved, dtype=dtype, device=device)
        alpha = solve_krr(Ktt, y_train, reg=reg, backend=resolved, dtype=dtype)
        return asnumpy(alpha), asnumpy(Kxt @ alpha)


def ridge_fit_predict(
    X_train: Any,
    y_train: Any,
    X_eval: Any,
    *,
    alpha: float,
    backend: str = "cupy",
    dtype: str = "float64",
    device: int | None = None,
) -> np.ndarray:
    """Fit ridge with an intercept and return evaluation predictions."""

    resolved = resolve_backend(backend)
    xp = np if resolved == "numpy" else import_cupy()
    with device_context(device) if resolved == "cupy" else _null_device():
        Xtr = xp.asarray(X_train, dtype=xp.dtype(dtype))
        Xev = xp.asarray(X_eval, dtype=xp.dtype(dtype))
        y = xp.asarray(y_train, dtype=xp.dtype(dtype)).reshape(-1)
        x_mean = xp.mean(Xtr, axis=0)
        y_mean = xp.mean(y)
        Xc = Xtr - x_mean
        Xec = Xev - x_mean
        yc = y - y_mean
        if Xc.shape[1] <= Xc.shape[0]:
            A = Xc.T @ Xc + float(alpha) * xp.eye(Xc.shape[1], dtype=Xc.dtype)
            coef = xp.linalg.solve(A, Xc.T @ yc)
            pred = Xec @ coef + y_mean
        else:
            K = Xc @ Xc.T
            dual = xp.linalg.solve(K + float(alpha) * xp.eye(K.shape[0], dtype=K.dtype), yc)
            pred = (Xec @ Xc.T) @ dual + y_mean
        return asnumpy(pred)


class _null_device:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False
