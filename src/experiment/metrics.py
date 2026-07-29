"""Numerically explicit empirical metrics for E1."""

from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape or true.size == 0:
        raise ValueError("MSE inputs must have the same nonempty shape.")
    value = float(np.mean((true - pred) ** 2))
    if not np.isfinite(value):
        raise FloatingPointError("MSE is not finite.")
    return value


def training_normalized_rmse(
    y_true: np.ndarray, y_pred: np.ndarray, y_training: np.ndarray
) -> float:
    sigma = float(np.std(np.asarray(y_training, dtype=float), ddof=0))
    if not np.isfinite(sigma) or sigma <= 1e-12:
        raise ValueError("Training target standard deviation is invalid or near zero.")
    return float(np.sqrt(mse(y_true, y_pred)) / sigma)


def rkhs_norm(alpha: np.ndarray, gram: np.ndarray) -> float:
    a = np.asarray(alpha, dtype=float).reshape(-1)
    K = np.asarray(gram, dtype=float)
    value = float(a @ K @ a)
    scale = max(1.0, float(np.linalg.norm(a) ** 2 * np.linalg.norm(K, ord=2)))
    if value < -1e-12 * scale:
        raise FloatingPointError(f"Invalid negative RKHS squared norm: {value}.")
    return float(np.sqrt(max(0.0, value)))


def gram_condition_numbers(
    gram: np.ndarray, regularization: float
) -> tuple[float, float, str]:
    K = np.asarray(gram, dtype=np.float64)
    singular_values = np.linalg.svd(K, compute_uv=False)
    largest = float(singular_values[0])
    smallest = float(singular_values[-1])
    singular = smallest < 1e-14 * largest
    condition = float("inf") if singular else largest / smallest
    shifted = (largest + float(regularization)) / (
        max(smallest, 0.0) + float(regularization)
    )
    return condition, float(shifted), "numerical_singularity" if singular else "ok"
