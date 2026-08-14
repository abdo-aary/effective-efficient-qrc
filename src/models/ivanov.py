"""Literal RMS-Matérn Ivanov least-squares readout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .matern_grid import matern_from_sq_dists, pairwise_sq_dists


@dataclass(frozen=True)
class IvanovFit:
    alphas: np.ndarray
    task_names: tuple[str, ...]
    nu: float
    xi: float
    radius: float
    tolerance: float
    diagnostics: tuple[dict[str, Any], ...]
    eigenvalues: np.ndarray

    def __post_init__(self) -> None:
        for name in ("alphas", "eigenvalues"):
            array = np.array(getattr(self, name), copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)


def rms_matern_kernel(
    left: np.ndarray,
    right: np.ndarray,
    *,
    nu: float,
    xi: float,
) -> np.ndarray:
    X = np.asarray(left, dtype=np.float64)
    Y = np.asarray(right, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError("Kernel inputs must have shapes (N,p) and (M,p).")
    if X.shape[1] < 1:
        raise ValueError("The RMS Matérn kernel requires at least one feature.")
    d2 = pairwise_sq_dists(X, Y, xp=np) / float(X.shape[1])
    kernel = matern_from_sq_dists(d2, nu=float(nu), xi=float(xi), xp=np)
    return np.asarray(kernel, dtype=np.float64)


def fit_rms_matern_ivanov(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    task_names: Sequence[str],
    nu: float,
    xi: float,
    radius: float,
    tolerance: float,
    spectral_clip: float = 1e-12,
) -> IvanovFit:
    X = np.asarray(features, dtype=np.float64)
    Y = np.asarray(targets, dtype=np.float64)
    names = tuple(str(item) for item in task_names)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError("Features and targets must have shapes (N,p) and (N,L).")
    if Y.shape[1] != len(names) or not names:
        raise ValueError("Task names must match the target columns.")
    if min(float(nu), float(xi), float(radius), float(tolerance)) <= 0.0:
        raise ValueError("Matérn and Ivanov parameters and tolerance must be positive.")
    if spectral_clip <= 0.0:
        raise ValueError("spectral_clip must be positive.")

    gram = rms_matern_kernel(X, X, nu=nu, xi=xi)
    gram = 0.5 * (gram + gram.T)
    np.fill_diagonal(gram, 1.0)
    eigenvalues_raw, eigenvectors = np.linalg.eigh(gram)
    scale = max(1.0, float(np.max(np.abs(eigenvalues_raw))))
    minimum = float(eigenvalues_raw[0])
    if minimum < -float(spectral_clip) * scale:
        raise FloatingPointError(
            f"Matérn Gram matrix has a materially negative eigenvalue {minimum}."
        )
    # The policy clips negative roundoff only. A separate numerical-rank
    # threshold controls the minimum-norm pseudoinverse.
    eigenvalues = np.maximum(eigenvalues_raw, 0.0)
    positive = eigenvalues > spectral_clip * scale
    projected_targets = eigenvectors.T @ Y
    alphas = np.empty((X.shape[0], Y.shape[1]), dtype=np.float64)
    diagnostics: list[dict[str, Any]] = []

    for column, task in enumerate(names):
        b = projected_targets[:, column]
        coefficients = np.zeros_like(b)
        coefficients[positive] = b[positive] / eigenvalues[positive]
        unconstrained_norm_sq = float(
            np.sum(eigenvalues[positive] * coefficients[positive] ** 2)
        )
        if unconstrained_norm_sq <= radius * radius + tolerance:
            multiplier = 0.0
            active = False
        else:
            multiplier = _solve_multiplier(
                eigenvalues[positive],
                b[positive],
                radius=float(radius),
                tolerance=float(tolerance),
            )
            coefficients.fill(0.0)
            coefficients[positive] = b[positive] / (
                eigenvalues[positive] + multiplier
            )
            active = True

        alpha = eigenvectors @ coefficients
        alphas[:, column] = alpha
        prediction = gram @ alpha
        rkhs_norm_sq = float(alpha @ gram @ alpha)
        stationarity = gram @ (prediction - Y[:, column]) + multiplier * (
            gram @ alpha
        )
        stationarity_scale = max(
            1.0,
            float(np.linalg.norm(gram @ Y[:, column])),
            float(np.linalg.norm(multiplier * gram @ alpha)),
        )
        feasibility = max(0.0, rkhs_norm_sq - radius * radius)
        complementarity = multiplier * (rkhs_norm_sq - radius * radius)
        diagnostics.append(
            {
                "task": task,
                "lagrange_multiplier": float(multiplier),
                "rkhs_norm": float(np.sqrt(max(0.0, rkhs_norm_sq))),
                "active_constraint": active,
                "stationarity_residual": float(
                    np.linalg.norm(stationarity) / stationarity_scale
                ),
                "primal_feasibility_residual": float(feasibility),
                "complementarity_residual": float(abs(complementarity)),
                "unconstrained_rkhs_norm": float(
                    np.sqrt(max(0.0, unconstrained_norm_sq))
                ),
            }
        )

    eigenvalues.setflags(write=False)
    alphas.setflags(write=False)
    return IvanovFit(
        alphas=alphas,
        task_names=names,
        nu=float(nu),
        xi=float(xi),
        radius=float(radius),
        tolerance=float(tolerance),
        diagnostics=tuple(diagnostics),
        eigenvalues=eigenvalues,
    )


def predict_rms_matern_ivanov(
    test_features: np.ndarray,
    train_features: np.ndarray,
    fit: IvanovFit | np.ndarray,
    *,
    nu: float,
    xi: float,
) -> np.ndarray:
    alphas = fit.alphas if isinstance(fit, IvanovFit) else np.asarray(fit)
    return rms_matern_kernel(test_features, train_features, nu=nu, xi=xi) @ alphas


def _solve_multiplier(
    eigenvalues: np.ndarray,
    projected_target: np.ndarray,
    *,
    radius: float,
    tolerance: float,
) -> float:
    radius_sq = radius * radius

    def norm_sq(multiplier: float) -> float:
        denominator = eigenvalues + multiplier
        return float(
            np.sum(eigenvalues * projected_target * projected_target / denominator**2)
        )

    lower = 0.0
    upper = max(1.0, float(np.max(eigenvalues)))
    while norm_sq(upper) > radius_sq:
        upper *= 2.0
        if not np.isfinite(upper):
            raise FloatingPointError("Could not bracket the Ivanov multiplier.")

    scale = max(1.0, radius_sq)
    for _ in range(256):
        midpoint = 0.5 * (lower + upper)
        value = norm_sq(midpoint)
        if abs(value - radius_sq) <= tolerance * scale:
            return midpoint
        if value > radius_sq:
            lower = midpoint
        else:
            upper = midpoint
        if upper - lower <= tolerance * max(1.0, upper):
            return 0.5 * (lower + upper)
    raise RuntimeError("Ivanov multiplier root finding did not converge.")
