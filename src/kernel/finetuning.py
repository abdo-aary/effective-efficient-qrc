import numpy as np
from typing import Tuple, Dict, Sequence
from scipy.optimize import minimize, minimize_scalar

from sklearn.gaussian_process.kernels import Matern as SkMatern
from sklearn.gaussian_process.kernels import ConstantKernel


def krr_val_mse_for_params(
        Xtr: np.ndarray,
        ytr: np.ndarray,
        Xva: np.ndarray,
        yva: np.ndarray,
        log_xi: float,
        nu: float,
        reg: float,
) -> float:
    xi = float(np.exp(log_xi))
    # Build kernel with fixed params
    ker = ConstantKernel(1.0, constant_value_bounds="fixed") * SkMatern(
        length_scale=xi, length_scale_bounds="fixed", nu=float(nu)
    )

    Ktt = ker(Xtr, Xtr)
    Kvt = ker(Xva, Xtr)

    A = Ktt + reg * np.eye(Ktt.shape[0])
    alpha = np.linalg.solve(A, ytr)
    yhat = Kvt @ alpha
    return float(np.mean((yhat - yva) ** 2))


def tune_matern_continuous_train_val(
        X: np.ndarray,
        y: np.ndarray,
        *,
        val_ratio: float = 0.2,
        seed: int = 0,
        reg: float = 1e-6,
        # bounds for xi and nu
        xi_bounds: Tuple[float, float] = (1e-3, 1e3),
        nu_bounds: Tuple[float, float] = (0.2, 5.0),
        n_restarts: int = 8,
) -> Tuple[Dict, float]:
    """
    Continuous search over (xi, nu) to minimize validation MSE (single split).
    Returns: (best_params_dict, best_mse).
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idx = rng.permutation(N)
    n_val = max(1, int(val_ratio * N))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[val_idx], y[val_idx]

    log_xi_lo, log_xi_hi = np.log(xi_bounds[0]), np.log(xi_bounds[1])
    nu_lo, nu_hi = nu_bounds

    def obj(theta):
        log_xi, nu = theta
        # keep within bounds (optimizer respects bounds, but safety is fine)
        if not (log_xi_lo <= log_xi <= log_xi_hi and nu_lo <= nu <= nu_hi):
            return 1e30
        return krr_val_mse_for_params(Xtr, ytr, Xva, yva, log_xi, nu, reg=reg)

    bounds = [(log_xi_lo, log_xi_hi), (nu_lo, nu_hi)]

    best_val = np.inf
    best_theta = None

    for _ in range(n_restarts):
        # random init in bounds
        init = np.array([
            rng.uniform(log_xi_lo, log_xi_hi),
            rng.uniform(nu_lo, nu_hi),
        ])
        res = minimize(obj, init, method="Powell", bounds=bounds, options={"maxiter": 200})
        if res.fun < best_val:
            best_val = float(res.fun)
            best_theta = res.x

    best_log_xi, best_nu = best_theta
    best = {"xi": float(np.exp(best_log_xi)), "nu": float(best_nu), "reg": float(reg)}
    return best, best_val


def tune_matern_grid_train_val(
    X: np.ndarray,
    y: np.ndarray,
    *,
    val_ratio: float = 0.2,
    seed: int = 0,
    reg: float = 1e-6,
    xi_bounds: Tuple[float, float] = (1e-3, 1e3),
    nu_grid: Sequence[float] = (0.5, 1.5, 2.5, 5.0),
    xi_maxiter: int = 80,
) -> Tuple[Dict, float]:
    """
    Grid search over nu, and 1D bounded optimization over log(xi), to minimize
    validation MSE (single split). Returns: (best_params_dict, best_mse).

    This is typically faster + more stable than 2D Powell in (xi, nu).
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N,d). Got shape {X.shape}.")
    N = X.shape[0]
    if N < 2:
        raise ValueError("Need at least 2 samples to do a train/val split.")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)

    n_val = int(val_ratio * N)
    n_val = max(1, n_val)
    if n_val >= N:
        n_val = N - 1  # ensure non-empty train set

    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[val_idx], y[val_idx]

    if xi_bounds[0] <= 0 or xi_bounds[1] <= 0 or xi_bounds[0] >= xi_bounds[1]:
        raise ValueError(f"xi_bounds must be positive and increasing. Got {xi_bounds}.")

    log_xi_lo, log_xi_hi = float(np.log(xi_bounds[0])), float(np.log(xi_bounds[1]))

    best_mse = float("inf")
    best_log_xi = None
    best_nu = None

    for nu in nu_grid:
        nu = float(nu)

        res = minimize_scalar(
            lambda log_xi: krr_val_mse_for_params(
                Xtr, ytr, Xva, yva, float(log_xi), nu, reg=reg
            ),
            bounds=(log_xi_lo, log_xi_hi),
            method="bounded",
            options={"maxiter": int(xi_maxiter)},
        )

        if float(res.fun) < best_mse:
            best_mse = float(res.fun)
            best_log_xi = float(res.x)
            best_nu = nu

    best = {
        "xi": float(np.exp(best_log_xi)),
        "nu": float(best_nu),
        "reg": float(reg),
    }
    return best, best_mse