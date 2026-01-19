import numpy as np
import pytest

import src.models.kernel as ft
from scipy.optimize import minimize


def _make_smooth_dataset(N=80, noise=0.03, seed=0):
    """
    1D smooth function with a characteristic lengthscale.
    Kernel methods should fit this well and produce stable tuning behavior.
    """
    rng = np.random.default_rng(seed)
    X = np.linspace(0.0, 1.0, N).reshape(-1, 1)
    y_clean = np.sin(6 * np.pi * X[:, 0]) + 0.3 * np.cos(2 * np.pi * X[:, 0])
    y = y_clean + noise * rng.standard_normal(N)
    return X.astype(float), y.astype(float)


def _train_val_split_indices(N, val_ratio, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_val = max(1, int(val_ratio * N))
    if n_val >= N:
        n_val = N - 1
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return tr_idx, val_idx


def _recompute_val_mse(X, y, params, *, val_ratio, seed):
    tr_idx, val_idx = _train_val_split_indices(X.shape[0], val_ratio, seed)
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[val_idx], y[val_idx]
    log_xi = float(np.log(params["xi"]))
    nu = float(params["nu"])
    reg = float(params["reg"])
    return ft.krr_val_mse_for_params(Xtr, ytr, Xva, yva, log_xi, nu, reg=reg)


# -------------------------
# Correctness / validation
# -------------------------

def test_grid_returns_expected_keys_and_ranges():
    X, y = _make_smooth_dataset(N=60, seed=1)

    xi_bounds = (1e-3, 1e3)
    nu_grid = (0.5, 1.5, 2.5, 5.0)
    reg = 1e-3

    best, best_mse = ft.tune_matern_grid_train_val(
        X, y,
        val_ratio=0.2,
        seed=123,
        reg=reg,
        xi_bounds=xi_bounds,
        nu_grid=nu_grid,
        xi_maxiter=30,
    )

    assert set(best.keys()) == {"xi", "nu", "reg"}
    assert xi_bounds[0] <= best["xi"] <= xi_bounds[1]
    assert best["nu"] in [float(v) for v in nu_grid]
    assert best["reg"] == float(reg)
    assert np.isfinite(best_mse) and best_mse >= 0.0


def test_grid_reported_mse_matches_recompute_on_same_split():
    X, y = _make_smooth_dataset(N=70, seed=2)

    best, best_mse = ft.tune_matern_grid_train_val(
        X, y,
        val_ratio=0.25,
        seed=7,
        reg=1e-3,
        xi_bounds=(1e-3, 1e3),
        nu_grid=(0.5, 1.5, 2.5),
        xi_maxiter=25,
    )

    mse_re = _recompute_val_mse(X, y, best, val_ratio=0.25, seed=7)
    assert np.isclose(best_mse, mse_re, rtol=1e-10, atol=1e-12)


def test_grid_is_deterministic_given_seed():
    X, y = _make_smooth_dataset(N=60, seed=3)

    kwargs = dict(
        val_ratio=0.2,
        seed=42,
        reg=1e-3,
        xi_bounds=(1e-3, 1e3),
        nu_grid=(0.5, 1.5, 2.5, 5.0),
        xi_maxiter=25,
    )

    best1, mse1 = ft.tune_matern_grid_train_val(X, y, **kwargs)
    best2, mse2 = ft.tune_matern_grid_train_val(X, y, **kwargs)

    assert best1 == best2
    assert mse1 == pytest.approx(mse2, rel=0.0, abs=0.0)


def test_grid_raises_on_bad_shapes_and_bounds():
    X, y = _make_smooth_dataset(N=20, seed=0)

    with pytest.raises(ValueError):
        ft.tune_matern_grid_train_val(X.squeeze(), y)  # X not 2D

    with pytest.raises(ValueError):
        ft.tune_matern_grid_train_val(X[:1], y[:1])  # N < 2

    with pytest.raises(ValueError):
        ft.tune_matern_grid_train_val(X, y, xi_bounds=(0.0, 1.0))  # non-positive

    with pytest.raises(ValueError):
        ft.tune_matern_grid_train_val(X, y, xi_bounds=(1.0, 1.0))  # not increasing

    # Extreme val_ratio should still work (internally clamps to keep train non-empty)
    best, mse = ft.tune_matern_grid_train_val(X, y, val_ratio=0.99, seed=0, reg=1e-3)
    assert np.isfinite(mse)
    assert best["xi"] > 0.0


# ------------------------------------------------------
# Comparison: grid vs grid+refine vs Powell multi-restart
# ------------------------------------------------------

def test_quality_and_efficiency_grid_refine_vs_powell(monkeypatch):
    X, y = _make_smooth_dataset(N=80, seed=10)

    # Count how many times the expensive objective is evaluated.
    counts = {"mode": None, "grid": 0, "refine": 0, "powell": 0}
    orig = ft.krr_val_mse_for_params

    def counted(*args, **kwargs):
        counts[counts["mode"]] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(ft, "krr_val_mse_for_params", counted)

    # Shared setup
    val_ratio = 0.2
    seed = 123
    reg = 1e-3
    xi_bounds = (1e-3, 1e3)
    nu_grid = (0.5, 1.5, 2.5, 5.0)

    # ---- (1) Grid method
    counts["mode"] = "grid"
    best_grid, mse_grid = ft.tune_matern_grid_train_val(
        X, y,
        val_ratio=val_ratio,
        seed=seed,
        reg=reg,
        xi_bounds=xi_bounds,
        nu_grid=nu_grid,
        xi_maxiter=25,
    )

    # ---- (2) Grid -> continuous refinement (L-BFGS-B) on same split
    # Rebuild same split as in the tuners to ensure apples-to-apples comparison.
    tr_idx, val_idx = _train_val_split_indices(X.shape[0], val_ratio, seed)
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[val_idx], y[val_idx]

    log_xi_lo, log_xi_hi = float(np.log(xi_bounds[0])), float(np.log(xi_bounds[1]))
    nu_lo, nu_hi = 0.2, 5.0  # match ft.tune_matern_continuous_train_val default

    def obj(theta):
        log_xi, nu = float(theta[0]), float(theta[1])
        return ft.krr_val_mse_for_params(Xtr, ytr, Xva, yva, log_xi, nu, reg=reg)

    x0 = np.array([np.log(best_grid["xi"]), best_grid["nu"]], dtype=float)
    bounds = [(log_xi_lo, log_xi_hi), (nu_lo, nu_hi)]

    counts["mode"] = "refine"
    res = minimize(
        obj, x0,
        method="L-BFGS-B",
        jac="2-point",               # finite-diff gradient (stable, usually few evals)
        bounds=bounds,
        options={"maxiter": 60},
    )
    best_ref = {
        "xi": float(np.exp(res.x[0])),
        "nu": float(res.x[1]),
        "reg": float(reg),
    }
    mse_ref = float(res.fun)

    # ---- (3) Powell multi-restart (your original continuous tuner)
    counts["mode"] = "powell"
    best_pow, mse_pow = ft.tune_matern_continuous_train_val(
        X, y,
        val_ratio=val_ratio,
        seed=seed,
        reg=reg,
        xi_bounds=xi_bounds,
        nu_bounds=(0.2, 5.0),
        n_restarts=3,  # keep tests fast; still representative
    )

    # ------------------------
    # Assertions: correctness
    # ------------------------
    # Refinement should never be worse than its starting point by much.
    # (Small numerical wiggle room)
    assert mse_ref <= mse_grid + 1e-10

    # Solutions should be in-bounds
    assert xi_bounds[0] <= best_ref["xi"] <= xi_bounds[1]
    assert 0.2 <= best_ref["nu"] <= 5.0
    assert xi_bounds[0] <= best_pow["xi"] <= xi_bounds[1]
    assert 0.2 <= best_pow["nu"] <= 5.0

    # ------------------------
    # Assertions: quality
    # ------------------------
    # Don’t make this flaky: require that grid+refine is competitive with Powell.
    # We scale tolerance by variance of y (problem-dependent).
    y_var = float(np.var(y))
    tol = 0.05 * y_var + 1e-12  # 5% of variance
    assert mse_ref <= mse_pow + tol

    # ------------------------
    # Assertions: efficiency proxy
    # ------------------------
    # Grid should typically use fewer objective calls than Powell multi-restart.
    assert counts["grid"] < counts["powell"]

    # And refinement should not explode objective calls relative to Powell.
    assert counts["refine"] < counts["powell"]
