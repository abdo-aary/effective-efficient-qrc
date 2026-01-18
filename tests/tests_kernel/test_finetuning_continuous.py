import numpy as np
import pytest

from src.kernel.finetuning import (
    tune_matern_continuous_train_val,
    krr_val_mse_for_params,
)


# ----------------------------- Helpers ---------------------------------

def make_smooth_regression_dataset(N=120, D=6, noise=0.05, seed=0):
    """
    Smooth target so Matérn with reasonable (xi, nu) should work well.
    Works for any D >= 1.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, D))

    y = np.sin(X[:, 0])
    if D >= 2:
        y = y + 0.5 * np.cos(0.7 * X[:, 1])
    if D >= 4:
        y = y + 0.25 * np.sin(0.3 * X[:, 2] * X[:, 3])
    elif D >= 3:
        # fallback interaction when only 3 dims available
        y = y + 0.25 * np.sin(0.3 * X[:, 1] * X[:, 2])

    y = y + noise * rng.normal(size=N)
    return X.astype(float), y.astype(float)


def train_val_split(X, y, val_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    n_val = max(1, int(val_ratio * X.shape[0]))
    va = idx[:n_val]
    tr = idx[n_val:]
    return X[tr], y[tr], X[va], y[va]


# ======================================================================
# 1) Smoke + bounds + types
# ======================================================================

def test_tune_returns_dict_and_float_and_respects_bounds():
    X, y = make_smooth_regression_dataset(N=80, D=4, noise=0.02, seed=1)

    xi_bounds = (1e-2, 1e2)
    nu_bounds = (0.3, 3.5)

    best, best_mse = tune_matern_continuous_train_val(
        X, y,
        val_ratio=0.25,
        seed=0,
        reg=1e-6,
        xi_bounds=xi_bounds,
        nu_bounds=nu_bounds,
        n_restarts=4,
    )

    assert isinstance(best, dict)
    assert isinstance(best_mse, float)
    assert "xi" in best and "nu" in best and "reg" in best

    assert xi_bounds[0] <= best["xi"] <= xi_bounds[1]
    assert nu_bounds[0] <= best["nu"] <= nu_bounds[1]
    assert best["reg"] == pytest.approx(1e-6)


# ======================================================================
# 2) Determinism under same seed
# ======================================================================

def test_tune_is_deterministic_given_seed():
    X, y = make_smooth_regression_dataset(N=90, D=5, noise=0.03, seed=2)

    best1, mse1 = tune_matern_continuous_train_val(
        X, y, seed=123, val_ratio=0.2, reg=1e-6, n_restarts=5
    )
    best2, mse2 = tune_matern_continuous_train_val(
        X, y, seed=123, val_ratio=0.2, reg=1e-6, n_restarts=5
    )

    assert best1["xi"] == pytest.approx(best2["xi"])
    assert best1["nu"] == pytest.approx(best2["nu"])
    assert mse1 == pytest.approx(mse2)


def test_tune_changes_with_different_seed_split_usually():
    # Not guaranteed mathematically, but "usually" should differ with different random split
    X, y = make_smooth_regression_dataset(N=120, D=6, noise=0.05, seed=3)

    best1, mse1 = tune_matern_continuous_train_val(
        X, y, seed=0, val_ratio=0.2, reg=1e-6, n_restarts=4
    )
    best2, mse2 = tune_matern_continuous_train_val(
        X, y, seed=1, val_ratio=0.2, reg=1e-6, n_restarts=4
    )

    # allow either params or mse to differ (split changes validation set)
    assert (abs(best1["xi"] - best2["xi"]) > 1e-9) or (abs(best1["nu"] - best2["nu"]) > 1e-9) or (abs(mse1 - mse2) > 1e-12)


# ======================================================================
# 3) Quantitative sanity: tuned MSE should beat a bad baseline
# ======================================================================

def test_tuned_params_improve_over_bad_params_on_validation():
    X, y = make_smooth_regression_dataset(N=140, D=6, noise=0.03, seed=4)
    Xtr, ytr, Xva, yva = train_val_split(X, y, val_ratio=0.25, seed=10)

    # Tune on same split by calling tune_matern_continuous_train_val with same seed
    # (it uses that seed for split permutation)
    best, best_mse = tune_matern_continuous_train_val(
        X, y, seed=10, val_ratio=0.25, reg=1e-6,
        xi_bounds=(1e-2, 1e2), nu_bounds=(0.3, 3.5),
        n_restarts=6,
    )

    # Evaluate the tuned choice explicitly
    tuned_mse = krr_val_mse_for_params(
        Xtr, ytr, Xva, yva,
        log_xi=np.log(best["xi"]),
        nu=best["nu"],
        reg=best["reg"]
    )

    # Build a "bad" setting: extremely large xi makes everything too similar (underfit),
    # and very small nu can make kernel very rough.
    bad_xi = 1e2
    bad_nu = 0.3
    bad_mse = krr_val_mse_for_params(
        Xtr, ytr, Xva, yva,
        log_xi=np.log(bad_xi),
        nu=bad_nu,
        reg=best["reg"],
    )

    assert tuned_mse == pytest.approx(best_mse, rel=1e-10, abs=1e-12)
    # tuned should be meaningfully better than bad
    assert tuned_mse < 0.7 * bad_mse


# ======================================================================
# 4) Edge cases: small N, extreme val_ratio
# ======================================================================

def test_small_dataset_does_not_crash():
    X, y = make_smooth_regression_dataset(N=8, D=3, noise=0.01, seed=5)

    best, best_mse = tune_matern_continuous_train_val(
        X, y, val_ratio=0.5, seed=0, reg=1e-6, n_restarts=2,
        xi_bounds=(1e-2, 1e2), nu_bounds=(0.3, 3.5),
    )

    assert np.isfinite(best_mse)
    assert best["xi"] > 0
    assert best["nu"] > 0


def test_val_ratio_near_1_still_works():
    X, y = make_smooth_regression_dataset(N=30, D=4, noise=0.02, seed=6)

    # With val_ratio very high, train set becomes small but should still run.
    best, best_mse = tune_matern_continuous_train_val(
        X, y, val_ratio=0.9, seed=0, reg=1e-6, n_restarts=2,
        xi_bounds=(1e-2, 1e2), nu_bounds=(0.3, 3.5),
    )

    assert np.isfinite(best_mse)
    assert best["xi"] > 0
    assert best["nu"] > 0
