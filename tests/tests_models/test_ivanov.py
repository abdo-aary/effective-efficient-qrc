from __future__ import annotations

import numpy as np

from src.models.ivanov import (
    fit_rms_matern_ivanov,
    predict_rms_matern_ivanov,
    rms_matern_kernel,
)


def test_rms_matern_distance_is_invariant_to_feature_duplication():
    X = np.asarray([[0.0, 1.0], [1.0, -1.0], [0.5, 0.25]])
    duplicated = np.repeat(X, 3, axis=1)
    np.testing.assert_allclose(
        rms_matern_kernel(X, X, nu=2.5, xi=1.2),
        rms_matern_kernel(duplicated, duplicated, nu=2.5, xi=1.2),
        atol=2e-15,
        rtol=0.0,
    )


def test_ivanov_inactive_constraint_is_minimum_norm_pseudoinverse():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(8, 3))
    Y = rng.normal(size=(8, 2))
    fit = fit_rms_matern_ivanov(
        X,
        Y,
        task_names=("a", "b"),
        nu=2.5,
        xi=1.0,
        radius=1e6,
        tolerance=1e-11,
    )
    gram = rms_matern_kernel(X, X, nu=2.5, xi=1.0)
    expected = np.linalg.pinv(gram, rcond=1e-12) @ Y
    np.testing.assert_allclose(fit.alphas, expected, atol=1e-8, rtol=1e-8)
    assert all(not row["active_constraint"] for row in fit.diagnostics)
    assert all(row["stationarity_residual"] < 1e-9 for row in fit.diagnostics)


def test_ivanov_active_constraint_satisfies_kkt_for_multioutput():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(10, 4))
    Y = rng.normal(size=(10, 2))
    radius = 0.2
    fit = fit_rms_matern_ivanov(
        X,
        Y,
        task_names=("left", "right"),
        nu=2.5,
        xi=0.8,
        radius=radius,
        tolerance=1e-11,
    )
    assert all(row["active_constraint"] for row in fit.diagnostics)
    for row in fit.diagnostics:
        assert abs(row["rkhs_norm"] - radius) < 2e-9
        assert row["lagrange_multiplier"] > 0.0
        assert row["stationarity_residual"] < 1e-9
        assert row["primal_feasibility_residual"] < 1e-10
        assert row["complementarity_residual"] < 1e-9

    predictions = predict_rms_matern_ivanov(
        X, X, fit, nu=2.5, xi=0.8
    )
    assert predictions.shape == Y.shape
    assert np.all(np.isfinite(predictions))


def test_singular_gram_uses_minimum_norm_tie_without_jitter():
    X = np.asarray([[0.0], [0.0], [1.0], [1.0]])
    Y = np.asarray([[1.0], [1.0], [-1.0], [-1.0]])
    fit = fit_rms_matern_ivanov(
        X,
        Y,
        task_names=("task",),
        nu=2.5,
        xi=1.0,
        radius=100.0,
        tolerance=1e-11,
    )
    threshold = 1e-12 * max(1.0, float(np.max(fit.eigenvalues)))
    assert np.count_nonzero(fit.eigenvalues > threshold) == 2
    prediction = predict_rms_matern_ivanov(X, X, fit, nu=2.5, xi=1.0)
    np.testing.assert_allclose(prediction, Y, atol=1e-10, rtol=0.0)
    assert fit.diagnostics[0]["active_constraint"] is False
