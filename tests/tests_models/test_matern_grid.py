from __future__ import annotations

import numpy as np
from sklearn.gaussian_process.kernels import Matern

from src.experiment.protocol import ChronologicalSplit
from src.models.matern_grid import (
    FiniteGridMaternKRR,
    matern_from_sq_dists,
    pairwise_sq_dists,
)


def test_protocol_matern_matches_sklearn_including_nu_five():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(9, 4))
    d2 = pairwise_sq_dists(X, X, xp=np)
    for nu in (1.5, 2.5, 5.0):
        actual = matern_from_sq_dists(d2, nu=nu, xi=1.3, xp=np)
        expected = Matern(length_scale=1.3, nu=nu)(X)
        np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)


def test_selection_is_validation_only_and_refits_outer_train():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(30, 5))
    Y = np.column_stack((np.sin(X[:, 0]), X[:, 1] * X[:, 2]))
    split = ChronologicalSplit(20, 25, 30)
    model = FiniteGridMaternKRR(
        nu_grid=(1.5, 5.0), xi_grid=(0.5, 1.0), lambda_grid=(1e-3, 1e-1), backend="numpy"
    )
    first = model.fit_select_refit(features=X, targets=Y, task_names=("a", "b"), split=split)
    changed = Y.copy()
    changed[split.test] += 1000.0
    second = model.fit_select_refit(
        features=X, targets=changed, task_names=("a", "b"), split=split
    )
    assert first.selected == second.selected
    assert len(first.candidate_rows) == 2 * model.candidate_count
    assert first.predictions.shape == Y.shape
    assert first.alphas.shape == (25, 2)
    assert all(row["eta"] == row["lambda_K"] / 25 for row in first.metric_rows)
