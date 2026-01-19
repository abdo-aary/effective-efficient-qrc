import numpy as np
import pytest

import src.models.qrc_matern_krr as mk


class DummyFeaturizer:
    """Deterministic featurizer: (N,w,d)->(N,D)"""
    def __init__(self, D=5):
        self.D = D
        self.calls = 0

    def transform(self, X):
        self.calls += 1
        X = np.asarray(X)
        N = X.shape[0]
        # Simple features from X
        m = X.mean(axis=(1, 2))
        s = X.std(axis=(1, 2))
        f = np.stack([m, s, m**2, np.sin(m), np.cos(m)], axis=1)
        return f[:, : self.D].astype(float)


def test_split_indices_deterministic_and_disjoint():
    tr1, te1 = mk._train_test_split_indices(20, test_ratio=0.2, seed=0)
    tr2, te2 = mk._train_test_split_indices(20, test_ratio=0.2, seed=0)
    assert np.array_equal(tr1, tr2)
    assert np.array_equal(te1, te2)
    assert len(set(tr1).intersection(set(te1))) == 0
    assert len(tr1) + len(te1) == 20


def test_build_kernel_diagonal_is_one():
    ker = mk._build_kernel(xi=1.23, nu=1.5)
    X = np.random.default_rng(0).normal(size=(6, 3))
    K = ker(X, X)
    assert K.shape == (6, 6)
    assert np.allclose(np.diag(K), 1.0, atol=1e-12)


def test_regressor_predict_raises_before_fit():
    model = mk.QRCMaternKRRRegressor(DummyFeaturizer(), tuning={"strategy": "grid"})
    with pytest.raises(ValueError):
        model.predict()  # no stored test set yet


def test_regressor_fit_shape_checks():
    model = mk.QRCMaternKRRRegressor(DummyFeaturizer(), tuning={"strategy": "grid"})
    with pytest.raises(ValueError):
        model.fit(np.zeros((10, 3)), np.zeros(10))  # X not 3D
    with pytest.raises(ValueError):
        model.fit(np.zeros((10, 2, 3)), np.zeros(9))  # y mismatched


def test_regressor_uses_grid_tuner(monkeypatch):
    # Patch tuner functions *in the module where they are imported*
    called = {"grid": 0, "powell": 0}

    def fake_grid(Phi, y, **kwargs):
        called["grid"] += 1
        return {"xi": 1.0, "nu": 1.5, "reg": kwargs["reg"]}, 0.123

    def fake_powell(*args, **kwargs):
        called["powell"] += 1
        raise AssertionError("Powell tuner should not be called in this test.")

    monkeypatch.setattr(mk, "tune_matern_grid_train_val", fake_grid)
    monkeypatch.setattr(mk, "tune_matern_continuous_train_val", fake_powell)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 4, 3))
    y = np.sin(X.mean(axis=(1, 2)))

    model = mk.QRCMaternKRRRegressor(
        DummyFeaturizer(),
        standardize=True,
        test_ratio=0.2,
        split_seed=0,
        tuning={"strategy": "grid", "val_ratio": 0.2, "seed": 0, "reg": 1e-6},
    )
    model.fit(X, y)

    assert called["grid"] == 1
    assert called["powell"] == 0
    assert model.best_params_["xi"] == 1.0
    assert model.best_params_["nu"] == 1.5
    assert model.scaler_ is not None
    assert model.alpha_ is not None
    assert model.X_train_features_ is not None


def test_regressor_uses_powell_tuner(monkeypatch):
    called = {"grid": 0, "powell": 0}

    def fake_grid(*args, **kwargs):
        called["grid"] += 1
        raise AssertionError("Grid tuner should not be called in this test.")

    def fake_powell(Phi, y, **kwargs):
        called["powell"] += 1
        return {"xi": 0.7, "nu": 2.5, "reg": kwargs["reg"]}, 0.456

    monkeypatch.setattr(mk, "tune_matern_grid_train_val", fake_grid)
    monkeypatch.setattr(mk, "tune_matern_continuous_train_val", fake_powell)

    rng = np.random.default_rng(1)
    X = rng.normal(size=(25, 3, 2))
    y = X.mean(axis=(1, 2))

    model = mk.QRCMaternKRRRegressor(
        DummyFeaturizer(),
        standardize=False,
        test_ratio=0.2,
        split_seed=0,
        tuning={"strategy": "powell", "val_ratio": 0.2, "seed": 0, "reg": 1e-6},
    )
    model.fit(X, y)

    assert called["powell"] == 1
    assert called["grid"] == 0
    assert model.scaler_ is None


def test_predict_none_matches_predict_on_stored_test_set(monkeypatch):
    # Make tuning deterministic and fast
    monkeypatch.setattr(
        mk,
        "tune_matern_grid_train_val",
        lambda Phi, y, **kwargs: ({"xi": 1.0, "nu": 1.5, "reg": kwargs["reg"]}, 0.0),
    )

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5, 3))
    y = np.sin(X.mean(axis=(1, 2)))

    model = mk.QRCMaternKRRRegressor(
        DummyFeaturizer(),
        standardize=True,
        test_ratio=0.25,
        split_seed=123,
        tuning={"strategy": "grid", "val_ratio": 0.2, "seed": 0, "reg": 1e-6},
    )
    model.fit(X, y)

    yhat_none = model.predict()
    assert yhat_none.shape == (len(model.y_test_),)

    # predict on stored test X should match predict(None) for deterministic featurizer
    yhat_X = model.predict(model.X_test_)
    assert np.allclose(yhat_none, yhat_X, atol=1e-12)
