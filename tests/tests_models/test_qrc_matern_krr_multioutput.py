import multiprocessing as pymp

import numpy as np
import pytest

import src.models.qrc_matern_krr as mk


class CountingFeaturizer:
    """Deterministic featurizer with a call counter.

    Notes
    -----
    Used in tests to ensure that for multi-output training, featurization is
    performed only once and reused across label functionals.
    """

    def __init__(self, D: int = 5):
        self.D = int(D)
        self.calls = 0

    def transform(self, X: np.ndarray) -> np.ndarray:
        self.calls += 1
        X = np.asarray(X)
        m = X.mean(axis=(1, 2))
        s = X.std(axis=(1, 2))
        f = np.stack([m, s, m**2, np.sin(m), np.cos(m)], axis=1)
        return f[:, : self.D].astype(float)


def _make_multioutput_dataset(
    rng: np.random.Generator,
    *,
    N: int = 48,
    w: int = 6,
    d: int = 3,
    L: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a small multi-output regression problem.

    Returns
    -------
    X:
        Windows of shape (N, w, d).
    Y:
        Labels of shape (L, N).
    """
    X = rng.normal(size=(N, w, d)).astype(float)
    z = X.mean(axis=(1, 2))

    outs = [
        np.sin(2.0 * z),
        np.cos(1.5 * z),
        (z**2) - 0.1 * z,
    ]
    Y = np.stack(outs[:L], axis=0)
    return X, Y


def _deterministic_grid_tuner_factory(counter: dict | None = None):
    """Fast deterministic tuner for tests."""

    def fake_grid(Phi: np.ndarray, y: np.ndarray, **kwargs):
        if counter is not None:
            counter["calls"] = counter.get("calls", 0) + 1

        reg = float(kwargs.get("reg", 1e-6))

        # Make xi output-dependent to exercise per-output best_params_ list.
        xi = 0.5 + abs(float(np.mean(y)))
        nu = 1.5

        return {"xi": xi, "nu": nu, "reg": reg}, 0.0

    return fake_grid


def test_multioutput_fit_shapes_and_single_featurization_call(monkeypatch):
    rng = np.random.default_rng(0)
    X, Y = _make_multioutput_dataset(rng, N=60, w=5, d=3, L=3)

    featurizer = CountingFeaturizer(D=5)

    called = {"calls": 0}
    monkeypatch.setattr(mk, "tune_matern_grid_train_val", _deterministic_grid_tuner_factory(called), raising=True)

    # Ensure we don't accidentally route to the continuous tuner
    monkeypatch.setattr(
        mk,
        "tune_matern_continuous_train_val",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("continuous tuner should not be called")),
        raising=True,
    )

    model = mk.QRCMaternKRRRegressor(
        featurizer,
        standardize=True,
        test_ratio=0.2,
        split_seed=123,
        tuning={"strategy": "grid", "val_ratio": 0.25, "seed": 0, "reg": 1e-6, "nu_grid": (1.5,), "xi_maxiter": 5},
    )

    model.fit(X, Y, num_workers=1)

    # Featurization done once (critical: reuse Phi for all outputs)
    assert featurizer.calls == 1

    # Tuning executed once per output
    assert called["calls"] == 3

    # Learned attributes: multi-output case
    assert model.n_outputs_ == 3
    assert isinstance(model.best_params_, list) and len(model.best_params_) == 3
    assert isinstance(model.kernel_, list) and len(model.kernel_) == 3
    assert model.alpha_ is not None and model.alpha_.ndim == 2

    n_train = model.X_train_features_.shape[0]
    assert model.alpha_.shape == (3, n_train)

    # predict() on stored test set returns (L, n_test)
    yhat = model.predict()
    assert model.y_test_ is not None
    assert yhat.shape == model.y_test_.shape
    assert yhat.shape[0] == 3
    assert np.all(np.isfinite(yhat))


def test_multioutput_accepts_N_by_L_layout(monkeypatch):
    rng = np.random.default_rng(1)
    X, Y = _make_multioutput_dataset(rng, N=50, w=4, d=2, L=2)

    featurizer = CountingFeaturizer(D=4)

    monkeypatch.setattr(mk, "tune_matern_grid_train_val", _deterministic_grid_tuner_factory(), raising=True)

    model = mk.QRCMaternKRRRegressor(
        featurizer,
        standardize=False,
        test_ratio=0.3,
        split_seed=0,
        tuning={"strategy": "grid", "val_ratio": 0.2, "seed": 0, "reg": 1e-6, "nu_grid": (1.5,), "xi_maxiter": 3},
    )

    # pass y as (N,L) instead of (L,N)
    model.fit(X, Y.T, num_workers=1)

    assert model.n_outputs_ == 2
    yhat = model.predict()
    assert yhat.shape == model.y_test_.shape == (2, model.X_test_.shape[0])


def test_multioutput_predict_none_matches_predict_on_stored_test_X(monkeypatch):
    rng = np.random.default_rng(2)
    X, Y = _make_multioutput_dataset(rng, N=64, w=5, d=3, L=3)

    featurizer = CountingFeaturizer(D=5)

    monkeypatch.setattr(mk, "tune_matern_grid_train_val", _deterministic_grid_tuner_factory(), raising=True)

    model = mk.QRCMaternKRRRegressor(
        featurizer,
        standardize=True,
        test_ratio=0.25,
        split_seed=999,
        tuning={"strategy": "grid", "val_ratio": 0.2, "seed": 0, "reg": 1e-6, "nu_grid": (1.5,), "xi_maxiter": 4},
    )

    model.fit(X, Y, num_workers=1)
    calls_after_fit = featurizer.calls

    # predict(None): should NOT re-featurize; returns cached test predictions from fit
    yhat_none = model.predict()
    assert featurizer.calls == calls_after_fit

    # predict(X_test): should featurize once more and match yhat_none (deterministic featurizer)
    yhat_X = model.predict(model.X_test_)
    assert featurizer.calls == calls_after_fit + 1

    assert np.allclose(yhat_none, yhat_X, atol=1e-12)


def test_multioutput_uses_pool_when_num_workers_gt1(monkeypatch):
    """Unit-test the multiprocessing branch without spawning real processes."""

    rng = np.random.default_rng(3)
    X, Y = _make_multioutput_dataset(rng, N=40, w=4, d=3, L=3)

    featurizer = CountingFeaturizer(D=5)
    monkeypatch.setattr(mk, "tune_matern_grid_train_val", _deterministic_grid_tuner_factory(), raising=True)

    # --- fake multiprocessing context to validate wiring
    class FakePool:
        def __init__(self, processes, initializer=None, initargs=()):
            self.processes = processes
            self.initializer = initializer
            self.initargs = initargs
            self.map_called = 0
            self.items = None
            if initializer is not None:
                initializer(*initargs)

        def map(self, func, iterable):
            self.map_called += 1
            self.items = list(iterable)
            return [func(item) for item in self.items]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeContext:
        def __init__(self):
            self.pool = None

        def Pool(self, processes, initializer=None, initargs=()):
            self.pool = FakePool(processes, initializer=initializer, initargs=initargs)
            return self.pool

    ctx = FakeContext()

    # Force the code to take the multiprocessing branch
    monkeypatch.setattr(mk.mp, "get_context", lambda *args, **kwargs: ctx, raising=True)

    model = mk.QRCMaternKRRRegressor(
        featurizer,
        standardize=True,
        test_ratio=0.2,
        split_seed=0,
        tuning={"strategy": "grid", "val_ratio": 0.2, "seed": 0, "reg": 1e-6, "nu_grid": (1.5,), "xi_maxiter": 3},
    )

    model.fit(X, Y, num_workers=2)

    assert ctx.pool is not None
    assert ctx.pool.processes == 2
    assert ctx.pool.map_called == 1
    assert ctx.pool.items is not None and len(ctx.pool.items) == 3

    # initializer received the train features matrix (shared across outputs)
    phi_tr_from_init = ctx.pool.initargs[0]
    assert isinstance(phi_tr_from_init, np.ndarray)
    assert model.X_train_features_ is not None
    assert np.allclose(phi_tr_from_init, model.X_train_features_, atol=0.0, rtol=0.0)


def test_multioutput_real_multiprocessing_smoke_fork(monkeypatch):
    """Smoke-test using a real process pool.

    Notes
    -----
    This is only run when the ``fork`` start method is available (typical on Linux).
    We patch the tuning function *before* pool creation so workers inherit it under fork.
    """

    try:
        pymp.get_context("fork")
    except ValueError:
        pytest.skip("multiprocessing start method 'fork' is not available")

    rng = np.random.default_rng(4)
    X, Y = _make_multioutput_dataset(rng, N=36, w=4, d=2, L=2)

    featurizer = CountingFeaturizer(D=4)

    monkeypatch.setattr(mk, "tune_matern_grid_train_val", _deterministic_grid_tuner_factory(), raising=True)

    model = mk.QRCMaternKRRRegressor(
        featurizer,
        standardize=True,
        test_ratio=0.25,
        split_seed=7,
        tuning={"strategy": "grid", "val_ratio": 0.2, "seed": 0, "reg": 1e-6, "nu_grid": (1.5,), "xi_maxiter": 3},
    )

    model.fit(X, Y, num_workers=2)
    yhat = model.predict()

    assert yhat.shape == model.y_test_.shape == (2, model.X_test_.shape[0])
    assert np.all(np.isfinite(yhat))
