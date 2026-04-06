import numpy as np
import pytest

from src.compute.backend import cupy_available
from src.models.qrc_matern_krr import QRCMaternKRRRegressor


class _FlattenFeaturizer:
    def transform(self, X):
        return np.asarray(X, dtype=float).reshape(X.shape[0], -1)


@pytest.mark.skipif(not cupy_available(), reason="CuPy/CUDA unavailable")
def test_qrc_matern_gpu_readout_matches_cpu_readout():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(16, 3, 2))
    y = X[:, -1, 0] + 0.1 * X[:, :, 1].sum(axis=1)
    tuning = {
        "strategy": "grid",
        "val_ratio": 0.25,
        "seed": 7,
        "reg": 1e-3,
        "xi_bounds": (0.2, 5.0),
        "nu_grid": [1.5],
        "xi_maxiter": 10,
    }

    cpu = QRCMaternKRRRegressor(
        _FlattenFeaturizer(),
        test_ratio=0.25,
        split_seed=2,
        tuning=tuning,
        readout_backend="numpy",
    ).fit(X, y)
    gpu = QRCMaternKRRRegressor(
        _FlattenFeaturizer(),
        test_ratio=0.25,
        split_seed=2,
        tuning=tuning,
        readout_backend="cupy",
        readout_device=0,
    ).fit(X, y)

    np.testing.assert_allclose(gpu.predict(), cpu.predict(), atol=1e-8, rtol=1e-8)
    assert gpu.best_params_["nu"] == pytest.approx(cpu.best_params_["nu"])
