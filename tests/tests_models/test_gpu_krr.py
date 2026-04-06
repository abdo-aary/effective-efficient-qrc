import numpy as np
import pytest
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import Matern as SkMatern

from src.compute.backend import cupy_available, resolve_backend
from src.models.gpu_krr import krr_lambda_sweep, matern_kernel_matrix, ridge_fit_predict, solve_krr


def test_backend_selection_prefers_cupy_when_available():
    assert resolve_backend("numpy") == "numpy"
    if cupy_available():
        assert resolve_backend("auto") == "cupy"
    else:
        assert resolve_backend("auto") == "numpy"


@pytest.mark.skipif(not cupy_available(), reason="CuPy/CUDA unavailable")
@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5])
def test_cupy_matern_matches_sklearn(nu):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(7, 4))
    Y = rng.normal(size=(5, 4))
    xi = 1.7

    got = matern_kernel_matrix(X, Y, xi=xi, nu=nu, backend="cupy")
    expected = ConstantKernel(1.0, constant_value_bounds="fixed") * SkMatern(
        length_scale=xi,
        length_scale_bounds="fixed",
        nu=nu,
    )

    np.testing.assert_allclose(got.get(), expected(X, Y), atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(not cupy_available(), reason="CuPy/CUDA unavailable")
def test_cupy_krr_solve_and_lambda_sweep_match_numpy():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 3))
    y = rng.normal(size=8)
    Xv = rng.normal(size=(4, 3))
    yv = rng.normal(size=4)
    K = matern_kernel_matrix(X, X, xi=2.3, nu=1.5, backend="numpy")
    Kv = matern_kernel_matrix(Xv, X, xi=2.3, nu=1.5, backend="numpy")
    reg = 0.07

    alpha_gpu = solve_krr(K, y, reg=reg, backend="cupy").get()
    alpha_np = np.linalg.solve(K + reg * np.eye(K.shape[0]), y)
    np.testing.assert_allclose(alpha_gpu, alpha_np, atol=1e-10, rtol=1e-10)

    sweep_gpu = krr_lambda_sweep(K, Kv, y, yv, [1e-3, 1e-1, 1e1], backend="cupy")
    for i, lam in enumerate([1e-3, 1e-1, 1e1]):
        alpha = np.linalg.solve(K + lam * np.eye(K.shape[0]), y)
        expected = np.mean((Kv @ alpha - yv) ** 2)
        assert sweep_gpu["eval_mse"][i] == pytest.approx(expected, rel=1e-10, abs=1e-10)


@pytest.mark.skipif(not cupy_available(), reason="CuPy/CUDA unavailable")
def test_cupy_ridge_predict_matches_numpy_dual_path():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(6, 20))
    y = rng.normal(size=6)
    Xv = rng.normal(size=(3, 20))
    alpha = 0.4

    got = ridge_fit_predict(X, y, Xv, alpha=alpha, backend="cupy")
    expected = ridge_fit_predict(X, y, Xv, alpha=alpha, backend="numpy")
    np.testing.assert_allclose(got, expected, atol=1e-10, rtol=1e-10)
