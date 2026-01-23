import numpy as np
import pytest

from src.data.label.functionals import OneStepForecastFunctional, ExpFadingLinearFunctional, VolteraFunctional
from src.data.label.base import LabelNoise


def _fading_projection_ref(X_win: np.ndarray, vec: np.ndarray, alpha: float) -> float:
    """Reference: sum_{k=0}^{w-1} alpha^k <vec, x_{t-k}>, with x_t = X_win[-1]."""
    w = X_win.shape[0]
    z = 0.0
    for k in range(w):
        x_tk = X_win[-1 - k]
        z += (alpha**k) * float(np.dot(vec, x_tk))
    return float(z)


def test_one_step_forecast_matches_formula_exact():
    X_win = np.array(
        [
            [1.0, 2.0, 3.0],
            [0.5, -1.0, 0.0],
            [2.0, 0.0, -2.0],
            [1.5, 1.0, 0.5],
        ],
        dtype=float,
    )  # w=4, d=3

    # choose a non-unit vector; implementation normalizes to unit
    u = np.array([2.0, 0.0, 0.0])
    u_unit = u / np.linalg.norm(u)

    f = OneStepForecastFunctional(u=u, nonlinearity="none", noise=LabelNoise(0.0))
    rng = np.random.default_rng(123)
    y = f(X_win, rng)

    expected = float(np.dot(u_unit, X_win[-1]))
    assert y == pytest.approx(expected, rel=0, abs=1e-12)


def test_exp_fading_linear_matches_formula_exact():
    alpha = 0.7
    X_win = np.array(
        [
            [1.0, 2.0],
            [-1.0, 0.5],
            [0.0, 1.0],
        ],
        dtype=float,
    )  # w=3, d=2

    u = np.array([1.0, 2.0])
    u_unit = u / np.linalg.norm(u)

    f = ExpFadingLinearFunctional(alpha=alpha, u=u, nonlinearity="none", noise=LabelNoise(0.0))
    rng = np.random.default_rng(0)
    y = f(X_win, rng)

    expected = _fading_projection_ref(X_win, u_unit, alpha)
    assert y == pytest.approx(expected, rel=0, abs=1e-12)


def test_volterra_matches_formula_exact_with_orthogonalization():
    alpha = 0.9
    X_win = np.array(
        [
            [1.0, 0.0, -1.0],
            [0.0, 2.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, -1.0, 0.5],
        ],
        dtype=float,
    )  # w=4, d=3

    u = np.array([1.0, 0.0, 1.0])
    v = np.array([1.0, 2.0, 3.0])  # not orthogonal to u

    u_unit = u / np.linalg.norm(u)
    v_orth = v - float(np.dot(u_unit, v)) * u_unit
    v_unit = v_orth / np.linalg.norm(v_orth)

    f = VolteraFunctional(alpha=alpha, u=u, v=v, orthogonalize_v=True, nonlinearity="none", noise=LabelNoise(0.0))
    rng = np.random.default_rng(0)
    y = f(X_win, rng)

    Lu = _fading_projection_ref(X_win, u_unit, alpha)
    Lv = _fading_projection_ref(X_win, v_unit, alpha)
    expected = float(Lu + 0.5 * (Lv**2))

    assert y == pytest.approx(expected, rel=0, abs=1e-12)


@pytest.mark.parametrize("kind", ["tanh", "sigmoid"])
def test_nonlinearity_applied_after_core_formula(kind):
    X_win = np.array([[0.0, 0.0], [1.0, -1.0]], dtype=float)  # w=2,d=2
    u = np.array([1.0, 0.0])
    u_unit = u / np.linalg.norm(u)

    f = OneStepForecastFunctional(u=u, nonlinearity=kind, noise=LabelNoise(0.0))
    rng = np.random.default_rng(0)
    y = f(X_win, rng)

    z = float(np.dot(u_unit, X_win[-1]))
    if kind == "tanh":
        expected = float(np.tanh(z))
    else:
        expected = float(1.0 / (1.0 + np.exp(-z)))

    assert y == pytest.approx(expected, rel=0, abs=1e-12)


def test_noise_added_last_and_reproducible():
    X_win = np.array([[0.0], [2.0]], dtype=float)
    u = np.array([1.0])
    noise = LabelNoise(0.1)

    f = OneStepForecastFunctional(u=u, nonlinearity="none", noise=noise)

    rng1 = np.random.default_rng(123)
    y1 = f(X_win, rng1)

    # reproduce the exact same noise draw using a fresh RNG with same seed
    rng2 = np.random.default_rng(123)
    core = float(X_win[-1, 0])  # u is already unit after normalization
    expected = core + rng2.normal(loc=0.0, scale=0.1)

    assert y1 == pytest.approx(expected, rel=0, abs=1e-12)


def test_input_validation_bad_alpha_and_shapes():
    X_win = np.zeros((3, 2), dtype=float)
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError):
        ExpFadingLinearFunctional(alpha=1.0)(X_win, rng)
    with pytest.raises(ValueError):
        ExpFadingLinearFunctional(alpha=-0.2)(X_win, rng)

    with pytest.raises(ValueError):
        OneStepForecastFunctional(u=np.ones(3))(X_win, rng)  # wrong d

    with pytest.raises(ValueError):
        OneStepForecastFunctional()(np.zeros((3, 2, 1)), rng)  # wrong ndim
