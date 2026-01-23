import numpy as np

from src.data.generate.beta_mixing import BetaMixingGenerator
from src.data.label.functionals import (
    OneStepForecastFunctional,
    ExpFadingLinearFunctional,
    VolteraFunctional,
)
from src.data.label.base import LabelNoise


class DeterministicBetaMixingGen(BetaMixingGenerator):
    """
    Deterministic path so the integration test is not flaky.
    path[t, k] = scale * (k+1) * t
    """

    def __init__(self, *, scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = float(scale)

    def simulate_path(self, *, T: int, d: int, rng: np.random.Generator) -> np.ndarray:
        t = np.arange(T, dtype=float)  # (T,)
        path = np.stack([(k + 1) * t for k in range(d)], axis=1)  # (T,d)
        return self.scale * path


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    return v / np.linalg.norm(v)


def _fading_projection_ref(X_win: np.ndarray, vec: np.ndarray, alpha: float) -> float:
    """
    Reference for Appendix E.2:
      sum_{k=0}^{w-1} alpha^k <vec, x_{t-k}>
    with x_t = X_win[-1].
    """
    w = X_win.shape[0]
    s = 0.0
    for k in range(w):
        s += (alpha ** k) * float(np.dot(vec, X_win[-1 - k]))
    return float(s)


def test_make_windows_dataset_three_functionals_matches_exact_formulas_no_bounding():
    N, w, d = 9, 6, 4
    s = 2
    burn_in = 3

    gen = DeterministicBetaMixingGen(
        scale=1.0,
        s=s,
        burn_in=burn_in,
        bounded="none",
        seed=123,
    )

    # Choose non-unit vectors on purpose (implementations normalize internally)
    u1 = np.array([2.0, 0.0, 0.0, 0.0])
    u2 = np.array([1.0, -2.0, 0.5, 0.0])

    # For Volterra, choose u3 and v3 not orthogonal to ensure we test orthogonalize_v=True path.
    u3 = np.array([1.0, 1.0, 0.0, 0.0])
    v3 = np.array([2.0, 0.0, 1.0, 0.0])

    alpha2 = 0.7
    alpha3 = 0.85

    f1 = OneStepForecastFunctional(u=u1, nonlinearity="none", noise=LabelNoise(0.0))
    f2 = ExpFadingLinearFunctional(alpha=alpha2, u=u2, nonlinearity="none", noise=LabelNoise(0.0))
    f3 = VolteraFunctional(alpha=alpha3, u=u3, v=v3, orthogonalize_v=True,
                           nonlinearity="none", noise=LabelNoise(0.0))

    ds = gen.make_windows_dataset(N=N, w=w, d=d, label_functionals=[f1, f2, f3])

    assert ds.X.shape == (N, w, d)
    assert ds.y.shape == (3, N)
    assert ds.meta["s"] == s
    assert ds.meta["burn_in"] == burn_in

    # Build exact expected labels from formulas in Appendix E.2
    u1u = _unit(u1)
    u2u = _unit(u2)

    u3u = _unit(u3)
    # orthogonalize v against u (as in VolteraFunctional when orthogonalize_v=True)
    v3u = _unit(v3)
    v3_orth = v3u - float(np.dot(u3u, v3u)) * u3u
    v3o = _unit(v3_orth)

    Y_expected = np.zeros((3, N), dtype=float)
    for i in range(N):
        X_win = ds.X[i]

        # (F1) one-step forecast under window convention: y = <u, X_win[-1]>
        Y_expected[0, i] = float(np.dot(u1u, X_win[-1]))

        # (F2) exponentially fading linear functional
        Y_expected[1, i] = _fading_projection_ref(X_win, u2u, alpha2)

        # (F3) Volterra (rank-one quadratic term): L_u + 0.5 * (L_v)^2
        Lu = _fading_projection_ref(X_win, u3u, alpha3)
        Lv = _fading_projection_ref(X_win, v3o, alpha3)
        Y_expected[2, i] = float(Lu + 0.5 * (Lv ** 2))

    diff = np.max(np.abs(ds.y - Y_expected))
    assert diff < 1e-10, f"max |ds.y - Y_expected| = {diff:.3e}"


def test_make_windows_dataset_three_functionals_matches_exact_formulas_with_tanh_bounding():
    """
    Same integration test, but checks that bounding happens BEFORE labeling,
    since labels are computed from ds.X windows (already bounded by tanh).
    """
    N, w, d = 7, 5, 3
    gen = DeterministicBetaMixingGen(
        scale=0.05,  # keep values small (avoid saturation), but any is fine
        s=1,
        burn_in=4,
        bounded="tanh",
        seed=0,
    )

    u1 = np.array([1.0, 2.0, 3.0])
    u2 = np.array([1.0, 0.0, -1.0])
    u3 = np.array([1.0, 1.0, 0.0])
    v3 = np.array([1.0, 0.0, 1.0])

    alpha2 = 0.6
    alpha3 = 0.9

    f1 = OneStepForecastFunctional(u=u1, nonlinearity="none", noise=LabelNoise(0.0))
    f2 = ExpFadingLinearFunctional(alpha=alpha2, u=u2, nonlinearity="none", noise=LabelNoise(0.0))
    f3 = VolteraFunctional(alpha=alpha3, u=u3, v=v3, orthogonalize_v=True,
                           nonlinearity="none", noise=LabelNoise(0.0))

    ds = gen.make_windows_dataset(N=N, w=w, d=d, label_functionals=[f1, f2, f3])
    assert ds.y.shape == (3, N)

    # Verify that X is bounded (tanh)
    assert np.all(ds.X > -1.0) and np.all(ds.X < 1.0)

    # Expected labels computed on the bounded windows ds.X (this is what make_windows_dataset uses)
    u1u, u2u, u3u = _unit(u1), _unit(u2), _unit(u3)
    v3u = _unit(v3)
    v3_orth = v3u - float(np.dot(u3u, v3u)) * u3u
    v3o = _unit(v3_orth)

    Y_expected = np.zeros((3, N), dtype=float)
    for i in range(N):
        X_win = ds.X[i]
        Y_expected[0, i] = float(np.dot(u1u, X_win[-1]))
        Y_expected[1, i] = _fading_projection_ref(X_win, u2u, alpha2)
        Lu = _fading_projection_ref(X_win, u3u, alpha3)
        Lv = _fading_projection_ref(X_win, v3o, alpha3)
        Y_expected[2, i] = float(Lu + 0.5 * (Lv ** 2))

    assert np.allclose(ds.y, Y_expected, atol=1e-12, rtol=0.0)
