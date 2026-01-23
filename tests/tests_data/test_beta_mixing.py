import numpy as np
import pytest

from src.data.generate.beta_mixing import BetaMixingGenerator
from src.data.generate.base import ensure_bounded


class DummyLabel:
    def __call__(self, X_win: np.ndarray, *, rng: np.random.Generator) -> float:
        # deterministic label: sum of last time step
        return float(X_win[-1].sum())


class DummyBetaMixingGen(BetaMixingGenerator):
    """Deterministic path generator for testing window indexing."""
    def simulate_path(self, *, T: int, d: int, rng: np.random.Generator) -> np.ndarray:
        # path[t,k] = t  (same across dims) -> easy to validate window slices
        t = np.arange(T, dtype=float).reshape(-1, 1)
        return np.repeat(t, d, axis=1)


def test_ensure_bounded_modes():
    X = np.array([[-2.0, 0.0, 2.0]])
    assert np.allclose(ensure_bounded(X, mode="none"), X)
    assert np.all(ensure_bounded(X, mode="tanh") > -1.0) and np.all(ensure_bounded(X, mode="tanh") < 1.0)
    clipped = ensure_bounded(X, mode="clip", clip=0.5)
    assert np.all(clipped <= 0.5) and np.all(clipped >= -0.5)

    with pytest.raises(ValueError):
        ensure_bounded(X, mode="unknown")


def test_make_windows_dataset_indexing_and_stride_no_bounding():
    N, w, d = 4, 5, 2
    s = 3
    burn_in = 10
    gen = DummyBetaMixingGen(s=s, burn_in=burn_in, bounded="none", seed=123)

    ds = gen.make_windows_dataset(N=N, w=w, d=d, label_functionals=[DummyLabel()])

    assert ds.X.shape == (N, w, d)
    assert ds.y.shape == (1, N,)
    assert ds.meta["stride"] == w + s
    assert ds.meta["burn_in"] == burn_in
    assert ds.meta["s"] == s
    assert ds.meta["bounded"] == "none"

    stride = w + s
    # window i should be path[start:end+1] with end = burn_in + (w-1) + i*stride
    for i in range(N):
        end = burn_in + (w - 1) + i * stride
        start = end - w + 1
        # Since path[t,k] = t, the window should equal [start,...,end] replicated on d dims
        expected = np.arange(start, end + 1, dtype=float).reshape(-1, 1)
        expected = np.repeat(expected, d, axis=1)

        assert np.allclose(ds.X[i], expected)
        # y is sum of last step across dims: end * d
        assert ds.y[0,i] == pytest.approx(float(end * d), abs=0.0)


def test_make_windows_dataset_applies_bounding():
    N, w, d = 3, 4, 1
    gen = DummyBetaMixingGen(s=0, burn_in=0, bounded="tanh", seed=0)
    ds = gen.make_windows_dataset(N=N, w=w, d=d, label_functionals=[DummyLabel()])
    assert np.all(ds.X > -1.0) and np.all(ds.X < 1.0)


def test_make_windows_dataset_validates_inputs():
    gen = DummyBetaMixingGen(s=1, burn_in=0, bounded="none", seed=0)
    with pytest.raises(ValueError):
        gen.make_windows_dataset(N=0, w=3, d=1, label_functionals=[DummyLabel()])
    with pytest.raises(ValueError):
        gen.make_windows_dataset(N=3, w=0, d=1, label_functionals=[DummyLabel()])
    with pytest.raises(ValueError):
        gen.make_windows_dataset(N=3, w=3, d=0, label_functionals=[DummyLabel()])

    gen_bad_s = DummyBetaMixingGen(s=-1, burn_in=0, bounded="none", seed=0)
    with pytest.raises(ValueError):
        gen_bad_s.make_windows_dataset(N=3, w=3, d=1, label_functionals=[DummyLabel()])

    gen_bad_burn = DummyBetaMixingGen(s=0, burn_in=-5, bounded="none", seed=0)
    with pytest.raises(ValueError):
        gen_bad_burn.make_windows_dataset(N=3, w=3, d=1, label_functionals=[DummyLabel()])


def test_make_windows_dataset_checks_simulate_shape():
    class BadShapeGen(DummyBetaMixingGen):
        def simulate_path(self, *, T: int, d: int, rng: np.random.Generator) -> np.ndarray:
            return np.zeros((T + 1, d))  # wrong

    gen = BadShapeGen(s=0, burn_in=0, bounded="none", seed=0)
    with pytest.raises(ValueError):
        gen.make_windows_dataset(N=3, w=3, d=2, label_functionals=[DummyLabel()])
