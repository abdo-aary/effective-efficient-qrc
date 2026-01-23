import numpy as np
import pytest

from src.data.generate.varma import VARMAGenerator


class DummyLabel:
    def __call__(self, X_win: np.ndarray, *, rng: np.random.Generator) -> float:
        return float(X_win[-1].sum())


def spectral_norm(A: np.ndarray) -> float:
    return float(np.linalg.svd(A, compute_uv=False)[0])


def sigma_min(A: np.ndarray) -> float:
    return float(np.linalg.svd(A, compute_uv=False)[-1])


def test_varma_init_validates_params():
    with pytest.raises(ValueError):
        VARMAGenerator(p=0, q=0, s=0)
    with pytest.raises(ValueError):
        VARMAGenerator(p=1, q=-1, s=0)
    with pytest.raises(ValueError):
        VARMAGenerator(p=1, q=0, gamma=1.0, s=0)
    with pytest.raises(ValueError):
        VARMAGenerator(p=1, q=0, gamma=0.0, s=0)
    with pytest.raises(ValueError):
        VARMAGenerator(p=1, q=0, rho=1.0, s=0)
    with pytest.raises(ValueError):
        VARMAGenerator(p=1, q=0, rho=0.0, s=0)
    with pytest.raises(ValueError):
        VARMAGenerator(p=1, q=0, innovation_scale=0.0, s=0)
    with pytest.raises(ValueError):
        VARMAGenerator(p=1, q=0, innovation="student_t", student_df=2.0, s=0)


def test_varma_draws_phi_with_budget_sum_norms_equals_gamma():
    gen = VARMAGenerator(p=3, q=2, gamma=0.6, eta=0.7, rho=0.5, s=0, seed=0, regenerate_params_each_call=True)
    rng = np.random.default_rng(0)
    _ = gen.simulate_path(T=50, d=4, rng=rng)

    assert gen.Phi_ is not None
    Phi = gen.Phi_
    assert len(Phi) == 3
    norms = [spectral_norm(M) for M in Phi]
    s_norm = sum(norms)

    # construction aims at sum ||Phi_i||_2 = gamma (up to floating error)
    assert s_norm <= gen.gamma + 1e-10
    assert s_norm == pytest.approx(gen.gamma, rel=1e-8, abs=1e-10)


def test_varma_stability_implies_invertibility_bound_on_unit_circle():
    """
    We validate the *sufficient stability condition* used in the paper:
      ||A(z)||_2 <= sum_i ||Phi_i||_2 <= gamma  for |z|=1,
      => sigma_min(I - A(z)) >= 1 - ||A(z)||_2 >= 1 - gamma.
    """
    gamma = 0.75
    gen = VARMAGenerator(p=4, q=1, gamma=gamma, eta=0.3, rho=0.8, s=0, seed=123, regenerate_params_each_call=True)
    rng = np.random.default_rng(123)
    _ = gen.simulate_path(T=60, d=3, rng=rng)

    Phi = gen.Phi_
    assert Phi is not None

    d = 3
    I = np.eye(d, dtype=complex)

    # Sample a few points on the unit circle
    for theta in np.linspace(0.0, 2 * np.pi, 9, endpoint=False):
        z = np.exp(1j * theta)
        Az = np.zeros((d, d), dtype=complex)
        for i, Phi_i in enumerate(Phi, start=1):
            Az += Phi_i.astype(complex) * (z**i)

        norm_Az = spectral_norm(Az)
        assert norm_Az <= gamma + 1e-8

        smin = sigma_min(I - Az)
        assert smin >= (1.0 - gamma) - 1e-8


def test_varma_theta_decay_norms_match_eta_rho_power():
    eta = 0.8
    rho = 0.6
    q = 5
    gen = VARMAGenerator(p=2, q=q, gamma=0.5, eta=eta, rho=rho, s=0, seed=0, regenerate_params_each_call=True)
    rng = np.random.default_rng(0)
    _ = gen.simulate_path(T=40, d=4, rng=rng)

    Theta = gen.Theta_
    assert Theta is not None
    assert len(Theta) == q

    for j, Th_j in enumerate(Theta, start=1):
        expected = eta * (rho ** (j - 1))
        assert spectral_norm(Th_j) == pytest.approx(expected, rel=1e-7, abs=1e-10)


def test_varma_caching_behavior_regenerate_flag():
    # regenerate_params_each_call=False -> Phi_ stays identical across calls
    gen = VARMAGenerator(p=2, q=1, gamma=0.6, eta=0.2, rho=0.5, s=0, seed=0, regenerate_params_each_call=False)
    rng = np.random.default_rng(0)

    _ = gen.simulate_path(T=30, d=3, rng=rng)
    Phi1 = [M.copy() for M in gen.Phi_]
    Theta1 = [M.copy() for M in gen.Theta_]

    _ = gen.simulate_path(T=30, d=3, rng=rng)
    Phi2 = gen.Phi_
    Theta2 = gen.Theta_

    assert all(np.allclose(a, b) for a, b in zip(Phi1, Phi2))
    assert all(np.allclose(a, b) for a, b in zip(Theta1, Theta2))

    # regenerate_params_each_call=True -> Phi_ typically changes (almost surely)
    gen2 = VARMAGenerator(p=2, q=1, gamma=0.6, eta=0.2, rho=0.5, s=0, seed=0, regenerate_params_each_call=True)
    rng2 = np.random.default_rng(0)
    _ = gen2.simulate_path(T=30, d=3, rng=rng2)
    PhiA = [M.copy() for M in gen2.Phi_]
    _ = gen2.simulate_path(T=30, d=3, rng=rng2)
    PhiB = gen2.Phi_

    assert not all(np.allclose(a, b) for a, b in zip(PhiA, PhiB))


def test_varma_user_provided_matrices_used_and_validated():
    d = 2
    p, q = 2, 1

    Phi_user = [
        np.array([[0.2, 0.0], [0.0, 0.1]]),
        np.array([[0.1, 0.0], [0.0, 0.1]]),
    ]
    Theta_user = [
        np.array([[0.3, 0.0], [0.0, 0.2]]),
    ]

    gen = VARMAGenerator(
        p=p, q=q, gamma=0.9, eta=0.9, rho=0.5,  # ignored for user-provided matrices
        Phi=Phi_user, Theta=Theta_user,
        s=0, seed=0, regenerate_params_each_call=True
    )
    rng = np.random.default_rng(0)
    Z = gen.simulate_path(T=20, d=d, rng=rng)

    assert Z.shape == (20, d)
    assert gen.Phi_ is not None and gen.Theta_ is not None
    assert all(np.allclose(a, b) for a, b in zip(gen.Phi_, Phi_user))
    assert all(np.allclose(a, b) for a, b in zip(gen.Theta_, Theta_user))

    # wrong length -> error
    with pytest.raises(ValueError):
        VARMAGenerator(p=2, q=1, Phi=[np.eye(2)], Theta=Theta_user, s=0)

    # wrong shape -> error (must trigger simulate_path because d is known there)
    with pytest.raises(ValueError):
        gen_bad = VARMAGenerator(p=2, q=1, Phi=[np.eye(3), np.eye(3)], Theta=Theta_user, s=0)
        gen_bad.simulate_path(T=10, d=2, rng=np.random.default_rng(0))


def test_varma_make_windows_dataset_smoke_shapes_and_bounded():
    gen = VARMAGenerator(p=2, q=1, gamma=0.7, eta=0.4, rho=0.7, s=5, burn_in=50, bounded="tanh", seed=1)
    ds = gen.make_windows_dataset(N=20, w=12, d=3, label_functionals=[DummyLabel()])

    assert ds.X.shape == (20, 12, 3)
    assert ds.y.shape == (1, 20,)
    assert np.all(ds.X > -1.0) and np.all(ds.X < 1.0)  # tanh-bounded
    assert ds.meta["s"] == 5
    assert ds.meta["stride"] == 12 + 5
