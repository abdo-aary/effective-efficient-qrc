# tests/test_exact_aer_runner_run_pubs_dm.py
import numpy as np
import pytest

pytest.importorskip("qiskit_aer")

from qiskit_aer import AerSimulator
from qiskit import transpile


from src.qrc.circuits.circuit_factory import CircuitFactory
from src.qrc.circuits.utils import angle_positioning_linear
from src.qrc.run.circuit_run import ExactAerCircuitsRunner, ExactResults

from src.qrc.circuits.configs import RingQRConfig


# ----------------------------
# helpers
# ----------------------------
def make_random_X(N: int, w: int, d: int, seed: int = 0, eps: float = 1e-8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1 + eps, 1 - eps, size=(N, w, d))


def get_create_pubs_fn():
    # your codebase sometimes uses "...reservoirs..." vs "...reservoir..."
    fn = getattr(CircuitFactory, "create_pubs_dataset_reservoirs_IsingRingSWAP", None)
    if fn is None:
        fn = getattr(CircuitFactory, "create_pubs_dataset_reservoir_IsingRingSWAP", None)
    if fn is None:
        raise AttributeError("Cannot find create_pubs_dataset_reservoir(s)_IsingRingSWAP on CircuitFactory.")
    return fn


def direct_aer_reduced_dm_from_row(qc, row, n_res: int, seed_simulator: int = 0, optimization_level: int = 1):
    """
    Independent reference:
      1) bind parameters via qc.assign_parameters (no parameter_binds)
      2) append save_density_matrix(qubits=0..n_res-1)
      3) run Aer density-matrix
      4) extract reduced density matrix
    """
    backend = AerSimulator(method="density_matrix")

    bind = dict(zip(list(qc.parameters), row))
    qc_bound = qc.assign_parameters(bind, inplace=False)

    qc_work = qc_bound.copy()
    save_qubits = list(range(n_res))

    # Always append a save on reservoir qubits; try a custom label if supported
    try:
        qc_work.save_density_matrix(qubits=save_qubits, label="dm_res")
        key = "dm_res"
    except TypeError:
        qc_work.save_density_matrix(qubits=save_qubits)
        key = "density_matrix"

    tqc = transpile(qc_work, backend=backend, optimization_level=optimization_level)
    result = backend.run(tqc, shots=1, seed_simulator=seed_simulator).result()
    data0 = result.data(0)

    if key in data0:
        dm = data0[key]
    else:
        # fallback: find a density-matrix key
        dm = None
        for k, v in data0.items():
            if "density_matrix" in k:
                dm = v
                break
        if dm is None:
            raise KeyError(f"No density_matrix found in result.data(0). Keys={list(data0.keys())}")

    return np.asarray(dm.data if hasattr(dm, "data") else dm, dtype=complex)


def assert_is_valid_density_matrix(dm: np.ndarray, atol: float = 1e-10):
    # Hermitian
    assert np.allclose(dm, dm.conj().T, atol=atol, rtol=0.0)
    # trace 1
    tr = np.trace(dm)
    assert np.allclose(tr, 1.0 + 0.0j, atol=atol, rtol=0.0)
    # PSD up to numerical tolerance
    evals = np.linalg.eigvalsh(dm)
    assert np.min(evals) >= -5e-9


# ----------------------------
# fixtures
# ----------------------------
@pytest.fixture
def cfg():
    # keep this small-ish for CI speed
    # adjust to your config signature if needed
    return RingQRConfig(input_dim=10, num_qubits=3, seed=12345)


@pytest.fixture
def X(cfg):
    N, w, d = 3, 2, cfg.input_dim
    return make_random_X(N=N, w=w, d=d, seed=777, eps=1e-8)


@pytest.fixture
def pubs(cfg, X):
    create_pubs = get_create_pubs_fn()
    pubs = create_pubs(
        cfg=cfg,
        angle_positioning=angle_positioning_linear,
        X=X,
        lam_0=0.05,
        num_reservoirs=3,
        seed=999,
        eps=1e-8,
    )
    return pubs


# ----------------------------
# tests: qualitative
# ----------------------------
def test_run_pubs_returns_exactresults_and_shapes(cfg, pubs):
    runner = ExactAerCircuitsRunner(cfg)

    res = runner.run_pubs(pubs=pubs, seed_simulator=0, optimization_level=1)

    assert isinstance(res, ExactResults)
    assert isinstance(res.states, np.ndarray)

    N = len(pubs)
    R = pubs[0][1].shape[0]
    n = cfg.num_qubits
    dim = 2 ** n

    assert res.states.shape == (N, R, dim, dim)

    # validate a couple density matrices
    assert_is_valid_density_matrix(res.states[0, 0])
    assert_is_valid_density_matrix(res.states[-1, -1])


def test_run_pubs_is_deterministic_given_seed(cfg, pubs):
    runner = ExactAerCircuitsRunner(cfg)

    res1 = runner.run_pubs(pubs=pubs, seed_simulator=0, optimization_level=1)
    res2 = runner.run_pubs(pubs=pubs, seed_simulator=0, optimization_level=1)

    assert np.allclose(res1.states, res2.states, atol=0.0, rtol=0.0)


def test_run_pubs_raises_on_param_matrix_column_mismatch(cfg, pubs):
    runner = ExactAerCircuitsRunner(cfg)

    qc0, vals0 = pubs[0]
    bad_vals0 = vals0[:, :-1]  # drop one column => mismatch with qc.parameters length
    bad_pubs = [(qc0, bad_vals0)] + pubs[1:]

    with pytest.raises(ValueError, match="param cols mismatch"):
        runner.run_pubs(pubs=bad_pubs, seed_simulator=0, optimization_level=1)


# ----------------------------
# tests: quantitative (matches independent Aer reference)
# ----------------------------
def test_run_pubs_matches_direct_aer_reference(cfg, pubs):
    """
    Quantitative check:
      runner.run_pubs(...) must match an independent direct Aer run where we:
        - bind params with assign_parameters
        - append save_density_matrix(qubits=reservoir)
        - run Aer
    """
    runner = ExactAerCircuitsRunner(cfg)
    out = runner.run_pubs(pubs=pubs, seed_simulator=0, optimization_level=1)

    n_res = cfg.num_qubits

    # check a small subset for runtime reasons
    # (window 0, all reservoirs) + (last window, reservoir 0)
    indices = [(0, 0), (0, 1), (0, 2), (len(pubs) - 1, 0)]

    for i, r in indices:
        qc_i, vals_i = pubs[i]
        row = vals_i[r]

        dm_expected = direct_aer_reduced_dm_from_row(
            qc=qc_i,
            row=row,
            n_res=n_res,
            seed_simulator=0,
            optimization_level=1,
        )
        dm_got = out.states[i, r]

        assert np.allclose(dm_got, dm_expected, atol=1e-10, rtol=0.0)
        assert_is_valid_density_matrix(dm_got)
