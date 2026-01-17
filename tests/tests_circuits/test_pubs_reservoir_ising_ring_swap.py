import numpy as np
import pytest

from qiskit import transpile
from qiskit.quantum_info import DensityMatrix, partial_trace

# Config import (adapt if your class name differs)
from src.qrc.circuits.configs import RingQRConfig
from src.qrc.circuits.circuit_factory import CircuitFactory
from src.qrc.circuits.utils import angle_positioning_linear


# -------------------------
# Aer helpers (skip if Aer not installed)
# -------------------------
qiskit_aer = pytest.importorskip("qiskit_aer")
from qiskit_aer import AerSimulator


def run_dm(qc, seed=0):
    backend = AerSimulator(method="density_matrix")
    qc2 = qc.copy()
    qc2.save_density_matrix()
    tqc = transpile(qc2, backend)
    res = backend.run(tqc, shots=1, seed_simulator=seed).result()
    return DensityMatrix(res.data(0)["density_matrix"])


def reduce_reservoir(dm_full: DensityMatrix, n: int) -> DensityMatrix:
    # keep 0..n-1, trace out env n..2n (aux + coin)
    env = list(range(n, 2 * n + 1))
    return DensityMatrix(partial_trace(dm_full, env))


def expected_param_order_from_step(cfg) -> list:
    """Canonical expected order for reservoir parameterization vectors."""
    qc_step = CircuitFactory.createIsingRingCircuitSWAP(cfg, angle_positioning_linear)
    J = list(qc_step.metadata["J"])
    hx = list(qc_step.metadata["h_x"])
    hz = list(qc_step.metadata["h_z"])
    lam = qc_step.metadata["lam"]
    return J + hx + hz + [lam]


def recompute_lam_values(lam_0: float, R: int, seed: int, eps: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # consume draws in same order as set_reservoirs_parameterizationSWAP:
    # rzz, rx, rz first, then lam
    # (sizes depend on cfg; we won't reproduce those draws here)
    # Instead, we recompute lambdas exactly as in the function:
    lam_values = np.empty(R, dtype=float)
    lam_values[0] = lam_0
    if R > 1:
        lam_values[1:] = rng.uniform(eps, 1 - eps, size=R - 1)
    return lam_values


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def cfg():
    return RingQRConfig(input_dim=10, num_qubits=3, seed=12345)


@pytest.fixture
def X(cfg):
    # small dataset: N windows, window length w, input_dim d
    rng = np.random.default_rng(0)
    N, w = 3, 4
    eps = 1e-8
    return rng.uniform(-1 + eps, 1 - eps, size=(N, w, cfg.input_dim))


# -------------------------
# Tests: set_reservoirs_parameterizationSWAP
# -------------------------

def test_set_param_values_shape_matches_expected_order(cfg):
    R = 5
    lam_0 = 0.07
    seed = 11
    eps = 1e-8

    param_values = CircuitFactory.set_reservoirs_parameterizationSWAP(
        cfg=cfg,
        angle_positioning=angle_positioning_linear,
        num_reservoirs=R,
        lam_0=lam_0,
        seed=seed,
        eps=eps,
    )

    expected_order = expected_param_order_from_step(cfg)
    assert param_values.shape == (R, len(expected_order))


def test_set_param_values_deterministic_same_seed(cfg):
    R = 4
    lam_0 = 0.05
    seed = 123
    eps = 1e-8

    v1 = CircuitFactory.set_reservoirs_parameterizationSWAP(cfg, angle_positioning_linear, R, lam_0, seed, eps)
    v2 = CircuitFactory.set_reservoirs_parameterizationSWAP(cfg, angle_positioning_linear, R, lam_0, seed, eps)
    assert np.array_equal(v1, v2)


def test_set_param_values_changes_with_seed(cfg):
    R = 4
    lam_0 = 0.05
    eps = 1e-8

    v1 = CircuitFactory.set_reservoirs_parameterizationSWAP(cfg, angle_positioning_linear, R, lam_0, seed=1, eps=eps)
    v2 = CircuitFactory.set_reservoirs_parameterizationSWAP(cfg, angle_positioning_linear, R, lam_0, seed=2, eps=eps)
    assert not np.array_equal(v1, v2)


# -------------------------
# Tests: create_pub_reservoirs_IsingRingSWAP
# -------------------------

def test_create_pub_has_matching_param_count(cfg, X):
    lam_0 = 0.05
    R = 3
    seed = 7
    eps = 1e-8

    param_values = CircuitFactory.set_reservoirs_parameterizationSWAP(
        cfg, angle_positioning_linear, num_reservoirs=R, lam_0=lam_0, seed=seed, eps=eps
    )

    x_window = X[0]
    qc, vals = CircuitFactory.create_pub_reservoirs_IsingRingSWAP(
        cfg=cfg,
        angle_positioning=angle_positioning_linear,
        x_window=x_window,
        num_reservoirs=R,
        param_values=param_values,
    )

    assert vals.shape[0] == R
    assert vals.shape[1] == len(list(qc.parameters))  # must match pub contract

    # ensure z is bound inside window circuit
    assert all((p.name != "z") and (not p.name.startswith("z[")) for p in qc.parameters)


def test_create_pub_lam_column_matches_expected(cfg, X):
    lam_0 = 0.05
    R = 5
    seed = 9
    eps = 1e-6

    param_values = CircuitFactory.set_reservoirs_parameterizationSWAP(
        cfg, angle_positioning_linear, R, lam_0, seed=seed, eps=eps
    )

    qc, vals = CircuitFactory.create_pub_reservoirs_IsingRingSWAP(
        cfg, angle_positioning_linear, x_window=X[0], num_reservoirs=R, param_values=param_values
    )

    lam_param = qc.metadata["lam"]
    lam_idx = list(qc.parameters).index(lam_param)

    assert vals[0, lam_idx] == pytest.approx(lam_0)
    for r in range(1, R):
        assert eps < vals[r, lam_idx] < 1 - eps


def test_each_row_binds_to_numeric_and_runs_aer(cfg, X):
    lam_0 = 0.05
    R = 3
    seed = 5
    eps = 1e-8

    param_values = CircuitFactory.set_reservoirs_parameterizationSWAP(
        cfg, angle_positioning_linear, R, lam_0, seed=seed, eps=eps
    )

    qc, vals = CircuitFactory.create_pub_reservoirs_IsingRingSWAP(
        cfg, angle_positioning_linear, x_window=X[0], num_reservoirs=R, param_values=param_values
    )

    n = cfg.num_qubits
    for r in range(R):
        bind = dict(zip(list(qc.parameters), vals[r]))
        qc_bound = qc.assign_parameters(bind, inplace=False)
        assert len(qc_bound.parameters) == 0

        dm_full = run_dm(qc_bound, seed=0)
        dm_res = reduce_reservoir(dm_full, n)
        # basic sanity: density matrix shape 2^n x 2^n
        assert dm_res.data.shape == (2**n, 2**n)


# -------------------------
# Tests: create_pubs_dataset_reservoirs_IsingRingSWAP
# -------------------------

def test_create_pubs_dataset_returns_N_pubs(cfg, X):
    lam_0 = 0.05
    R = 4

    pubs = CircuitFactory.create_pubs_dataset_reservoirs_IsingRingSWAP(
        cfg=cfg,
        angle_positioning=angle_positioning_linear,
        X=X,
        lam_0=lam_0,
        num_reservoirs=R,
        seed=123,
        eps=1e-8,
    )

    assert isinstance(pubs, list)
    assert len(pubs) == X.shape[0]

    # all pubs share the same parameter matrix (values-wise)
    base_vals = pubs[0][1]
    for qc, vals in pubs:
        assert vals.shape == base_vals.shape
        assert np.array_equal(vals, base_vals)
        assert vals.shape[1] == len(list(qc.parameters))
