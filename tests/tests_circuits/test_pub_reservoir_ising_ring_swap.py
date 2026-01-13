import numpy as np
import pytest

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, partial_trace

from src.configs import RingQRConfig as Config

from src.circuit_factory import CircuitFactory
from src.utils import angle_positioning_linear


# -------------------------
# Aer helpers
# -------------------------

qiskit_aer = pytest.importorskip("qiskit_aer")
from qiskit_aer import AerSimulator


def run_density_matrix(qc: QuantumCircuit, seed: int = 0) -> DensityMatrix:
    """Deterministic DM run (shots=1) for SWAP+reset circuits (no measurements)."""
    backend = AerSimulator(method="density_matrix")
    qc2 = qc.copy()
    qc2.save_density_matrix()
    tqc = transpile(qc2, backend)
    res = backend.run(tqc, shots=1, seed_simulator=seed).result()
    return DensityMatrix(res.data(0)["density_matrix"])


def reduced_reservoir(dm_full: DensityMatrix, n: int) -> DensityMatrix:
    """Keep reservoir qubits 0..n-1, trace out env n..2n (aux+coin)."""
    env = list(range(n, 2 * n + 1))
    return DensityMatrix(partial_trace(dm_full, env))


def build_full_unitary_reference_window(cfg, x_window: np.ndarray, qc_template: QuantumCircuit) -> QuantumCircuit:
    """
    Reference evolution over the full window WITHOUT contraction/env:
      init |+>^n then for each t: data-injection -> Ising unitary W
    Uses the same Parameter objects (J, h_x, h_z) as qc_template.metadata.
    """
    n = cfg.num_qubits
    topology = cfg.topology

    J  = qc_template.metadata["J"]
    hx = qc_template.metadata["h_x"]
    hz = qc_template.metadata["h_z"]

    qc_ref = QuantumCircuit(n, name="unitary_ref")
    qc_ref.h(list(range(n)))  # match instantiateFullIsingRingEvolution init

    for x_t in x_window:
        z_t = x_t @ cfg.projection  # shape (n,)
        angles = angle_positioning_linear(z_t)  # numeric list length n

        for i in range(n):
            qc_ref.ry(float(angles[i]), i)

        for e, (q1, q2) in enumerate(topology.edges):
            qc_ref.rzz(J[e], q1, q2)

        for i in range(n):
            qc_ref.rz(hz[i], i)
        for i in range(n):
            qc_ref.rx(hx[i], i)

    return qc_ref


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def cfg():
    return Config(input_dim=10, num_qubits=3, seed=12345)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def x_window(cfg, rng):
    # choose a small window length
    w = 4
    eps = 1e-8
    return rng.uniform(-1 + eps, 1 - eps, size=(w, cfg.input_dim))


# -------------------------
# Tests
# -------------------------

def test_pub_structure_and_shapes(cfg, x_window):
    lam_0 = 0.05
    R = 5
    pub = CircuitFactory.create_pub_reservoir_IsingRingSWAP(
        cfg=cfg,
        angle_positioning=angle_positioning_linear,
        x_window=x_window,
        lam_0=lam_0,
        num_reservoirs=R,
        seed=7,
    )
    assert isinstance(pub, tuple) and len(pub) == 2
    qc, values = pub
    assert isinstance(qc, QuantumCircuit)
    assert isinstance(values, np.ndarray)

    assert qc.num_qubits == 2 * cfg.num_qubits + 1
    assert values.shape[0] == R
    assert values.shape[1] == len(list(qc.parameters))

    # z must be fully bound by instantiateFullIsingRingEvolution => no "z" parameters remain
    assert all((p.name != "z") and (not p.name.startswith("z[")) for p in qc.parameters)


def test_pub_metadata_contains_param_order_and_lams(cfg, x_window):
    lam_0 = 0.05
    R = 4
    qc, values = CircuitFactory.create_pub_reservoir_IsingRingSWAP(
        cfg, angle_positioning_linear, x_window, lam_0=lam_0, num_reservoirs=R, seed=11
    )

    assert "param_order" in qc.metadata
    assert "lam_values" in qc.metadata

    param_order = qc.metadata["param_order"]
    lam_values = qc.metadata["lam_values"]

    assert isinstance(param_order, list)
    assert len(param_order) == len(list(qc.parameters))
    assert isinstance(lam_values, list) and len(lam_values) == R

    # first lambda is lam_0 exactly
    assert lam_values[0] == pytest.approx(lam_0)


def test_pub_lam_column_matches_metadata(cfg, x_window):
    lam_0 = 0.05
    R = 6
    eps = 1e-8
    qc, values = CircuitFactory.create_pub_reservoir_IsingRingSWAP(
        cfg, angle_positioning_linear, x_window, lam_0=lam_0, num_reservoirs=R, seed=22, eps=eps
    )

    lam_param = qc.metadata["lam"]
    param_order = qc.metadata["param_order"]
    lam_values = qc.metadata["lam_values"]
    lam_col = param_order.index(lam_param)

    # values column equals metadata lam_values
    assert np.allclose(values[:, lam_col], np.array(lam_values), atol=0.0, rtol=0.0)

    # check ranges for r>=1
    for r in range(1, R):
        assert eps < lam_values[r] < 1 - eps


def test_pub_is_deterministic_given_seed(cfg, x_window):
    lam_0 = 0.05
    R = 3

    pub1 = CircuitFactory.create_pub_reservoir_IsingRingSWAP(
        cfg, angle_positioning_linear, x_window, lam_0=lam_0, num_reservoirs=R, seed=123
    )
    pub2 = CircuitFactory.create_pub_reservoir_IsingRingSWAP(
        cfg, angle_positioning_linear, x_window, lam_0=lam_0, num_reservoirs=R, seed=123
    )

    qc1, v1 = pub1
    qc2, v2 = pub2

    # same seed => same sampled params
    assert np.array_equal(v1, v2)
    assert qc1.metadata["lam_values"] == qc2.metadata["lam_values"]


def test_pub_changes_with_different_seed(cfg, x_window):
    lam_0 = 0.05
    R = 3

    _, v1 = CircuitFactory.create_pub_reservoir_IsingRingSWAP(
        cfg, angle_positioning_linear, x_window, lam_0=lam_0, num_reservoirs=R, seed=1
    )
    _, v2 = CircuitFactory.create_pub_reservoir_IsingRingSWAP(
        cfg, angle_positioning_linear, x_window, lam_0=lam_0, num_reservoirs=R, seed=2
    )

    # probability of exact equality is essentially zero
    assert not np.array_equal(v1, v2)


def test_each_row_binds_to_fully_numeric_circuit(cfg, x_window):
    lam_0 = 0.05
    R = 4
    qc, values = CircuitFactory.create_pub_reservoir_IsingRingSWAP(
        cfg, angle_positioning_linear, x_window, lam_0=lam_0, num_reservoirs=R, seed=9
    )

    param_order = qc.metadata["param_order"]

    for r in range(R):
        bind = dict(zip(param_order, values[r]))
        qc_bound = qc.assign_parameters(bind, inplace=False)
        assert len(qc_bound.parameters) == 0


def test_aer_density_matrix_is_deterministic_shots1(cfg, x_window):
    lam_0 = 0.05
    R = 2
    qc, values = CircuitFactory.create_pub_reservoir_IsingRingSWAP(
        cfg, angle_positioning_linear, x_window, lam_0=lam_0, num_reservoirs=R, seed=9
    )
    param_order = qc.metadata["param_order"]

    bind = dict(zip(param_order, values[0]))
    qc_bound = qc.assign_parameters(bind, inplace=False)

    dm1 = run_density_matrix(qc_bound, seed=1)
    dm2 = run_density_matrix(qc_bound, seed=999)

    assert np.allclose(dm1.data, dm2.data, atol=1e-12, rtol=0.0)


def test_functional_check_lam_one_matches_unitary_reference(cfg, x_window):
    """
    Strong correctness test:
    If we override lambda=1, contraction should do nothing, so the reservoir
    reduced state must match the unitary-only reference evolution over the same window.
    """
    lam_0 = 0.05
    R = 1
    qc, values = CircuitFactory.create_pub_reservoir_IsingRingSWAP(
        cfg, angle_positioning_linear, x_window, lam_0=lam_0, num_reservoirs=R, seed=5
    )

    param_order = qc.metadata["param_order"]
    lam_param = qc.metadata["lam"]
    lam_col = param_order.index(lam_param)

    # override lambda to 1.0 (not allowed by the sampler in create_pub, but OK for a correctness test)
    values_override = values.copy()
    values_override[0, lam_col] = 1.0

    bind = dict(zip(param_order, values_override[0]))
    qc_bound = qc.assign_parameters(bind, inplace=False)

    dm_full = run_density_matrix(qc_bound, seed=0)
    dm_res = reduced_reservoir(dm_full, cfg.num_qubits)

    # Reference circuit: no ancillas/no contraction
    qc_ref = build_full_unitary_reference_window(cfg, x_window, qc_template=qc)
    bind_ref = {p: v for p, v in bind.items() if p != lam_param}
    dm_ref = run_density_matrix(qc_ref.assign_parameters(bind_ref, inplace=False), seed=0)

    assert np.allclose(dm_res.data, dm_ref.data, atol=1e-10, rtol=0.0)
