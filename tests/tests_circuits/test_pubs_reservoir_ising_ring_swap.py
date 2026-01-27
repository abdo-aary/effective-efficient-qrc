import numpy as np
import pytest

from qiskit import transpile
from qiskit.quantum_info import DensityMatrix, partial_trace

# Config import (adapt if your class name differs)
from src.qrc.circuits.qrc_configs import RingQRConfig
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


def expected_param_order_from_step(qrc_cfg) -> list:
    """Canonical expected order for reservoir parameterization vectors."""
    qc_step = CircuitFactory.createIsingRingCircuitSWAP(qrc_cfg, angle_positioning_linear)
    J = list(qc_step.metadata["J"])
    hx = list(qc_step.metadata["h_x"])
    hz = list(qc_step.metadata["h_z"])
    lam = qc_step.metadata["lam"]
    return J + hx + hz + [lam]


def recompute_lam_values(lam_0: float, R: int, seed: int, eps: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # consume draws in same order as set_reservoirs_parameterizationSWAP:
    # rzz, rx, rz first, then lam
    # (sizes depend on qrc_cfg; we won't reproduce those draws here)
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
def qrc_cfg():
    return RingQRConfig(input_dim=10, num_qubits=3, seed=12345)


@pytest.fixture
def X(qrc_cfg):
    # small dataset: N windows, window length w, input_dim d
    rng = np.random.default_rng(0)
    N, w = 3, 4
    eps = 1e-8
    return rng.uniform(-1 + eps, 1 - eps, size=(N, w, qrc_cfg.input_dim))


# -------------------------
# Tests: set_reservoirs_parameterizationSWAP
# -------------------------

def test_set_param_values_shape_matches_expected_order(qrc_cfg):
    R = 5
    lam_0 = 0.07
    seed = 11
    eps = 1e-8

    param_values = CircuitFactory.set_reservoirs_parameterizationSWAP(
        qrc_cfg=qrc_cfg,
        angle_positioning=angle_positioning_linear,
        num_reservoirs=R,
        lam_0=lam_0,
        seed=seed,
        eps=eps,
    )

    expected_order = expected_param_order_from_step(qrc_cfg)
    assert param_values.shape == (R, len(expected_order))


def test_set_param_values_deterministic_same_seed(qrc_cfg):
    R = 4
    lam_0 = 0.05
    seed = 123
    eps = 1e-8

    v1 = CircuitFactory.set_reservoirs_parameterizationSWAP(qrc_cfg, angle_positioning_linear, R, lam_0, seed, eps)
    v2 = CircuitFactory.set_reservoirs_parameterizationSWAP(qrc_cfg, angle_positioning_linear, R, lam_0, seed, eps)
    assert np.array_equal(v1, v2)


def test_set_param_values_changes_with_seed(qrc_cfg):
    R = 4
    lam_0 = 0.05
    eps = 1e-8

    v1 = CircuitFactory.set_reservoirs_parameterizationSWAP(qrc_cfg, angle_positioning_linear, R, lam_0, seed=1, eps=eps)
    v2 = CircuitFactory.set_reservoirs_parameterizationSWAP(qrc_cfg, angle_positioning_linear, R, lam_0, seed=2, eps=eps)
    assert not np.array_equal(v1, v2)
