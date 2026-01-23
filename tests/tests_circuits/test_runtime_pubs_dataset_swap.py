import numpy as np
import pytest

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, partial_trace

try:
    from src.qrc.circuits.configs import RingQRConfig as Config
except Exception:
    from src.qrc.circuits.configs import RingQRCConfig as Config

from src.qrc.circuits.circuit_factory import CircuitFactory
from src.qrc.circuits.utils import angle_positioning_tanh

qiskit_aer = pytest.importorskip("qiskit_aer")
from qiskit_aer import AerSimulator


# -------------------------
# Aer helpers
# -------------------------

def run_dm(qc: QuantumCircuit, seed: int = 0) -> DensityMatrix:
    backend = AerSimulator(method="density_matrix")
    qc2 = qc.copy()
    qc2.save_density_matrix()
    tqc = transpile(qc2, backend)
    res = backend.run(tqc, shots=1, seed_simulator=seed).result()
    return DensityMatrix(res.data(0)["density_matrix"])


def reduce_reservoir(dm_full: DensityMatrix, n: int) -> DensityMatrix:
    """Keep reservoir qubits 0..n-1, trace out env n..2n (aux+coin)."""
    env = list(range(n, 2 * n + 1))
    return DensityMatrix(partial_trace(dm_full, env))


# -------------------------
# Closed-form expected state for n=1, z=0, hz=pi, hx=0
# -------------------------

def expected_rho_after_w_steps(lam: float, w: int) -> np.ndarray:
    """
    Starting from |+>, each step does:
      U = Rz(pi) flips x -> -x
      rho -> lam * U rho U^† + (1-lam) * |+><+|
    On Bloch-x: x_{t+1} = (1-lam) - lam x_t, x_0 = 1.
    """
    assert 0.0 <= lam <= 1.0
    x_star = (1.0 - lam) / (1.0 + lam) if lam != 1.0 else 0.0
    x_w = x_star + (1.0 - x_star) * ((-lam) ** w)

    # rho = 1/2 [[1, x], [x, 1]]
    return np.array([[0.5, 0.5 * x_w], [0.5 * x_w, 0.5]], dtype=complex)


def make_param_values_for_step_order(qc_step: QuantumCircuit, *, hx: float, hz: float, lam: float, R: int) -> np.ndarray:
    """
    Build param_values in the SAME column order as CircuitFactory.set_reservoirs_parameterizationSWAP:
      [J..., h_x..., h_z..., lam]
    For n=1, J is empty, so this is [h_x[0], h_z[0], lam].
    """
    J = list(qc_step.metadata["J"])
    hx_params = list(qc_step.metadata["h_x"])
    hz_params = list(qc_step.metadata["h_z"])
    lam_param = qc_step.metadata["lam"]

    order = J + hx_params + hz_params + [lam_param]
    P = len(order)

    vals = np.zeros((R, P), dtype=float)
    # fill: J=0, hx,hz constant, lam varies per row (caller can overwrite)
    # here we just set constant lam for all rows; test can modify
    for r in range(R):
        for j, p in enumerate(order):
            if p.name.startswith("h_x"):
                vals[r, j] = hx
            elif p.name.startswith("h_z"):
                vals[r, j] = hz
            elif p.name == "lam":
                vals[r, j] = lam
            else:
                vals[r, j] = 0.0
    return vals


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def cfg1():
    # Analytic regime: n=1, d=1
    return Config(input_dim=1, num_qubits=1, seed=0)


@pytest.fixture
def X_zero(cfg1):
    # dataset X: shape (N, w, d), all zeros => z_t = 0
    N, w, d = 3, 4, cfg1.input_dim
    return np.zeros((N, w, d), dtype=float)


# -------------------------
# Runtime tests for create_pubs_dataset_reservoirs_IsingRingSWAP
# -------------------------

def test_dataset_pubs_runtime_expected_state_single_reservoir(cfg1, X_zero, monkeypatch):
    """
    Strong runtime check:
    Patch parameterization to force hz=pi, hx=0, lambda=lam0 for all reservoirs (R=1),
    then every pub circuit output (reduced to reservoir) must match the closed-form rho_w.
    This ALSO catches any param_values <-> qc.parameters column-order mismatch.
    """
    lam0 = 0.37
    R = 1
    w = X_zero.shape[1]

    qc_step = CircuitFactory.createIsingRingCircuitSWAP(cfg1, angle_positioning_tanh)
    forced_param_values = make_param_values_for_step_order(qc_step, hx=0.0, hz=np.pi, lam=lam0, R=R)

    def patched_set_reservoirs_parameterizationSWAP(*args, **kwargs):
        return forced_param_values

    monkeypatch.setattr(
        CircuitFactory,
        "set_reservoirs_parameterizationSWAP",
        staticmethod(patched_set_reservoirs_parameterizationSWAP),
    )

    pubs = CircuitFactory.create_pubs_dataset_reservoirs_IsingRingSWAP(
        cfg=cfg1,
        angle_positioning=angle_positioning_tanh,
        X=X_zero,
        lam_0=lam0,
        num_reservoirs=R,
        seed=123,
        eps=1e-8,
    )

    expected = expected_rho_after_w_steps(lam=lam0, w=w)

    for qc, vals in pubs:
        # bind using pub convention: vals row aligns with qc.parameters
        bind = dict(zip(list(qc.parameters), vals[0]))
        dm_full = run_dm(qc.assign_parameters(bind, inplace=False), seed=0)
        dm_res = reduce_reservoir(dm_full, n=cfg1.num_qubits)
        assert np.allclose(dm_res.data, expected, atol=1e-10, rtol=0.0)


def test_dataset_pubs_runtime_expected_state_multiple_reservoirs(cfg1, X_zero, monkeypatch):
    """
    Same as above but with multiple reservoirs (R>1) and different lambdas per reservoir row.
    Verifies each row produces the correct analytic rho_w(lam_r).
    """
    R = 3
    w = X_zero.shape[1]
    lams = [0.15, 0.37, 0.63]

    qc_step = CircuitFactory.createIsingRingCircuitSWAP(cfg1, angle_positioning_tanh)
    forced_param_values = make_param_values_for_step_order(qc_step, hx=0.0, hz=np.pi, lam=lams[0], R=R)
    # overwrite lam column per row
    lam_col = forced_param_values.shape[1] - 1
    for r in range(R):
        forced_param_values[r, lam_col] = lams[r]

    def patched_set_reservoirs_parameterizationSWAP(*args, **kwargs):
        return forced_param_values

    monkeypatch.setattr(
        CircuitFactory,
        "set_reservoirs_parameterizationSWAP",
        staticmethod(patched_set_reservoirs_parameterizationSWAP),
    )

    pubs = CircuitFactory.create_pubs_dataset_reservoirs_IsingRingSWAP(
        cfg=cfg1,
        angle_positioning=angle_positioning_tanh,
        X=X_zero,
        lam_0=lams[0],
        num_reservoirs=R,
        seed=999,
        eps=1e-8,
    )

    for qc, vals in pubs:
        for r in range(R):
            bind = dict(zip(list(qc.parameters), vals[r]))
            dm_full = run_dm(qc.assign_parameters(bind, inplace=False), seed=0)
            dm_res = reduce_reservoir(dm_full, n=cfg1.num_qubits)

            expected = expected_rho_after_w_steps(lam=lams[r], w=w)
            assert np.allclose(dm_res.data, expected, atol=1e-10, rtol=0.0)


def test_dataset_pubs_runtime_consistency_row_vs_name_binding(cfg1, X_zero, monkeypatch):
    """
    Detect subtle parameter-order mismatches:
    Compare output when binding via pub row vector (zip(qc.parameters, row))
    versus binding by parameter NAMES (independent of column order).
    If these differ, your pub column order does not match qc.parameters.
    """
    lam0 = 0.41
    R = 1

    qc_step = CircuitFactory.createIsingRingCircuitSWAP(cfg1, angle_positioning_tanh)
    forced_param_values = make_param_values_for_step_order(qc_step, hx=0.0, hz=np.pi, lam=lam0, R=R)

    def patched_set_reservoirs_parameterizationSWAP(*args, **kwargs):
        return forced_param_values

    monkeypatch.setattr(
        CircuitFactory,
        "set_reservoirs_parameterizationSWAP",
        staticmethod(patched_set_reservoirs_parameterizationSWAP),
    )

    pubs = CircuitFactory.create_pubs_dataset_reservoirs_IsingRingSWAP(
        cfg=cfg1,
        angle_positioning=angle_positioning_tanh,
        X=X_zero[:1],  # one window is enough
        lam_0=lam0,
        num_reservoirs=R,
        seed=0,
        eps=1e-8,
    )

    qc, vals = pubs[0]
    row = vals[0]

    # (A) pub binding: assumes correct column order
    bind_row = dict(zip(list(qc.parameters), row))
    dmA = run_dm(qc.assign_parameters(bind_row, inplace=False), seed=0)
    dmA_res = reduce_reservoir(dmA, n=cfg1.num_qubits)

    # (B) name-based binding: immune to column ordering
    bind_name = {}
    for p in list(qc.parameters):
        if p.name.startswith("h_x"):
            bind_name[p] = 0.0
        elif p.name.startswith("h_z"):
            bind_name[p] = float(np.pi)
        elif p.name == "lam":
            bind_name[p] = float(lam0)
        elif p.name.startswith("J"):
            bind_name[p] = 0.0
        else:
            raise AssertionError(f"Unexpected parameter in window circuit: {p} (name={p.name})")

    dmB = run_dm(qc.assign_parameters(bind_name, inplace=False), seed=0)
    dmB_res = reduce_reservoir(dmB, n=cfg1.num_qubits)

    assert np.allclose(dmA_res.data, dmB_res.data, atol=1e-12, rtol=0.0)
