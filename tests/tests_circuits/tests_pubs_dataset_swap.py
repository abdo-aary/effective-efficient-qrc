import numpy as np
import pytest

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, partial_trace
from src.qrc.circuits.qrc_configs import RingQRConfig as Config

from src.qrc.circuits.circuit_factory import CircuitFactory
from src.qrc.circuits.utils import angle_positioning_tanh, angle_positioning_linear

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


def bind_from_pub_row(qc: QuantumCircuit, row: np.ndarray) -> dict:
    """
    Optimized/template PUBs define an explicit parameter column order
    (qc.metadata['param_order']). Fall back to qc.parameters if absent.
    """
    order = qc.metadata.get("param_order", None)
    if order is None:
        order = list(qc.parameters)
    assert len(order) == len(row), f"Param order length {len(order)} != row length {len(row)}"
    return dict(zip(order, row))


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
    return np.array([[0.5, 0.5 * x_w], [0.5 * x_w, 0.5]], dtype=complex)


def make_param_values_for_step_order(
    qc_step: QuantumCircuit, *, hx: float, hz: float, lam: float, R: int
) -> np.ndarray:
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
# Tests: create_pubs_dataset_reservoirs_IsingRingSWAP (template entrypoint)
# -------------------------

def test_create_pubs_dataset_returns_template_pub(cfg1, X_zero):
    """
    The updated implementation should ALWAYS return template mode:
      pubs = [(qc_template, vals)] with vals shape (N, R, P_total).
    """
    lam_0 = 0.05
    R = 4

    pubs = CircuitFactory.create_pubs_dataset_reservoirs_IsingRingSWAP(
        qrc_cfg=cfg1,
        angle_positioning=angle_positioning_linear,
        X=X_zero,
        lam_0=lam_0,
        num_reservoirs=R,
        seed=123,
        eps=1e-8,
    )

    assert isinstance(pubs, list)
    assert len(pubs) == 1

    qc0, vals0 = pubs[0]
    assert isinstance(qc0, QuantumCircuit)
    assert isinstance(vals0, np.ndarray)
    assert vals0.ndim == 3

    assert vals0.shape[0] == X_zero.shape[0]  # N
    assert vals0.shape[1] == R                 # R
    assert "param_order" in qc0.metadata
    assert vals0.shape[2] == len(qc0.metadata["param_order"])


# -------------------------
# Runtime tests (adapted to internal reservoir sampling)
# -------------------------

def test_dataset_runtime_expected_state_single_reservoir(cfg1, X_zero, monkeypatch):
    """
    Patch sampling to force hz=pi, hx=0, lambda=lam0 for all reservoirs (R=1),
    then the window circuit output reduced to reservoir must match closed-form rho_w.
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
        qrc_cfg=cfg1,
        angle_positioning=angle_positioning_tanh,
        X=X_zero,
        lam_0=lam0,
        num_reservoirs=R,
        seed=123,
        eps=1e-8,
    )

    qc, vals = pubs[0]
    assert vals.ndim == 3

    expected = expected_rho_after_w_steps(lam=lam0, w=w)

    for i in range(X_zero.shape[0]):  # over windows
        bind = bind_from_pub_row(qc, vals[i, 0])
        dm_full = run_dm(qc.assign_parameters(bind, inplace=False), seed=0)
        dm_res = reduce_reservoir(dm_full, n=cfg1.num_qubits)
        assert np.allclose(dm_res.data, expected, atol=1e-10, rtol=0.0)


def test_dataset_runtime_expected_state_multiple_reservoirs(cfg1, X_zero, monkeypatch):
    """
    Multiple reservoirs (R>1), each with its own lambda row.
    Verifies each reservoir row produces correct analytic rho_w(lam_r).
    """
    R = 3
    w = X_zero.shape[1]
    lams = [0.15, 0.37, 0.63]

    qc_step = CircuitFactory.createIsingRingCircuitSWAP(cfg1, angle_positioning_tanh)
    forced_param_values = make_param_values_for_step_order(qc_step, hx=0.0, hz=np.pi, lam=lams[0], R=R)
    forced_param_values[:, -1] = np.asarray(lams, dtype=float)  # last col = lam

    def patched_set_reservoirs_parameterizationSWAP(*args, **kwargs):
        return forced_param_values

    monkeypatch.setattr(
        CircuitFactory,
        "set_reservoirs_parameterizationSWAP",
        staticmethod(patched_set_reservoirs_parameterizationSWAP),
    )

    pubs = CircuitFactory.create_pubs_dataset_reservoirs_IsingRingSWAP(
        qrc_cfg=cfg1,
        angle_positioning=angle_positioning_tanh,
        X=X_zero,
        lam_0=lams[0],
        num_reservoirs=R,
        seed=999,
        eps=1e-8,
    )

    qc, vals = pubs[0]
    assert vals.ndim == 3

    for i in range(X_zero.shape[0]):
        for r in range(R):
            bind = bind_from_pub_row(qc, vals[i, r])
            dm_full = run_dm(qc.assign_parameters(bind, inplace=False), seed=0)
            dm_res = reduce_reservoir(dm_full, n=cfg1.num_qubits)

            expected = expected_rho_after_w_steps(lam=lams[r], w=w)
            assert np.allclose(dm_res.data, expected, atol=1e-10, rtol=0.0)


def test_dataset_runtime_consistency_row_vs_name_binding(cfg1, X_zero, monkeypatch):
    """
    Detect parameter-order mismatches:
    Compare binding via pub row vector in qc.metadata['param_order']
    vs binding by parameter NAMES (immune to column order).
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
        qrc_cfg=cfg1,
        angle_positioning=angle_positioning_tanh,
        X=X_zero[:1],
        lam_0=lam0,
        num_reservoirs=R,
        seed=0,
        eps=1e-8,
    )

    qc, vals = pubs[0]
    assert vals.ndim == 3

    row = vals[0, 0]

    # (A) pub row binding via param_order
    bind_row = bind_from_pub_row(qc, row)
    dmA = run_dm(qc.assign_parameters(bind_row, inplace=False), seed=0)
    dmA_res = reduce_reservoir(dmA, n=cfg1.num_qubits)

    # (B) name-based binding
    bind_name = {}
    for p in list(qc.parameters):
        if p.name.startswith("z_") or p.name.startswith("z"):
            bind_name[p] = 0.0
        elif p.name.startswith("h_x"):
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
