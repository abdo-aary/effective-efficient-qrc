import numpy as np
import pytest

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, partial_trace

from src.qrc.circuits.qrc_configs import RingQRConfig
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


def plus_state_density(n: int) -> DensityMatrix:
    qc = QuantumCircuit(n)
    qc.h(range(n))
    return DensityMatrix.from_instruction(qc)


# -------------------------
# Closed-form expected state (same regime as your dataset test)
# -------------------------

def expected_rho_after_w_steps(lam: float, w: int) -> np.ndarray:
    """
    n=1, z_t=0 => no injection.
    Choose hz=pi, hx=0, J empty:
      U = Rz(pi) flips x -> -x
      rho -> lam * U rho U^† + (1-lam) * |+><+|
    On Bloch-x: x_{t+1} = (1-lam) - lam x_t, x_0 = 1.
    """
    assert 0.0 <= lam <= 1.0
    x_star = (1.0 - lam) / (1.0 + lam) if lam != 1.0 else 0.0
    x_w = x_star + (1.0 - x_star) * ((-lam) ** w)
    return np.array([[0.5, 0.5 * x_w], [0.5 * x_w, 0.5]], dtype=complex)


# -------------------------
# Tests
# -------------------------

def test_template_structure_and_metadata():
    cfg = RingQRConfig(input_dim=5, num_qubits=3, seed=123)
    w = 4

    qc, z_steps = CircuitFactory.instantiateFullIsingRingEvolutionTemplate(
        qrc_cfg=cfg,
        angle_positioning=angle_positioning_tanh,
        w=w,
    )

    n = cfg.num_qubits
    assert qc.name == "QRC_template"
    assert qc.num_qubits == 2 * n + 1

    # Metadata keys
    for k in ["z_steps", "J", "h_x", "h_z", "lam"]:
        assert k in qc.metadata

    # z_steps list
    assert isinstance(z_steps, list)
    assert len(z_steps) == w
    assert qc.metadata["z_steps"] is z_steps  # should be the same list object

    # Each z_t has length n and correct naming
    for t, zt in enumerate(z_steps):
        assert len(zt) == n
        assert zt.name == f"z_{t}"


def test_template_parameter_count_and_no_leftover_base_z():
    cfg = RingQRConfig(input_dim=5, num_qubits=3, seed=0)
    w = 3
    n = cfg.num_qubits
    m = len(cfg.topology.edges)

    qc, z_steps = CircuitFactory.instantiateFullIsingRingEvolutionTemplate(
        qrc_cfg=cfg,
        angle_positioning=angle_positioning_tanh,
        w=w,
    )

    # Expected unique parameters:
    #   - injected: w*n
    #   - shared reservoir: m (J) + n (h_x) + n (h_z) + 1 (lam)
    expected = w * n + m + 2 * n + 1
    assert len(qc.parameters) == expected

    # Ensure the original base ParameterVector("z", n) is not present anymore
    # (base elements are typically named "z[0]", "z[1]", ...)
    names = {p.name for p in qc.parameters}
    assert not any(name.startswith("z[") for name in names), f"Found leftover base z params: {sorted(names)}"

    # Ensure per-step params are present
    for t in range(w):
        assert any(name.startswith(f"z_{t}[") for name in names), f"Missing z_{t}[*] params"


def test_template_per_step_injection_is_used_correctly_lam_one():
    """
    Strong functional check with n=1, lam=1, J=hx=hz=0.
    Then each step should act as only the injection Ry(theta(z_t)).
    So the window is just Ry(theta(z_1)) * Ry(theta(z_0)) applied to |+>.
    """
    cfg = RingQRConfig(input_dim=1, num_qubits=1, seed=0)
    w = 2
    qc, z_steps = CircuitFactory.instantiateFullIsingRingEvolutionTemplate(
        qrc_cfg=cfg,
        angle_positioning=angle_positioning_tanh,
        w=w,
    )

    # choose distinct z values to make ordering matter
    z0 = 0.20
    z1 = -0.35

    # bind template params
    bind = {}
    bind.update({z_steps[0][0]: z0, z_steps[1][0]: z1})

    # shared reservoir params (n=1 => J empty)
    bind[qc.metadata["h_x"][0]] = 0.0
    bind[qc.metadata["h_z"][0]] = 0.0
    bind[qc.metadata["lam"]] = 1.0

    qc_bound = qc.assign_parameters(bind, inplace=False)
    dm_full = run_dm(qc_bound, seed=0)
    dm_res = reduce_reservoir(dm_full, n=1)

    # reference 1-qubit circuit: |0> -> H -> Ry(theta(z0)) -> Ry(theta(z1))
    # angle_positioning_tanh uses theta(z)=scale*tanh(z), with default scale=pi*(1-1e-6)
    scale = np.pi * (1.0 - 1e-6)
    theta0 = scale * np.tanh(z0)
    theta1 = scale * np.tanh(z1)

    qc_ref = QuantumCircuit(1)
    qc_ref.h(0)
    qc_ref.ry(theta0, 0)
    qc_ref.ry(theta1, 0)
    dm_ref = DensityMatrix.from_instruction(qc_ref)

    assert np.allclose(dm_res.data, dm_ref.data, atol=1e-10, rtol=0.0)


def test_template_matches_closed_form_expected_state_n1():
    """
    n=1 analytic test (same regime as your optimized dataset tests):
      z_t = 0, hz = pi, hx=0, lam=const
    Compare reduced reservoir state to closed-form rho_w.
    """
    cfg = RingQRConfig(input_dim=1, num_qubits=1, seed=0)
    w = 5
    lam = 0.37

    qc, z_steps = CircuitFactory.instantiateFullIsingRingEvolutionTemplate(
        qrc_cfg=cfg,
        angle_positioning=angle_positioning_tanh,
        w=w,
    )

    bind = {}
    # z_t = 0 for all steps
    for t in range(w):
        bind[z_steps[t][0]] = 0.0

    # J empty, set fields and lam
    bind[qc.metadata["h_x"][0]] = 0.0
    bind[qc.metadata["h_z"][0]] = float(np.pi)
    bind[qc.metadata["lam"]] = float(lam)

    qc_bound = qc.assign_parameters(bind, inplace=False)
    dm_full = run_dm(qc_bound, seed=0)
    dm_res = reduce_reservoir(dm_full, n=1)

    expected = expected_rho_after_w_steps(lam=lam, w=w)
    assert np.allclose(dm_res.data, expected, atol=1e-10, rtol=0.0)


def test_template_equivalent_to_instantiate_full_evolution_when_binding_z_steps():
    """
    Equivalence check:
      - qc_bound_window = instantiateFullIsingRingEvolution(...)  (inputs already bound)
      - qc_template_bound = instantiateFullIsingRingEvolutionTemplate(...), then bind each z_t = x_t @ Pi
    Then after also binding the shared reservoir params to the SAME numeric values,
    the reduced reservoir density matrices must match.
    """
    cfg = RingQRConfig(input_dim=2, num_qubits=2, seed=123)
    n = cfg.num_qubits
    w = 3

    rng = np.random.default_rng(7)
    x_window = rng.normal(size=(w, cfg.input_dim))

    qc_bound_window = CircuitFactory.instantiateFullIsingRingEvolution(
        qrc_cfg=cfg,
        angle_positioning=angle_positioning_tanh,
        x_window=x_window,
    )

    qc_template, z_steps = CircuitFactory.instantiateFullIsingRingEvolutionTemplate(
        qrc_cfg=cfg,
        angle_positioning=angle_positioning_tanh,
        w=w,
    )

    # Bind template z_steps from the same projection used by instantiateFullIsingRingEvolution
    bind_template_inputs = {}
    for t in range(w):
        z_t = x_window[t] @ cfg.projection  # shape (n,)
        for j in range(n):
            bind_template_inputs[z_steps[t][j]] = float(z_t[j])

    qc_template_inputs_bound = qc_template.assign_parameters(bind_template_inputs, inplace=False)

    # Now bind *shared reservoir parameters* to the same numeric values in both circuits.
    # (The Parameter objects differ between circuits, so bind by each circuit's own metadata.)
    J_vals = rng.uniform(-np.pi, np.pi, size=len(cfg.topology.edges))
    hx_vals = rng.uniform(-np.pi, np.pi, size=n)
    hz_vals = rng.uniform(-np.pi, np.pi, size=n)
    lam_val = float(rng.uniform(0.0, 1.0))

    def bind_shared(qc: QuantumCircuit) -> QuantumCircuit:
        bind = {}
        for p, v in zip(qc.metadata["J"], J_vals):
            bind[p] = float(v)
        for p, v in zip(qc.metadata["h_x"], hx_vals):
            bind[p] = float(v)
        for p, v in zip(qc.metadata["h_z"], hz_vals):
            bind[p] = float(v)
        bind[qc.metadata["lam"]] = lam_val
        return qc.assign_parameters(bind, inplace=False)

    qcA = bind_shared(qc_bound_window)
    qcB = bind_shared(qc_template_inputs_bound)

    dmA = reduce_reservoir(run_dm(qcA, seed=0), n=n)
    dmB = reduce_reservoir(run_dm(qcB, seed=0), n=n)

    assert np.allclose(dmA.data, dmB.data, atol=1e-10, rtol=0.0)
