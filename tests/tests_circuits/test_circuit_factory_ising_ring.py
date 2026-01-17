import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, partial_trace

from src.qrc.circuits.configs import RingQRConfig
from src.qrc.circuits.circuit_factory import CircuitFactory
from src.qrc.circuits.utils import angle_positioning_linear


# -------------------------
# Helpers
# -------------------------

def aer_density_backend():
    # Deterministic density-matrix simulator (no sampling noise in the saved density matrix)
    return AerSimulator(method="density_matrix")


def run_density_matrix(qc: QuantumCircuit, seed: int = 0) -> DensityMatrix:
    """Run circuit on Aer density-matrix and return final DensityMatrix."""
    backend = aer_density_backend()
    qc2 = qc.copy()
    qc2.save_density_matrix()

    tqc = transpile(qc2, backend)
    job = backend.run(tqc, shots=1, seed_simulator=seed)
    res = job.result()
    dm = res.data(0)["density_matrix"]
    return DensityMatrix(dm)


def plus_state_density(n: int) -> DensityMatrix:
    """|+><+|^{⊗n} as a DensityMatrix."""
    qc = QuantumCircuit(n)
    qc.h(list(range(n)))
    return DensityMatrix.from_instruction(qc)


def random_bind_map(qc: QuantumCircuit, rng: np.random.Generator, lam_value=None):
    """Create a full parameter binding dict for the circuit."""
    eps = 1e-8

    z_params = qc.metadata["z"]
    J_params = qc.metadata["J"]
    hx_params = qc.metadata["h_x"]
    hz_params = qc.metadata["h_z"]
    lam_param = qc.metadata["lam"]

    # IMPORTANT: z_t must match len(z_params) (not len(J_params))
    z_t = rng.uniform(-1 + eps, 1 - eps, size=len(z_params))
    J_vals = rng.uniform(-np.pi, np.pi, size=len(J_params))
    hx_vals = rng.uniform(-np.pi, np.pi, size=len(hx_params))
    hz_vals = rng.uniform(-np.pi, np.pi, size=len(hz_params))

    if lam_value is None:
        lam_value = float(rng.uniform(0.0, 1.0))  # allow exact [0,1] in tests
    else:
        lam_value = float(lam_value)

    bind = {}
    bind.update(dict(zip(z_params, z_t)))
    bind.update(dict(zip(J_params, J_vals)))
    bind.update(dict(zip(hx_params, hx_vals)))
    bind.update(dict(zip(hz_params, hz_vals)))
    bind.update({lam_param: lam_value})

    return bind


def build_unitary_reference_circuit(cfg: RingQRConfig, qc_template: QuantumCircuit) -> QuantumCircuit:
    """
    Reference circuit implementing ONLY:
        data injection -> Ising unitary
    using the *same* Parameter objects from qc_template.metadata.
    """
    n = cfg.num_qubits
    topology = cfg.topology

    z  = qc_template.metadata["z"]
    J  = qc_template.metadata["J"]
    hx = qc_template.metadata["h_x"]
    hz = qc_template.metadata["h_z"]

    angles = angle_positioning_linear(z)  # list of ParameterExpressions, length len(z)

    qc_ref = QuantumCircuit(n)

    # Data injection (match factory: first n angles applied to the n reservoir qubits)
    for i in range(n):
        qc_ref.ry(angles[i], i)

    # Ising ring unitary (match edge order)
    for e, (q1, q2) in enumerate(topology.edges):
        qc_ref.rzz(J[e], q1, q2)

    # Local fields
    for i in range(n):
        qc_ref.rz(hz[i], i)
    for i in range(n):
        qc_ref.rx(hx[i], i)

    return qc_ref


def reduced_reservoir(dm_full: DensityMatrix, n: int) -> DensityMatrix:
    """Trace out aux (n..2n-1) and coin (2n); keep reservoir (0..n-1)."""
    env = list(range(n, 2 * n + 1))   # aux + coin
    dm_red = partial_trace(dm_full, env)
    return DensityMatrix(dm_red)


# -------------------------
# Tests
# -------------------------

def test_create_ising_ring_circuit_structure_and_metadata():
    cfg = RingQRConfig(input_dim=10, num_qubits=3, seed=12345)
    qc = CircuitFactory.createIsingRingCircuitSWAP(
        cfg=cfg, angle_positioning=angle_positioning_linear
    )

    assert qc.num_qubits == 2 * cfg.num_qubits + 1  # reservoir + coin

    # Metadata present
    for k in ["z", "J", "h_x", "h_z", "lam"]:
        assert k in qc.metadata

    # Parameter vector lengths
    assert len(qc.metadata["z"]) == cfg.num_qubits
    assert len(qc.metadata["J"]) == len(cfg.topology.edges)
    assert len(qc.metadata["h_x"]) == cfg.num_qubits
    assert len(qc.metadata["h_z"]) == cfg.num_qubits


def test_parameter_binding_leaves_no_unbound_params():
    cfg = RingQRConfig(input_dim=7, num_qubits=3, seed=1)
    qc = CircuitFactory.createIsingRingCircuitSWAP(cfg, angle_positioning_linear)

    rng = np.random.default_rng(0)
    bind = random_bind_map(qc, rng, lam_value=0.37)
    qc_bound = qc.assign_parameters(bind, inplace=False)

    # after full bind, no free parameters
    assert len(qc_bound.parameters) == 0


def test_contraction_lambda_zero_outputs_plus_state():
    cfg = RingQRConfig(input_dim=9, num_qubits=3, seed=2)
    qc = CircuitFactory.createIsingRingCircuitSWAP(cfg, angle_positioning_linear)

    rng = np.random.default_rng(123)
    bind = random_bind_map(qc, rng, lam_value=0.0)
    qc_bound = qc.assign_parameters(bind, inplace=False)

    dm_full = run_density_matrix(qc_bound, seed=0)

    dm_res = reduced_reservoir(dm_full, cfg.num_qubits)

    dm_plus = plus_state_density(cfg.num_qubits)

    assert np.allclose(dm_res.data, dm_plus.data, atol=1e-10, rtol=0.0)


def test_contraction_lambda_one_matches_unitary_reference():
    cfg = RingQRConfig(input_dim=9, num_qubits=3, seed=3)
    qc = CircuitFactory.createIsingRingCircuitSWAP(cfg, angle_positioning_linear)
    qc_ref = build_unitary_reference_circuit(cfg, qc)

    rng = np.random.default_rng(456)
    bind = random_bind_map(qc, rng, lam_value=1.0)

    # Full circuit
    dm_full = run_density_matrix(qc.assign_parameters(bind, inplace=False), seed=0)
    dm_res = reduced_reservoir(dm_full, cfg.num_qubits)

    # Reference unitary-only circuit (bind same params, but ignore 'lam')
    bind_ref = {p: v for p, v in bind.items() if p != qc.metadata["lam"]}
    dm_ref = run_density_matrix(qc_ref.assign_parameters(bind_ref, inplace=False), seed=0)

    assert np.allclose(dm_res.data, dm_ref.data, atol=1e-10, rtol=0.0)


def test_contraction_mixture_for_intermediate_lambda():
    cfg = RingQRConfig(input_dim=10, num_qubits=3, seed=4)
    qc = CircuitFactory.createIsingRingCircuitSWAP(cfg, angle_positioning_linear)
    qc_ref = build_unitary_reference_circuit(cfg, qc)

    rng = np.random.default_rng(789)
    lam = 0.23
    bind = random_bind_map(qc, rng, lam_value=lam)

    dm_full = run_density_matrix(qc.assign_parameters(bind, inplace=False), seed=0)
    dm_res = reduced_reservoir(dm_full, cfg.num_qubits)

    bind_ref = {p: v for p, v in bind.items() if p != qc.metadata["lam"]}
    dm_unitary = run_density_matrix(qc_ref.assign_parameters(bind_ref, inplace=False), seed=0)

    dm_plus = plus_state_density(cfg.num_qubits)
    expected = lam * dm_unitary.data + (1.0 - lam) * dm_plus.data

    assert np.allclose(dm_res.data, expected, atol=1e-10, rtol=0.0)


def test_density_matrix_output_is_deterministic_across_seeds():
    """
    Strong determinism check: run with different Aer seeds.
    If measurement were being sampled per-shot in the saved density matrix, this would differ.
    """
    cfg = RingQRConfig(input_dim=8, num_qubits=3, seed=5)
    qc = CircuitFactory.createIsingRingCircuitSWAP(cfg, angle_positioning_linear)

    rng = np.random.default_rng(999)
    bind = random_bind_map(qc, rng, lam_value=0.61)
    qc_bound = qc.assign_parameters(bind, inplace=False)

    dm1 = run_density_matrix(qc_bound, seed=1)
    dm2 = run_density_matrix(qc_bound, seed=999)

    assert np.allclose(dm1.data, dm2.data, atol=1e-10, rtol=0.0)
