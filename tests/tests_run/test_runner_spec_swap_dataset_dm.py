import numpy as np
import pytest

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace
from src.qrc.circuits.circuit_factory import CircuitFactory
from src.qrc.circuits.configs import RingQRConfig
from src.qrc.circuits.utils import angle_positioning_linear
from src.qrc.run.circuit_run import ExactAerCircuitsRunner


# ----------------------------
# Helpers
# ----------------------------
def get_pub_dataset_fn():
    fn = getattr(CircuitFactory, "create_pubs_dataset_reservoirs_IsingRingSWAP", None)
    if fn is None:
        fn = getattr(CircuitFactory, "create_pubs_dataset_reservoir_IsingRingSWAP", None)
    if fn is None:
        raise AttributeError(
            "Could not find create_pubs_dataset_reservoirs_IsingRingSWAP or create_pubs_dataset_reservoir_IsingRingSWAP"
        )
    return fn


def param_order_from_metadata(qc):
    """Must match how param_values columns are built: J + h_x + h_z + [lam]."""
    J = list(qc.metadata["J"])
    hx = list(qc.metadata["h_x"])
    hz = list(qc.metadata["h_z"])
    lam = qc.metadata["lam"]
    return J + hx + hz + [lam]


def lam_col_index(qc):
    order = param_order_from_metadata(qc)
    return len(order) - 1  # by construction


def plus_density(n):
    sv = Statevector.from_label("+" * n)
    return DensityMatrix(sv)


def run_reservoir_dm_direct(qc, cfg, row, seed=0, label="dm_res"):
    """
    Direct Aer run for a single bound parameter row.
    Saves ONLY the reservoir DM (qubits 0..n-1), returns DensityMatrix.
    """
    n = int(cfg.num_qubits)
    backend = AerSimulator(method="density_matrix")

    qc2 = qc.copy()
    # Always save with a label we control to avoid key ambiguity
    qc2.save_density_matrix(qubits=list(range(n)), label=label)

    order = param_order_from_metadata(qc2)
    bind = {p: float(row[j]) for j, p in enumerate(order)}
    bound = qc2.assign_parameters(bind, inplace=False)

    tqc = transpile(bound, backend=backend, optimization_level=0)
    result = backend.run(tqc, shots=1, seed_simulator=seed).result()
    data = result.data(0)
    dm = data[label]
    return dm if isinstance(dm, DensityMatrix) else DensityMatrix(dm)


def run_full_and_reduced_dm_direct(qc, cfg, row, seed=0):
    """
    Direct Aer run saving:
      - full density matrix
      - reduced reservoir density matrix
    and returning both DensityMatrix objects.
    """
    n = int(cfg.num_qubits)
    backend = AerSimulator(method="density_matrix")

    qc2 = qc.copy()
    qc2.save_density_matrix(label="dm_full")
    qc2.save_density_matrix(qubits=list(range(n)), label="dm_res")

    order = param_order_from_metadata(qc2)
    bind = {p: float(row[j]) for j, p in enumerate(order)}
    bound = qc2.assign_parameters(bind, inplace=False)

    tqc = transpile(bound, backend=backend, optimization_level=0)
    result = backend.run(tqc, shots=1, seed_simulator=seed).result()
    data = result.data(0)

    dm_full = data["dm_full"]
    dm_res = data["dm_res"]
    dm_full = dm_full if isinstance(dm_full, DensityMatrix) else DensityMatrix(dm_full)
    dm_res = dm_res if isinstance(dm_res, DensityMatrix) else DensityMatrix(dm_res)
    return dm_full, dm_res


def run_runner(runner, pubs, seed=0, opt=0):
    """
    Call run_pubs with common kwargs, but tolerate differing signatures.
    """
    try:
        return runner.run_pubs(
            pubs=pubs,
            seed_simulator=seed,
            optimization_level=opt,
            device="CPU",
            max_parallel_threads=1,
            max_parallel_experiments=1,
            max_parallel_shots=1,
        )
    except TypeError:
        return runner.run_pubs(pubs=pubs, seed_simulator=seed, optimization_level=opt)


# ----------------------------
# Fixtures
# ----------------------------
@pytest.fixture
def cfg_small():
    # Keep it tiny/fast but non-trivial
    input_dim = 5
    n = 2
    seed = 123
    return RingQRConfig(input_dim=input_dim, num_qubits=n, seed=seed)


@pytest.fixture
def random_X(cfg_small):
    eps = 1e-6
    N, w, d = 2, 1, cfg_small.input_dim  # IMPORTANT: w=1 for exact mixture spec test
    rng = np.random.default_rng(2026)
    return rng.uniform(-1 + eps, 1 - eps, size=(N, w, d))


@pytest.fixture
def pubs(cfg_small, random_X):
    fn = get_pub_dataset_fn()
    return fn(
        cfg=cfg_small,
        angle_positioning=angle_positioning_linear,
        X=random_X,
        lam_0=0.05,
        num_reservoirs=3,
        seed=999,   # ensure deterministic parameterization draw if supported
    )


# ============================================================
# 1) Spec test: one-step SWAP channel matches exact mixture
# ============================================================
def test_one_step_swap_channel_matches_mixture_formula(cfg_small, pubs):
    """
    For w=1, the circuit implements:
        rho_out = lam * rho_unitary + (1-lam) * rho_plus
    where rho_unitary is the output with lam=1 (i.e., no replacement).
    """
    runner = ExactAerCircuitsRunner(cfg_small)
    res = run_runner(runner, pubs, seed=0, opt=0)

    assert res.states.ndim == 4  # (N,R,2^n,2^n)
    N, R, D1, D2 = res.states.shape
    assert N == len(pubs)
    assert D1 == D2 == 2 ** cfg_small.num_qubits

    dm_plus = plus_density(cfg_small.num_qubits)

    # Use the first window only (w=1, spec is exact per window)
    qc, vals = pubs[0]
    lam_idx = lam_col_index(qc)

    for r in range(R):
        lam = float(vals[r, lam_idx])
        dm_out = DensityMatrix(res.states[0, r])

        # Reference unitary output: same row but with lam=1
        row_unitary = vals[r].copy()
        row_unitary[lam_idx] = 1.0
        dm_unitary = run_reservoir_dm_direct(qc, cfg_small, row_unitary, seed=0, label="dm_res")

        expected = lam * dm_unitary.data + (1.0 - lam) * dm_plus.data

        assert np.allclose(dm_out.data, expected, atol=1e-10, rtol=0.0), f"Mismatch at reservoir r={r}, lam={lam}"


# ============================================================
# 2) Spec test: saved reservoir DM equals partial trace of full DM
# ============================================================
def test_saved_reservoir_dm_matches_partial_trace(cfg_small, pubs):
    """
    Verifies 'save_density_matrix(qubits=reservoir)' is consistent with full DM.
    This catches qubit-order / subset mistakes.
    """
    qc, vals = pubs[0]
    row = vals[0]  # any reservoir row

    dm_full, dm_res_saved = run_full_and_reduced_dm_direct(qc, cfg_small, row, seed=0)

    n = int(cfg_small.num_qubits)
    total = qc.num_qubits
    # env qubits are everything except reservoir [0..n-1]
    env = list(range(n, total))

    dm_res_traced = partial_trace(dm_full, env)

    # Ensure same dims and numeric match
    assert dm_res_traced.dim == dm_res_saved.dim
    assert np.allclose(dm_res_traced.data, dm_res_saved.data, atol=1e-10, rtol=0.0)


# ============================================================
# 3) Dataset-level invariants: parameter reuse + outputs vary + determinism
# ============================================================
def test_dataset_parameterization_reuse_and_output_variation(cfg_small, pubs):
    """
    The intention being to reuse the same (R,P) reservoir parameterization across all windows.
    This test verifies:
      - pubs[i][1] matrices are identical across windows
      - outputs differ across windows for same reservoir (with high prob)
      - outputs differ across reservoirs for same window (with high prob)
      - determinism: same seed => same states
    """
    assert len(pubs) >= 2, "Need at least 2 windows in pubs fixture"
    qc0, vals0 = pubs[0]
    qc1, vals1 = pubs[1]

    # Parameterization reuse across windows
    assert vals0.shape == vals1.shape
    assert np.allclose(vals0, vals1, atol=0.0, rtol=0.0), "Expected identical parameterization matrices across windows"

    runner = ExactAerCircuitsRunner(cfg_small)
    res1 = run_runner(runner, pubs, seed=0, opt=0)
    res2 = run_runner(runner, pubs, seed=0, opt=0)

    assert np.allclose(res1.states, res2.states, atol=0.0, rtol=0.0), "Runner should be deterministic with fixed seed"

    states = res1.states
    N, R, d1, d2 = states.shape
    assert N == len(pubs)
    assert d1 == d2 == 2 ** cfg_small.num_qubits

    # Outputs should vary across windows for at least one reservoir
    # (very unlikely to fail unless something is wrong)
    diffs_win = []
    for r in range(R):
        diffs_win.append(np.linalg.norm(states[0, r] - states[1, r]))
    assert max(diffs_win) > 1e-6, "Suspicious: outputs identical across different windows for all reservoirs"

    # Outputs should vary across reservoirs for at least one window
    diffs_res = []
    for r in range(1, R):
        diffs_res.append(np.linalg.norm(states[0, 0] - states[0, r]))
    assert max(diffs_res) > 1e-6, "Suspicious: outputs identical across different reservoirs for same window"
