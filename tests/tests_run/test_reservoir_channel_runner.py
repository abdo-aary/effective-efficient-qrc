import numpy as np
import pytest

pytest.importorskip("qiskit_aer")

from hydra.utils import instantiate
from omegaconf import OmegaConf
from qiskit.quantum_info import DensityMatrix, Operator, SparsePauliOp, Statevector

from src.models.qrc_featurizer import QRCFeaturizer
from src.qrc.circuits.circuit_factory import CircuitFactory
from src.qrc.circuits.qrc_configs import RingQRConfig
from src.qrc.circuits.utils import angle_positioning_linear, angle_positioning_tanh, generate_k_local_paulis
from src.qrc.run.circuit_run import ExactAerCircuitsRunner, ExactExpectationResults, ExactResults
from src.qrc.run.cs_fmp_retriever import CSFeatureMapsRetriever
from src.qrc.run.fmp_retriever import ExactFeatureMapsRetriever
from src.qrc.run.reservoir_channel_runner import ExactReservoirChannelRunner


def _make_pubs(cfg, X, *, angle_positioning=angle_positioning_linear, R=2, lam_0=0.17, seed=11):
    return CircuitFactory.create_pubs_dataset_reservoirs_IsingRingSWAP(
        qrc_cfg=cfg,
        angle_positioning=angle_positioning,
        X=X,
        num_reservoirs=R,
        lam_0=lam_0,
        seed=seed,
        eps=1e-8,
    )


def _run_aer(cfg, pubs):
    return ExactAerCircuitsRunner(cfg).run_pubs(
        pubs=pubs,
        seed_simulator=0,
        optimization_level=0,
        device="CPU",
        max_parallel_threads=1,
        max_parallel_experiments=1,
        max_parallel_shots=1,
    )


def _run_channel(cfg, pubs, *, angle_positioning_name="linear", engine="batched"):
    return ExactReservoirChannelRunner(cfg, state_dtype="complex128", chunk_size=3, engine=engine).run_pubs(
        pubs=pubs,
        angle_positioning_name=angle_positioning_name,
    )


def _require_cupy():
    cp = pytest.importorskip("cupy")
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CuPy is installed but CUDA is unavailable: {exc}")
    if device_count < 1:
        pytest.skip("No CUDA device available for CuPy reservoir-channel test.")
    return cp


def _assert_density_matrix_close(got, expected, atol=1e-10):
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=atol, rtol=0.0)
    np.testing.assert_allclose(got, got.conj().T, atol=atol, rtol=0.0)
    np.testing.assert_allclose(np.trace(got), 1.0 + 0.0j, atol=atol, rtol=0.0)


def _param_cols_by_name(qc):
    return {param.name: idx for idx, param in enumerate(qc.metadata["param_order"])}


def _assert_channel_reference_close(cfg, pubs, *, angle_positioning_name="linear", atol=1e-10):
    batched = _run_channel(cfg, pubs, angle_positioning_name=angle_positioning_name, engine="batched")
    reference = _run_channel(cfg, pubs, angle_positioning_name=angle_positioning_name, engine="reference")
    np.testing.assert_allclose(batched.states, reference.states, atol=atol, rtol=0.0)
    return batched, reference


def test_reservoir_channel_n1_w1_matches_analytic_lambda_cases():
    cfg = RingQRConfig(input_dim=1, num_qubits=1, seed=0)
    X = np.zeros((1, 1, 1), dtype=float)
    pubs = _make_pubs(cfg, X, R=3, lam_0=0.5, seed=0)
    qc, vals = pubs[0]
    cols = _param_cols_by_name(qc)

    lam_values = [0.0, 1.0, 0.23]
    for r, lam in enumerate(lam_values):
        vals[0, r, cols["h_z[0]"]] = np.pi
        vals[0, r, cols["h_x[0]"]] = 0.0
        vals[0, r, cols["lam"]] = lam

    channel, reference = _assert_channel_reference_close(cfg, pubs)
    aer = _run_aer(cfg, pubs)

    plus = DensityMatrix(Statevector.from_label("+")).data
    minus = DensityMatrix(Statevector.from_label("-")).data
    expected = np.stack([lam * minus + (1.0 - lam) * plus for lam in lam_values], axis=0)

    np.testing.assert_allclose(channel.states[0], expected, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(reference.states[0], expected, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(channel.states, aer.states, atol=1e-10, rtol=0.0)


@pytest.mark.parametrize(
    ("angle_positioning", "angle_positioning_name"),
    [(angle_positioning_linear, "linear"), (angle_positioning_tanh, "tanh")],
)
def test_reservoir_channel_batched_matches_reference_n2_w3(angle_positioning, angle_positioning_name):
    cfg = RingQRConfig(input_dim=2, num_qubits=2, seed=101)
    X = np.random.default_rng(102).uniform(-0.7, 0.7, size=(3, 3, 2))
    pubs = _make_pubs(cfg, X, angle_positioning=angle_positioning, R=2, lam_0=0.29, seed=103)

    _assert_channel_reference_close(cfg, pubs, angle_positioning_name=angle_positioning_name, atol=1e-10)


def test_reservoir_channel_matches_aer_n2_w1_all_states():
    cfg = RingQRConfig(input_dim=3, num_qubits=2, seed=3)
    X = np.random.default_rng(4).uniform(-0.5, 0.5, size=(2, 1, 3))
    pubs = _make_pubs(cfg, X, R=3, lam_0=0.19, seed=5)

    channel, _ = _assert_channel_reference_close(cfg, pubs)
    aer = _run_aer(cfg, pubs)

    assert isinstance(channel, ExactResults)
    assert channel.states.shape == aer.states.shape == (2, 3, 4, 4)
    np.testing.assert_allclose(channel.states, aer.states, atol=1e-10, rtol=0.0)
    _assert_density_matrix_close(channel.states[0, 0], aer.states[0, 0])


def test_reservoir_channel_matches_aer_n2_w3_all_states_tanh():
    cfg = RingQRConfig(input_dim=2, num_qubits=2, seed=7)
    X = np.random.default_rng(8).uniform(-0.7, 0.7, size=(2, 3, 2))
    pubs = _make_pubs(cfg, X, angle_positioning=angle_positioning_tanh, R=2, lam_0=0.11, seed=9)

    channel, _ = _assert_channel_reference_close(cfg, pubs, angle_positioning_name="tanh", atol=1e-10)
    aer = _run_aer(cfg, pubs)

    assert channel.states.shape == aer.states.shape == (2, 2, 4, 4)
    np.testing.assert_allclose(channel.states, aer.states, atol=1e-9, rtol=0.0)


def test_reservoir_channel_exact_feature_maps_match_aer():
    cfg = RingQRConfig(input_dim=2, num_qubits=2, seed=10)
    X = np.random.default_rng(11).uniform(-0.4, 0.4, size=(2, 2, 2))
    pubs = _make_pubs(cfg, X, R=2, lam_0=0.21, seed=12)
    observables = generate_k_local_paulis(locality=2, num_qubits=2)

    aer_phi = ExactFeatureMapsRetriever(cfg, observables).get_feature_maps(_run_aer(cfg, pubs))
    batched, reference = _assert_channel_reference_close(cfg, pubs)
    channel_phi = ExactFeatureMapsRetriever(cfg, observables).get_feature_maps(batched)
    reference_phi = ExactFeatureMapsRetriever(cfg, observables).get_feature_maps(reference)

    np.testing.assert_allclose(channel_phi, reference_phi, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(channel_phi, aer_phi, atol=1e-10, rtol=0.0)


def test_reservoir_channel_cshadow_feature_maps_match_aer_given_seed():
    cfg = RingQRConfig(input_dim=2, num_qubits=2, seed=13)
    X = np.random.default_rng(14).uniform(-0.4, 0.4, size=(2, 2, 2))
    pubs = _make_pubs(cfg, X, R=2, lam_0=0.27, seed=15)
    observables = generate_k_local_paulis(locality=2, num_qubits=2)

    aer_phi = CSFeatureMapsRetriever(cfg, observables).get_feature_maps(
        _run_aer(cfg, pubs),
        shots=100,
        seed=123,
    )
    batched, reference = _assert_channel_reference_close(cfg, pubs)
    channel_phi = CSFeatureMapsRetriever(cfg, observables).get_feature_maps(
        batched,
        shots=100,
        seed=123,
    )
    reference_phi = CSFeatureMapsRetriever(cfg, observables).get_feature_maps(
        reference,
        shots=100,
        seed=123,
    )

    np.testing.assert_array_equal(channel_phi, reference_phi)
    np.testing.assert_array_equal(channel_phi, aer_phi)


def test_reservoir_channel_cupy_matches_batched_when_available():
    cp = pytest.importorskip("cupy")
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CuPy is installed but CUDA is unavailable: {exc}")
    if device_count < 1:
        pytest.skip("No CUDA device available for CuPy reservoir-channel test.")

    cfg = RingQRConfig(input_dim=2, num_qubits=2, seed=21)
    X = np.random.default_rng(22).uniform(-0.5, 0.5, size=(3, 3, 2))
    pubs = _make_pubs(cfg, X, angle_positioning=angle_positioning_tanh, R=2, lam_0=0.37, seed=23)

    batched = _run_channel(cfg, pubs, angle_positioning_name="tanh", engine="batched")
    cupy = ExactReservoirChannelRunner(cfg, state_dtype="complex128", chunk_size=4, engine="cupy", gpu_id=0).run_pubs(
        pubs=pubs,
        angle_positioning_name="tanh",
    )

    np.testing.assert_allclose(cupy.states, batched.states, atol=1e-10, rtol=0.0)


def test_reservoir_channel_cupy_output_backend_feeds_gpu_retrievers_when_available():
    cp = pytest.importorskip("cupy")
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CuPy is installed but CUDA is unavailable: {exc}")
    if device_count < 1:
        pytest.skip("No CUDA device available for CuPy reservoir-channel test.")

    cfg = RingQRConfig(input_dim=2, num_qubits=2, seed=24)
    X = np.random.default_rng(25).uniform(-0.5, 0.5, size=(3, 3, 2))
    pubs = _make_pubs(cfg, X, angle_positioning=angle_positioning_tanh, R=2, lam_0=0.19, seed=26)
    observables = generate_k_local_paulis(locality=2, num_qubits=2)

    cpu_results = _run_channel(cfg, pubs, angle_positioning_name="tanh", engine="batched")
    gpu_results = ExactReservoirChannelRunner(
        cfg,
        state_dtype="complex128",
        chunk_size=4,
        engine="cupy",
        gpu_id=0,
        output_backend="cupy",
    ).run_pubs(pubs=pubs, angle_positioning_name="tanh")
    assert isinstance(gpu_results.states, cp.ndarray)

    cpu_phi = ExactFeatureMapsRetriever(cfg, observables, backend="numpy").get_feature_maps(cpu_results)
    gpu_phi = ExactFeatureMapsRetriever(cfg, observables, backend="cupy").get_feature_maps(gpu_results)
    assert isinstance(gpu_phi, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(gpu_phi), cpu_phi, atol=1e-10, rtol=0.0)

    cs_gpu_1 = CSFeatureMapsRetriever(cfg, observables, backend="cupy").get_feature_maps(
        gpu_results,
        shots=200,
        seed=321,
    )
    cs_gpu_2 = CSFeatureMapsRetriever(cfg, observables, backend="cupy").get_feature_maps(
        gpu_results,
        shots=200,
        seed=321,
    )
    assert isinstance(cs_gpu_1, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(cs_gpu_1), cp.asnumpy(cs_gpu_2), atol=0.0, rtol=0.0)
    assert cp.asnumpy(cp.max(cp.abs(cs_gpu_1))) <= 1.0


def test_reservoir_channel_cupy_direct_expectations_match_density_and_aer_when_available():
    cp = _require_cupy()

    cfg = RingQRConfig(input_dim=2, num_qubits=2, seed=41)
    X = np.random.default_rng(42).uniform(-0.5, 0.5, size=(2, 3, 2))
    pubs = _make_pubs(cfg, X, angle_positioning=angle_positioning_tanh, R=2, lam_0=0.23, seed=43)
    observables = generate_k_local_paulis(locality=2, num_qubits=2)

    density = ExactReservoirChannelRunner(
        cfg,
        state_dtype="complex128",
        chunk_size=4,
        engine="cupy",
        gpu_id=0,
        output_backend="cupy",
    ).run_pubs(pubs=pubs, angle_positioning_name="tanh")
    direct = ExactReservoirChannelRunner(
        cfg,
        state_dtype="complex128",
        chunk_size=4,
        engine="cupy",
        gpu_id=0,
        output_backend="cupy",
        output_kind="expectation",
    ).run_pubs(pubs=pubs, angle_positioning_name="tanh", observables=observables)

    assert isinstance(direct, ExactExpectationResults)
    assert isinstance(direct.expectations, cp.ndarray)
    assert direct.expectations.shape == (2, 2, len(observables))

    density_phi = ExactFeatureMapsRetriever(cfg, observables, backend="cupy").get_feature_maps(density)
    direct_phi = ExactFeatureMapsRetriever(cfg, observables, backend="cupy").get_feature_maps(direct)
    aer_phi = ExactFeatureMapsRetriever(cfg, observables, backend="numpy").get_feature_maps(_run_aer(cfg, pubs))

    np.testing.assert_allclose(cp.asnumpy(direct_phi), cp.asnumpy(density_phi), atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(cp.asnumpy(direct_phi), aer_phi, atol=1e-10, rtol=0.0)


def test_reservoir_channel_cupy_direct_cshadow_is_seed_deterministic_when_available():
    cp = _require_cupy()

    cfg = RingQRConfig(input_dim=2, num_qubits=2, seed=44)
    X = np.random.default_rng(45).uniform(-0.5, 0.5, size=(2, 3, 2))
    pubs = _make_pubs(cfg, X, angle_positioning=angle_positioning_tanh, R=2, lam_0=0.23, seed=46)
    observables = generate_k_local_paulis(locality=2, num_qubits=2)

    direct = ExactReservoirChannelRunner(
        cfg,
        state_dtype="complex128",
        chunk_size=4,
        engine="cupy",
        gpu_id=0,
        output_backend="cupy",
        output_kind="expectation",
    ).run_pubs(pubs=pubs, angle_positioning_name="tanh", observables=observables)

    cs_1 = CSFeatureMapsRetriever(cfg, observables, backend="cupy").get_feature_maps(
        direct,
        shots=200,
        seed=2026,
    )
    cs_2 = CSFeatureMapsRetriever(cfg, observables, backend="cupy").get_feature_maps(
        direct,
        shots=200,
        seed=2026,
    )

    assert isinstance(cs_1, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(cs_1), cp.asnumpy(cs_2), atol=0.0, rtol=0.0)
    assert cp.asnumpy(cp.max(cp.abs(cs_1))) <= 1.0


def test_reservoir_channel_cupy_direct_does_not_materialize_density(monkeypatch):
    _require_cupy()

    cfg = RingQRConfig(input_dim=2, num_qubits=2, seed=47)
    X = np.random.default_rng(48).uniform(-0.5, 0.5, size=(2, 3, 2))
    pubs = _make_pubs(cfg, X, angle_positioning=angle_positioning_tanh, R=2, lam_0=0.23, seed=49)
    observables = generate_k_local_paulis(locality=2, num_qubits=2)
    runner = ExactReservoirChannelRunner(
        cfg,
        state_dtype="complex128",
        chunk_size=4,
        engine="cupy",
        gpu_id=0,
        output_backend="cupy",
        output_kind="expectation",
    )

    def fail_density(*args, **kwargs):
        raise AssertionError("direct expectation mode must not materialize density matrices")

    monkeypatch.setattr(runner, "_density_from_ensemble_cupy", fail_density)
    direct = runner.run_pubs(pubs=pubs, angle_positioning_name="tanh", observables=observables)

    assert isinstance(direct, ExactExpectationResults)


def test_reservoir_channel_direct_validation_errors_when_available():
    _require_cupy()

    cfg = RingQRConfig(input_dim=2, num_qubits=2, seed=50)
    X = np.random.default_rng(51).uniform(-0.5, 0.5, size=(2, 3, 2))
    pubs = _make_pubs(cfg, X, angle_positioning=angle_positioning_tanh, R=2, lam_0=0.23, seed=52)

    with pytest.raises(ValueError, match="engine='cupy'"):
        ExactReservoirChannelRunner(cfg, engine="batched", output_kind="expectation")

    runner = ExactReservoirChannelRunner(cfg, engine="cupy", output_kind="expectation")
    with pytest.raises(ValueError, match="requires observables"):
        runner.run_pubs(pubs=pubs, angle_positioning_name="tanh")

    with pytest.raises(ValueError, match="single-term SparsePauliOp"):
        runner.run_pubs(
            pubs=pubs,
            angle_positioning_name="tanh",
            observables=[Operator(np.eye(4, dtype=complex))],
        )


def test_exact_retriever_rejects_mismatched_expectation_observable_count():
    cfg = RingQRConfig(input_dim=2, num_qubits=2, seed=53)
    results = ExactExpectationResults(
        expectations=np.zeros((2, 1, 2), dtype=float),
        qrc_cfg=cfg,
        observable_labels=["IZ", "ZI"],
    )

    with pytest.raises(ValueError, match="observable count mismatch"):
        ExactFeatureMapsRetriever(cfg, [SparsePauliOp("IZ")]).get_feature_maps(results)


def test_reservoir_channel_config_defaults_to_batched():
    cfg = RingQRConfig(input_dim=2, num_qubits=2, seed=16)
    node = OmegaConf.load("src/experiment/conf/model/qrc/runner/reservoir_channel.yaml")
    runner_cfg = OmegaConf.to_container(node, resolve=True)
    runner_cfg.pop("runner_kwargs", None)

    runner = instantiate(OmegaConf.create(runner_cfg), qrc_cfg=cfg)

    assert isinstance(runner, ExactReservoirChannelRunner)
    assert runner.engine == "batched"


def test_reservoir_channel_cupy_config_selects_gpu_engine():
    node = OmegaConf.load("src/experiment/conf/model/qrc/runner/reservoir_channel_cupy.yaml")
    assert node.engine == "cupy"
    assert node.gpu_id == 0
    assert node.output_backend == "cupy"


def test_reservoir_channel_cupy_direct_config_selects_expectation_output():
    node = OmegaConf.load("src/experiment/conf/model/qrc/runner/reservoir_channel_cupy_direct.yaml")
    assert node.engine == "cupy"
    assert node.gpu_id == 0
    assert node.output_backend == "cupy"
    assert node.output_kind == "expectation"


def test_reservoir_channel_featurizer_integration_uses_swap_family(monkeypatch):
    cfg = RingQRConfig(input_dim=2, num_qubits=2, seed=16)
    X = np.random.default_rng(17).uniform(-0.4, 0.4, size=(2, 2, 2))
    observables = [SparsePauliOp("IZ"), SparsePauliOp("ZI")]

    featurizer = QRCFeaturizer(
        qrc_cfg=cfg,
        runner=ExactReservoirChannelRunner(cfg, engine="batched"),
        fmp_retriever=ExactFeatureMapsRetriever(cfg, observables),
        pubs_family="ising_ring_swap",
        angle_positioning_name="linear",
        pubs_kwargs={"num_reservoirs": 2, "lam_0": 0.31, "seed": 18, "eps": 1e-8},
        runner_kwargs={},
        fmp_kwargs={},
    )

    Phi = featurizer.transform(X)

    pubs = _make_pubs(cfg, X, R=2, lam_0=0.31, seed=18)
    expected = ExactFeatureMapsRetriever(cfg, observables).get_feature_maps(_run_aer(cfg, pubs))
    np.testing.assert_allclose(Phi, expected, atol=1e-10, rtol=0.0)


def test_reservoir_channel_cupy_direct_featurizer_matches_density_when_available():
    cp = _require_cupy()

    cfg = RingQRConfig(input_dim=2, num_qubits=2, seed=54)
    X = np.random.default_rng(55).uniform(-0.4, 0.4, size=(2, 3, 2))
    observables = generate_k_local_paulis(locality=2, num_qubits=2)

    direct_featurizer = QRCFeaturizer(
        qrc_cfg=cfg,
        runner=ExactReservoirChannelRunner(
            cfg,
            engine="cupy",
            gpu_id=0,
            output_backend="cupy",
            output_kind="expectation",
        ),
        fmp_retriever=ExactFeatureMapsRetriever(cfg, observables, backend="cupy"),
        pubs_family="ising_ring_swap",
        angle_positioning_name="tanh",
        pubs_kwargs={"num_reservoirs": 2, "lam_0": 0.31, "seed": 56, "eps": 1e-8},
        runner_kwargs={},
        fmp_kwargs={},
    )

    density_featurizer = QRCFeaturizer(
        qrc_cfg=cfg,
        runner=ExactReservoirChannelRunner(
            cfg,
            engine="cupy",
            gpu_id=0,
            output_backend="cupy",
        ),
        fmp_retriever=ExactFeatureMapsRetriever(cfg, observables, backend="cupy"),
        pubs_family="ising_ring_swap",
        angle_positioning_name="tanh",
        pubs_kwargs={"num_reservoirs": 2, "lam_0": 0.31, "seed": 56, "eps": 1e-8},
        runner_kwargs={},
        fmp_kwargs={},
    )

    direct_phi = direct_featurizer.transform(X)
    density_phi = density_featurizer.transform(X)

    assert isinstance(direct_phi, cp.ndarray)
    np.testing.assert_allclose(cp.asnumpy(direct_phi), cp.asnumpy(density_phi), atol=1e-10, rtol=0.0)
