from __future__ import annotations

import numpy as np
import pytest

from src.backends.aer import AerCPUBackend
from src.backends.ibm import IBMRuntimeBackend
from src.backends.nvidia import NvidiaBackend
from src.core import (
    ExecutionSpec,
    ObservableSet,
    ProjectionSpec,
    QuaRKProgram,
    ResetChannelSpec,
    ReservoirParameters,
    ReservoirTopology,
    SeedBundle,
    UnsupportedEstimatorError,
)
from src.estimators import CSMoMFeatureEstimator, ExactFeatureEstimator


def make_program() -> QuaRKProgram:
    return QuaRKProgram(
        projection=ProjectionSpec(np.eye(2), "identity"),
        topology=ReservoirTopology(2, ((0, 1),), "chain"),
        reservoirs=ReservoirParameters(
            zz=np.asarray([[0.31], [-0.27]]),
            x_fields=np.asarray([[0.2, -0.4], [0.5, 0.1]]),
            z_fields=np.asarray([[-0.1, 0.3], [0.4, -0.2]]),
            reset_rates=np.asarray([0.2, 0.73]),
        ),
        reset_channel=ResetChannelSpec(),
        observables=ObservableSet(("IX", "IY", "IZ", "XI", "XX", "ZZ")),
        window_length=3,
        angle_map="tanh",
    )


def test_aer_exact_returns_structured_features():
    pytest.importorskip("qiskit_aer")
    program = make_program()
    windows = np.random.default_rng(3).normal(size=(2, 3, 2))
    batch = ExactFeatureEstimator().estimate(
        program,
        windows,
        AerCPUBackend(max_qubits=3, max_windows=4),
        ExecutionSpec(seeds=SeedBundle.from_root(4)),
    )
    assert batch.shape == (2, 2, 6)
    assert batch.execution_metadata.exact is True
    assert np.max(np.abs(batch.values)) <= 1.0 + 1e-10


def test_ibm_rejects_exact_before_backend_access():
    backend = IBMRuntimeBackend(backend_name="must-not-be-accessed")
    assert backend.capabilities.mid_circuit_reset is False
    assert backend.capabilities.noise_models is False
    with pytest.raises(UnsupportedEstimatorError):
        backend.compile(make_program(), ExactFeatureEstimator())


def test_backend_capabilities_are_explicit_about_reset_and_noise():
    assert AerCPUBackend().capabilities.mid_circuit_reset is True
    assert AerCPUBackend().capabilities.noise_models is False
    assert NvidiaBackend().capabilities.mid_circuit_reset is False
    assert NvidiaBackend().capabilities.noise_models is False


def test_ibm_local_csmom_smoke():
    pytest.importorskip("qiskit_aer")
    program = make_program()
    windows = np.zeros((1, 3, 2))
    estimator = CSMoMFeatureEstimator(60, 5, retain_raw_snapshots=True)
    batch = estimator.estimate(
        program,
        windows,
        IBMRuntimeBackend(backend_name="aer-local", local_simulator=True),
        ExecutionSpec(seeds=SeedBundle.from_root(8)),
    )
    assert batch.shape == (1, 2, 6)
    assert batch.raw_snapshots.bases.shape == (1, 2, 60, 2)
    assert batch.raw_snapshots.trajectory_suffix_starts.shape == (1, 2, 60)
    assert batch.raw_snapshots.trajectory_reset_counts.shape == (1, 2, 60)
    assert batch.execution_metadata.exact is False


def test_nvidia_exact_matches_independent_aer_oracle():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(str(exc))
    pytest.importorskip("qiskit_aer")
    program = make_program()
    windows = np.random.default_rng(11).normal(size=(2, 3, 2))
    execution = ExecutionSpec(seeds=SeedBundle.from_root(12), chunk_size=2)
    cpu = ExactFeatureEstimator().estimate(
        program, windows, AerCPUBackend(max_qubits=3, max_windows=4), execution
    )
    gpu = ExactFeatureEstimator().estimate(
        program, windows, NvidiaBackend(gpu_id=0, chunk_size=2), execution
    )
    np.testing.assert_allclose(gpu.values, cpu.values, atol=1e-9, rtol=0.0)


def test_nvidia_csmom_is_chunk_invariant():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(str(exc))
    program = make_program()
    windows = np.random.default_rng(21).normal(size=(2, 3, 2))
    estimator = CSMoMFeatureEstimator(90, 5, retain_raw_snapshots=True)
    backend = NvidiaBackend(gpu_id=0, chunk_size=4)
    first = estimator.estimate(
        program,
        windows,
        backend,
        ExecutionSpec(seeds=SeedBundle.from_root(22), chunk_size=1),
    )
    second = estimator.estimate(
        program,
        windows,
        backend,
        ExecutionSpec(seeds=SeedBundle.from_root(22), chunk_size=4),
    )
    np.testing.assert_array_equal(first.raw_snapshots.bases, second.raw_snapshots.bases)
    np.testing.assert_array_equal(first.raw_snapshots.outcomes, second.raw_snapshots.outcomes)
    np.testing.assert_array_equal(first.values, second.values)


def test_ibm_local_trajectory_csmom_converges_to_exact_oracle():
    pytest.importorskip("qiskit_aer")
    program = QuaRKProgram(
        projection=ProjectionSpec(np.eye(1), "identity"),
        topology=ReservoirTopology(1, (), "single"),
        reservoirs=ReservoirParameters(
            np.zeros((1, 0)),
            np.asarray([[0.37]]),
            np.asarray([[-0.21]]),
            np.asarray([0.63]),
        ),
        reset_channel=ResetChannelSpec(),
        observables=ObservableSet(("X", "Y", "Z")),
        window_length=3,
        angle_map="tanh",
    )
    windows = np.asarray([[[0.2], [-0.4], [0.7]]])
    execution = ExecutionSpec(seeds=SeedBundle.from_root(31))
    exact = ExactFeatureEstimator().estimate(
        program,
        windows,
        AerCPUBackend(max_qubits=2, max_windows=2),
        execution,
    )
    sampled = CSMoMFeatureEstimator(6000, 1).estimate(
        program,
        windows,
        IBMRuntimeBackend(backend_name="aer-local", local_simulator=True),
        execution,
    )
    np.testing.assert_allclose(sampled.values, exact.values, atol=0.08, rtol=0.0)
