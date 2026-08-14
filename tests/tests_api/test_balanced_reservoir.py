from __future__ import annotations

import numpy as np
import pytest

from src.api import (
    AerCPUBackend,
    BalancedReservoirParameters,
    ExactFeatureEstimator,
    ExecutionSpec,
    NvidiaBackend,
    ObservableSet,
    ProjectionSpec,
    QuaRKProgram,
    ResetChannelSpec,
    ReservoirTopology,
    SeedBundle,
)
from src.backends.balanced import balanced_mixer_matrix
from src.core.factories import (
    cycle_matchings,
    gaussian_jl_projection,
    ring_topology,
    sample_balanced_reservoirs,
)


def _parameters(*, tau_plus: int, count: int = 5) -> BalancedReservoirParameters:
    return sample_balanced_reservoirs(
        num_reservoirs=count,
        topology=ring_topology(4),
        lambda_plus=np.exp(-1.0 / tau_plus),
        gamma=0.5,
        mixer_rng=np.random.default_rng(101),
        memory_rng=np.random.default_rng(202),
    )


def _program() -> QuaRKProgram:
    topology = ring_topology(3)
    parameters = sample_balanced_reservoirs(
        num_reservoirs=2,
        topology=topology,
        lambda_plus=np.exp(-1.0 / 8.0),
        gamma=0.5,
        mixer_rng=np.random.default_rng(12),
        memory_rng=np.random.default_rng(13),
    )
    return QuaRKProgram(
        projection=ProjectionSpec(
            gaussian_jl_projection(
                input_dim=1,
                encoded_width=3,
                rng=np.random.default_rng(14),
            )
        ),
        topology=topology,
        reservoirs=parameters,
        reset_channel=ResetChannelSpec(),
        observables=ObservableSet.cycle_complete(
            num_qubits=3,
            edges=topology.edges,
        ),
        window_length=3,
        angle_map="tanh",
        angle_scale=np.pi,
    )


def test_balanced_sampling_is_paired_prefix_stable_and_in_bounds():
    short = _parameters(tau_plus=8)
    long = _parameters(tau_plus=64)
    prefix = sample_balanced_reservoirs(
        num_reservoirs=3,
        topology=ring_topology(4),
        lambda_plus=np.exp(-1.0 / 8.0),
        gamma=0.5,
        mixer_rng=np.random.default_rng(101),
        memory_rng=np.random.default_rng(202),
    )

    for name in (
        "local_axes",
        "local_angles",
        "edge_axes_left",
        "edge_axes_right",
        "edge_angles",
        "matching_orders",
    ):
        np.testing.assert_array_equal(getattr(short, name), getattr(long, name))
        np.testing.assert_array_equal(getattr(prefix, name), getattr(short, name)[:3])
    assert prefix.reservoir_ids == short.reservoir_ids[:3]
    assert short.reservoir_ids == long.reservoir_ids

    np.testing.assert_allclose(np.linalg.norm(short.local_axes, axis=-1), 1.0)
    np.testing.assert_allclose(np.linalg.norm(short.edge_axes_left, axis=-1), 1.0)
    np.testing.assert_allclose(np.linalg.norm(short.edge_axes_right, axis=-1), 1.0)
    assert np.max(np.abs(short.local_angles)) <= 0.5
    assert np.max(np.abs(short.edge_angles)) <= 0.5
    assert np.all(short.reset_rates > np.exp(-1.0))
    assert np.all(short.reset_rates < np.exp(-1.0 / 8.0))
    assert np.all(long.reset_rates > short.reset_rates)
    for row in short.matching_orders:
        np.testing.assert_array_equal(np.sort(row), np.arange(2))


def test_cycle_matchings_and_complete_bank_have_locked_order():
    even = ring_topology(4)
    odd = ring_topology(5)
    assert cycle_matchings(even) == ((0, 2), (1, 3))
    assert cycle_matchings(odd) == ((0, 2), (1, 3), (4,))

    bank = ObservableSet.cycle_complete(num_qubits=4, edges=even.edges)
    assert bank.size == 48
    assert bank.labels[:12] == (
        "IIIX", "IIIY", "IIIZ",
        "IIXI", "IIYI", "IIZI",
        "IXII", "IYII", "IZII",
        "XIII", "YIII", "ZIII",
    )
    assert bank.labels[12:21] == (
        "IIXX", "IIYX", "IIZX",
        "IIXY", "IIYY", "IIZY",
        "IIXZ", "IIYZ", "IIZZ",
    )


def test_balanced_program_fingerprint_and_mixer_unitarity():
    program = _program()
    mixer = balanced_mixer_matrix(program, 0)
    identity = np.eye(mixer.shape[0])
    np.testing.assert_allclose(mixer.conj().T @ mixer, identity, atol=2e-14, rtol=0.0)

    changed = QuaRKProgram(
        projection=program.projection,
        topology=program.topology,
        reservoirs=sample_balanced_reservoirs(
            num_reservoirs=2,
            topology=program.topology,
            lambda_plus=np.exp(-1.0 / 8.0),
            gamma=0.5,
            mixer_rng=np.random.default_rng(15),
            memory_rng=np.random.default_rng(13),
        ),
        reset_channel=program.reset_channel,
        observables=program.observables,
        window_length=program.window_length,
        angle_map=program.angle_map,
        angle_scale=program.angle_scale,
    )
    assert program.fingerprint() != changed.fingerprint()


def test_balanced_aer_exact_canary():
    program = _program()
    windows = np.random.default_rng(16).uniform(-1.0, 1.0, size=(2, 3, 1))
    batch = ExactFeatureEstimator().estimate(
        program,
        windows,
        AerCPUBackend(max_qubits=3, max_windows=2),
        ExecutionSpec(seeds=SeedBundle.from_root(17)),
    )
    assert batch.shape == (2, 2, 36)
    assert batch.execution_metadata.details["channel_realization"] == (
        "dense-balanced-replacement-channel"
    )
    assert np.all(np.isfinite(batch.values))
    assert np.max(np.abs(batch.values)) <= 1.0 + 1e-12


@pytest.mark.gpu
def test_balanced_nvidia_matches_independent_aer_oracle():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(str(exc))
    program = _program()
    windows = np.random.default_rng(18).uniform(-1.0, 1.0, size=(2, 3, 1))
    execution = ExecutionSpec(seeds=SeedBundle.from_root(19), chunk_size=2)
    cpu = ExactFeatureEstimator().estimate(
        program,
        windows,
        AerCPUBackend(max_qubits=3, max_windows=2),
        execution,
    )
    gpu = ExactFeatureEstimator().estimate(
        program,
        windows,
        NvidiaBackend(gpu_id=0, chunk_size=2),
        execution,
    )
    np.testing.assert_allclose(gpu.values, cpu.values, atol=2e-10, rtol=0.0)
