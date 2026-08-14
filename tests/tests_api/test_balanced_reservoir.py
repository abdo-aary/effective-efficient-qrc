from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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
    SeedBundle,
)
from src.backends.balanced import balanced_mixer_matrix
from src.core.factories import (
    cycle_matchings,
    gaussian_jl_projection,
    ring_topology,
    sample_balanced_reservoirs,
)
from src.experiment.manifest import load_manifest
from src.experiment.numerical import ExactQuaRKRepresentationProvider


ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "experiments/empirical_evaluation/manifests/smoke.yaml"


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


@pytest.mark.gpu
def test_balanced_nvidia_rate_sweep_matches_independent_endpoint_runs():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(str(exc))

    base = _program()
    lambda_0 = float(np.exp(-1.0))
    base_lambda_plus = float(np.exp(-1.0 / 8.0))
    uniforms = np.log(base.reservoirs.reset_rates / lambda_0) / np.log(
        base_lambda_plus / lambda_0
    )
    tau_values = np.asarray((2.0, 8.0, 32.0))
    lambda_plus_values = np.exp(-1.0 / tau_values)
    rate_matrix = lambda_0 * np.power(
        lambda_plus_values[:, None] / lambda_0,
        uniforms[None, :],
    )
    programs = tuple(
        replace(
            base,
            reservoirs=replace(base.reservoirs, reset_rates=reset_rates),
        )
        for reset_rates in rate_matrix
    )
    windows = np.random.default_rng(181).uniform(-1.0, 1.0, size=(2, 3, 1))
    execution = ExecutionSpec(seeds=SeedBundle.from_root(182), chunk_size=2)
    backend = NvidiaBackend(gpu_id=0, chunk_size=2)
    estimator = ExactFeatureEstimator()
    expected = np.stack(
        [
            estimator.estimate(program, windows, backend, execution).as_numpy().values
            for program in programs
        ],
        axis=1,
    )
    result = backend.execute_exact_rate_sweep(
        programs[-1],
        windows,
        rate_matrix,
        execution,
    ).as_numpy()

    assert result.values.shape == expected.shape
    assert result.execution_metadata.details["rate_count"] == len(tau_values)
    np.testing.assert_allclose(result.values, expected, atol=2e-10, rtol=0.0)


@pytest.mark.gpu
def test_balanced_rate_sweep_shards_deterministically_across_two_gpus():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 2:
            pytest.skip("Two CUDA devices are required")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(str(exc))

    base = _program()
    lambda_0 = float(np.exp(-1.0))
    lambda_anchor = float(np.exp(-1.0 / 8.0))
    uniforms = np.log(base.reservoirs.reset_rates / lambda_0) / np.log(
        lambda_anchor / lambda_0
    )
    lambda_plus_values = np.exp(-1.0 / np.asarray((2.0, 8.0)))
    rate_matrix = lambda_0 * np.power(
        lambda_plus_values[:, None] / lambda_0,
        uniforms[None, :],
    )
    program = replace(
        base,
        reservoirs=replace(base.reservoirs, reset_rates=rate_matrix[-1]),
    )
    windows = np.random.default_rng(191).uniform(-1.0, 1.0, size=(4, 3, 1))
    execution = ExecutionSpec(seeds=SeedBundle.from_root(192), chunk_size=2)
    backends = (
        NvidiaBackend(gpu_id=0, chunk_size=2),
        NvidiaBackend(gpu_id=1, chunk_size=2),
    )
    expected = backends[0].execute_exact_rate_sweep(
        program,
        windows,
        rate_matrix,
        execution,
    ).as_numpy()
    provider = ExactQuaRKRepresentationProvider(
        manifest=load_manifest(SMOKE),
        backend=backends[0],
        backend_name="nvidia",
        gpu_id=0,
        chunk_size=2,
        nvidia_backends=backends,
    )
    values, details = provider._execute_nvidia_rate_sweep(
        program,
        windows,
        rate_matrix,
        execution,
    )

    assert details["gpu_ids"] == [0, 1]
    assert [item["window_count"] for item in details["shards"]] == [2, 2]
    np.testing.assert_allclose(values, expected.values, atol=2e-10, rtol=0.0)
