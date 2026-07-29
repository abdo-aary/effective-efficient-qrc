from __future__ import annotations

import numpy as np
import pytest

from src.core import (
    BackendKind,
    EstimatorKind,
    ExecutionMetadata,
    FeatureBatch,
    ObservableSet,
    ProjectionSpec,
    QuaRKProgram,
    ResetChannelSpec,
    ReservoirParameters,
    ReservoirTopology,
    SeedBundle,
)


def make_program(*, window_length: int = 2) -> QuaRKProgram:
    return QuaRKProgram(
        projection=ProjectionSpec(np.eye(2), mode="identity"),
        topology=ReservoirTopology(2, ((0, 1),), "chain"),
        reservoirs=ReservoirParameters(
            zz=np.asarray([[0.2], [-0.3]]),
            x_fields=np.asarray([[0.1, 0.2], [0.3, 0.4]]),
            z_fields=np.asarray([[0.4, 0.3], [0.2, 0.1]]),
            reset_rates=np.asarray([0.1, 0.8]),
        ),
        reset_channel=ResetChannelSpec(),
        observables=ObservableSet(("IX", "ZI", "XX")),
        window_length=window_length,
        angle_map="tanh",
    )


def test_seed_bundle_roundtrip_and_named_isolation():
    bundle = SeedBundle.from_root(12345)
    restored = SeedBundle.from_dict(bundle.to_dict())
    assert restored == bundle
    np.testing.assert_array_equal(
        bundle.generator("shadow_bases").integers(0, 100, size=20),
        restored.generator("shadow_bases").integers(0, 100, size=20),
    )
    assert not np.array_equal(
        bundle.generator("shadow_bases").integers(0, 100, size=20),
        bundle.generator("reset_trajectories").integers(0, 100, size=20),
    )


def test_program_is_immutable_and_fingerprint_tracks_arrays():
    program = make_program()
    with pytest.raises(ValueError):
        program.projection.matrix[0, 0] = 2.0
    changed = QuaRKProgram(
        projection=ProjectionSpec(2.0 * np.eye(2), mode="identity"),
        topology=program.topology,
        reservoirs=program.reservoirs,
        reset_channel=program.reset_channel,
        observables=program.observables,
        window_length=program.window_length,
        angle_map=program.angle_map,
    )
    assert changed.fingerprint() != program.fingerprint()


def test_feature_batch_flattens_reservoir_major_observable_minor():
    values = np.arange(12).reshape(2, 2, 3)
    batch = FeatureBatch(
        values=values,
        estimator_kind=EstimatorKind.EXACT,
        estimator_metadata={},
        execution_metadata=ExecutionMetadata(
            backend_kind=BackendKind.AER,
            estimator_kind=EstimatorKind.EXACT,
            exact=True,
            program_fingerprint="test",
        ),
        observable_labels=("IX", "ZI", "XX"),
        reservoir_ids=("r0", "r1"),
    )
    np.testing.assert_array_equal(
        batch.flatten_for_readout(),
        np.asarray([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]),
    )
