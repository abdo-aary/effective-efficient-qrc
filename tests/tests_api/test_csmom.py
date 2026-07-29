from __future__ import annotations

import numpy as np
import pytest

from src.estimators.csmom import (
    CSMoMFeatureEstimator,
    ShadowSnapshots,
    deterministic_block_ids,
    reconstruct_csmom,
)
from src.features.shadows import sample_local_pauli_shadows_density


def test_csmom_reconstruction_uses_shared_random_pauli_snapshots():
    # One complete X/Y/Z cycle has mean one for each matching +1 outcome.
    bases = np.asarray([[[[0], [1], [2], [0], [1], [2]]]], dtype=np.uint8)
    outcomes = np.ones_like(bases, dtype=np.int8)
    values = reconstruct_csmom(
        ShadowSnapshots(bases, outcomes),
        ("I", "X", "Y", "Z"),
        median_blocks=1,
    )
    np.testing.assert_allclose(values, np.ones((1, 1, 4)), atol=0.0, rtol=0.0)


def test_csmom_validation_and_near_equal_blocks():
    with pytest.raises(ValueError):
        CSMoMFeatureEstimator(snapshots=4, median_blocks=5)
    ids = deterministic_block_ids(10, 3)
    assert np.bincount(ids).tolist() == [4, 3, 3]


def test_density_shadow_sampler_converges_for_plus_state():
    plus = np.asarray([1.0, 1.0]) / np.sqrt(2.0)
    rho = np.outer(plus, plus).reshape(1, 1, 2, 2)
    rng = np.random.default_rng(9)
    bases = rng.integers(0, 3, size=(1, 1, 12000, 1), dtype=np.uint8)
    snapshots = sample_local_pauli_shadows_density(
        rho,
        bases,
        outcome_rng=np.random.default_rng(10),
    )
    values = reconstruct_csmom(
        snapshots,
        ("X", "Y", "Z"),
        median_blocks=1,
    )
    np.testing.assert_allclose(values[0, 0], [1.0, 0.0, 0.0], atol=0.05, rtol=0.0)
