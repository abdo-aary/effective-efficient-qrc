from __future__ import annotations

import numpy as np
import pytest

from src.core.seeds import SeedBundle
from src.data.generate.stationary_varma import (
    StationaryVARMASpec,
    generate_e1_dataset,
    spectral_radius,
)


def test_stationary_e1_varma_is_deterministic_and_protocol_aligned():
    seeds = SeedBundle.from_root(91001)
    first = generate_e1_dataset(
        seeds=seeds, num_windows=16, window_length=25, stride=100
    )
    second = generate_e1_dataset(
        seeds=seeds, num_windows=16, window_length=25, stride=100
    )
    np.testing.assert_array_equal(first.windows, second.windows)
    np.testing.assert_array_equal(first.targets, second.targets)
    assert first.fingerprint() == second.fingerprint()
    assert first.windows.shape == (16, 25, 3)
    assert first.targets.shape == (16, 8)
    np.testing.assert_array_equal(np.diff(first.prediction_origins), 100)
    np.testing.assert_array_equal(
        first.context.future_indices, first.prediction_origins + 1
    )
    assert abs(spectral_radius(first.transition) - 0.7) < 1e-12
    assert first.metadata["lyapunov_relative_residual"] < 1e-10
    assert np.linalg.eigvalsh(first.stationary_covariance).min() > -1e-10
    assert np.max(np.abs(first.windows)) <= 1.0
    assert np.max(np.abs(first.targets)) <= 1.0


def test_stationary_varma_spec_rejects_semantic_config_drift():
    config = {
        "process": "stationary_varma",
        "d": 3,
        "p": 3,
        "q": 3,
        "innovation_covariance": "identity",
        "observation_scale": 1.0,
        "target_companion_radius": 0.7,
        "ar_draw": "gaussian_unit_spectral_weighted",
        "ma_amplitude": 0.5,
        "ma_decay": 0.5,
        "stationary_initialization": "lyapunov",
        "burn_in": 0,
    }
    assert StationaryVARMASpec.from_mapping(config) == StationaryVARMASpec()
    config["burn_in"] = 100
    with pytest.raises(ValueError, match="data.burn_in"):
        StationaryVARMASpec.from_mapping(config)
