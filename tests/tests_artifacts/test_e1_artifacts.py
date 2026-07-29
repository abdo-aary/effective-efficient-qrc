from __future__ import annotations

import numpy as np

from src.artifacts.experiment import load_e1_dataset_asset, save_e1_dataset_asset
from src.core.seeds import SeedBundle
from src.data.generate.stationary_varma import generate_e1_dataset


def test_e1_dataset_asset_roundtrip_and_immutability(tmp_path):
    seeds = SeedBundle.from_root(91001)
    dataset = generate_e1_dataset(
        seeds=seeds, num_windows=8, window_length=25, stride=100
    )
    destination = save_e1_dataset_asset(tmp_path / "dataset", dataset=dataset, seeds=seeds)
    restored = load_e1_dataset_asset(destination)
    assert restored.fingerprint() == dataset.fingerprint()
    np.testing.assert_array_equal(restored.windows, dataset.windows)
    assert save_e1_dataset_asset(destination, dataset=dataset, seeds=seeds) == destination
