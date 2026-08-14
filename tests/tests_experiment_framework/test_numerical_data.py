from __future__ import annotations

from pathlib import Path

import numpy as np

from src.experiment.domain import DataSpec
from src.experiment.manifest import load_manifest
from src.experiment.numerical import ControlledLegendreDataProvider
from src.experiment.seeding import PathSeedTree


ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "experiments/empirical_evaluation/manifests/smoke.yaml"
TASKS = tuple(f"F_mem_{lag}" for lag in (0, 1, 2, 4, 8, 16, 32, 64))


def _spec(*, split: str = "train", sample_count: int = 128, rho_half_life=None):
    return DataSpec(
        id=f"memory/data/{split}",
        study_id="memory_vs_lag",
        split=split,
        trajectory_id="memory/trajectory",
        task_ids=TASKS,
        sample_count=sample_count,
        window_length=80,
        gap=3,
        input_dim=1,
        pairing_key="pair",
        dependence_half_life=rho_half_life,
    )


def test_memory_data_uses_exact_origins_labels_and_path_reproducibility():
    provider = ControlledLegendreDataProvider(load_manifest(SMOKE))
    seeds = PathSeedTree.from_root(314)
    first = provider.prepare(_spec(sample_count=16), seeds)
    second = provider.prepare(_spec(sample_count=16), seeds)
    trajectory = first.assets["trajectory"]
    origins = first.assets["label_times"]
    targets = first.assets["targets"]

    np.testing.assert_array_equal(origins, 79 + np.arange(16) * 83)
    expected = np.stack(
        [
            0.5 * (3.0 * trajectory[origins - lag, 0] ** 2 - 1.0)
            for lag in (0, 1, 2, 4, 8, 16, 32, 64)
        ],
        axis=1,
    )
    np.testing.assert_array_equal(targets, expected)
    np.testing.assert_array_equal(first.assets["trajectory"], second.assets["trajectory"])
    np.testing.assert_array_equal(first.assets["targets"], second.assets["targets"])


def test_stationary_uniform_marginals_ar_dependence_and_split_independence():
    provider = ControlledLegendreDataProvider(load_manifest(SMOKE))
    seeds = PathSeedTree.from_root(2718)
    dependent = provider.prepare(
        _spec(sample_count=1000, rho_half_life=4.0), seeds
    ).assets["trajectory"][:, 0]
    expected_rho = 2.0 ** (-1.0 / 4.0)
    assert abs(float(np.mean(dependent))) < 0.02
    assert abs(float(np.var(dependent)) - 1.0 / 3.0) < 0.02
    # The Gaussian copula preserves positive dependence but not Pearson rho exactly.
    assert np.corrcoef(dependent[:-1], dependent[1:])[0, 1] > 0.75
    assert expected_rho > 0.8

    train = provider.prepare(_spec(split="train", sample_count=32), seeds)
    test = provider.prepare(_spec(split="test", sample_count=32), seeds)
    assert not np.array_equal(train.assets["trajectory"], test.assets["trajectory"])
