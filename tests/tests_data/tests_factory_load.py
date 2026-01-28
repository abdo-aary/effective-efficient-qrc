from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.data.factory import load_dataset_artifact, load_windows_dataset, save_dataset
from src.data.generate.base import WindowsDataset


def _base_cfg(tmp_path: Path, *, name: str, fmt: str, save_meta: bool = True, save_config: bool = False):
    """Small in-memory config sufficient for save_dataset/load roundtrips."""
    return OmegaConf.create(
        {
            "seed": 0,
            "sampling": {"N": 8, "w": 4, "d": 3, "s": 10},
            "process": {"kind": "unit"},
            # only needed if save_config=True and we want instantiate_functionals=True
            "functionals": {
                "kind": "unit",
                "items": [
                    {"_target_": "src.data.label.functionals.OneStepForecastFunctional"},
                    {"_target_": "src.data.label.functionals.ExpFadingLinearFunctional", "alpha": 0.9},
                ],
            },
            "output": {
                "save_dir": str(tmp_path),
                "name": name,
                "format": fmt,
                "overwrite": True,
                "save_meta": bool(save_meta),
                "save_config": bool(save_config),
            },
        }
    )


@pytest.mark.parametrize("fmt", ["npz", "npy"])
def test_load_windows_dataset_roundtrip_arrays(tmp_path: Path, fmt: str):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 4, 3)).astype(float)
    y = rng.normal(size=(2, 8)).astype(float)
    ds = WindowsDataset(X=X, y=y, label_functionals=[], meta={"functionals_kind": "unit"})

    cfg = _base_cfg(tmp_path, name=f"ds_{fmt}", fmt=fmt, save_meta=True, save_config=False)
    art = save_dataset(ds, cfg)

    # Provide file path
    ds2, art2 = load_windows_dataset(art.data_path, instantiate_functionals=True)
    assert art2.data_path.exists()
    assert np.allclose(ds2.X, X)
    assert np.allclose(ds2.y, y)
    assert ds2.label_functionals == []
    assert ds2.meta.get("functionals_kind") == "unit"

    # Provide directory path (should infer the unique dataset file)
    ds3, art3 = load_windows_dataset(art.root, instantiate_functionals=False)
    assert art3.root == art.root
    assert np.allclose(ds3.X, X)
    assert np.allclose(ds3.y, y)


def test_load_dataset_artifact_raises_if_multiple_candidates(tmp_path: Path):
    # Create two candidate files under the same directory
    (tmp_path / "a.npz").write_bytes(b"dummy")
    (tmp_path / "b.npz").write_bytes(b"dummy")

    with pytest.raises(ValueError):
        load_dataset_artifact(tmp_path)


def test_load_windows_dataset_can_instantiate_functionals_from_saved_config(tmp_path: Path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 4, 3)).astype(float)
    y = rng.normal(size=(2, 8)).astype(float)
    ds = WindowsDataset(X=X, y=y, label_functionals=[], meta={"functionals_kind": "unit"})

    cfg = _base_cfg(tmp_path, name="ds_with_cfg", fmt="npz", save_meta=True, save_config=True)
    art = save_dataset(ds, cfg)

    # Ensure sidecar config exists
    assert art.config_path is not None and art.config_path.exists()

    ds2, _ = load_windows_dataset(art.root, instantiate_functionals=True)
    assert len(ds2.label_functionals) == 2
    assert ds2.label_functionals[0].__class__.__name__ == "OneStepForecastFunctional"
    assert ds2.label_functionals[1].__class__.__name__ == "ExpFadingLinearFunctional"

    # Meta roundtrip
    assert ds2.meta.get("functionals_kind") == "unit"
    # And meta JSON exists and is parseable
    meta = json.loads((tmp_path / "ds_with_cfg.meta.json").read_text(encoding="utf-8"))
    assert meta["functionals_kind"] == "unit"
