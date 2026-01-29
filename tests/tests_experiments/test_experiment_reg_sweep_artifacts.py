from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import pytest
from omegaconf import OmegaConf

import src.models.qrc_matern_krr as mk
from src.data.factory import save_dataset
from src.data.generate.base import WindowsDataset
from src.experiment.experiment import Experiment


class CountingFeaturizer:
    """Deterministic featurizer with a call counter (fast tests; no Qiskit)."""

    def __init__(self, D: int = 6):
        self.D = int(D)
        self.calls = 0

    def transform(self, X: np.ndarray) -> np.ndarray:
        self.calls += 1
        X = np.asarray(X)
        m = X.mean(axis=(1, 2))
        s = X.std(axis=(1, 2))
        feats = np.stack([m, s, m**2, s**2, np.sin(m), np.cos(m), np.tanh(s)], axis=1).astype(float)
        return feats[:, : self.D]


def _fast_grid_tuner(Phi: np.ndarray, y: np.ndarray, **kwargs):
    reg = float(kwargs.get("reg", 1e-6))
    xi = 0.75 + abs(float(np.mean(y)))
    nu = 1.5
    return {"xi": xi, "nu": nu, "reg": reg}, 0.0


def _cfg_for_saving_dataset(tmp_path: Path):
    return OmegaConf.create(
        {
            "seed": 0,
            "sampling": {"N": 40, "w": 5, "d": 3, "s": 10},
            "process": {"kind": "unit"},
            "functionals": {"kind": "unit", "items": []},
            "output": {
                "save_dir": str(tmp_path),
                "name": "exp_reg_sweep_artifacts",
                "format": "npz",
                "overwrite": True,
                "save_meta": True,
                "save_config": False,
            },
        }
    )


def test_reg_sweep_artifacts_csv_and_plots_created(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(mk, "tune_matern_grid_train_val", _fast_grid_tuner, raising=True)

    # Multi-output (L=3) to test multiple curves.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5, 3)).astype(float)
    z = X.mean(axis=(1, 2))
    y = np.stack([np.sin(z), np.cos(1.5 * z), z**2], axis=0).astype(float)  # (L, N)

    ds = WindowsDataset(X=X, y=y, label_functionals=[], meta={"functionals_kind": "unit"})
    art = save_dataset(ds, _cfg_for_saving_dataset(tmp_path))

    featurizer = CountingFeaturizer(D=6)
    model = mk.QRCMaternKRRRegressor(
        featurizer,
        standardize=True,
        test_ratio=0.25,
        split_seed=123,
        tuning={"strategy": "grid", "val_ratio": 0.2, "seed": 0, "reg": 1e-6, "nu_grid": (1.5,), "xi_maxiter": 3},
    )

    exp = Experiment.from_paths(dataset_path=art.root, model=model, instantiate_functionals=False)
    exp.fit(num_workers=1)

    reg_grid = [1e-6, 1e-4, 1e-2]
    exp.run_reg_sweep(reg_grid)

    # Build table
    rows = exp.build_reg_sweep_metrics_table()
    assert isinstance(rows, list)
    assert len(rows) == 2 * 3 * len(reg_grid)  # (train+test) * L * R
    assert set(r["split"] for r in rows) == {"train", "test"}

    # Save artifacts
    out_dir = tmp_path / "artifacts"
    artifacts = exp.save_reg_sweep_artifacts(out_dir, formats=("png",))
    assert artifacts["csv"].exists()
    assert artifacts["train_plot_png"].exists()
    assert artifacts["test_plot_png"].exists()

    # CSV row count check (header + rows)
    with artifacts["csv"].open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)
    assert len(csv_rows) == len(rows)

    # Ensure saving artifacts does not re-call the featurizer
    assert featurizer.calls == 1
