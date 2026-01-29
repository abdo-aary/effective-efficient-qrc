from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest
from omegaconf import OmegaConf

# Script under test
from src.experiment.scripts.run_reg_sweep_experiment import _save_sweep_npz, main as script_main


# ---------------------------------------------------------------------
# Unit-ish tests (no real dataset/model; test orchestration + IO)
# ---------------------------------------------------------------------

@dataclass
class _DummyExperiment:
    dataset_path: Path
    model_cfg: dict
    instantiate_functionals: bool

    fit_called: int = 0
    save_model_called: int = 0
    run_reg_sweep_called: int = 0
    save_artifacts_called: int = 0

    reg_sweep_: Optional[Dict[str, Any]] = None

    @classmethod
    def from_paths(cls, *, dataset_path, model_cfg, instantiate_functionals=True, **kwargs):
        return cls(
            dataset_path=Path(dataset_path),
            model_cfg=dict(model_cfg) if model_cfg is not None else {},
            instantiate_functionals=bool(instantiate_functionals),
        )

    def fit(self, *, num_workers: int = 1, **kwargs):
        self.fit_called += 1
        return self

    def save_model(self, path):
        self.save_model_called += 1
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        # mimic model persistence artifact presence
        (p / "meta.json").write_text("{}", encoding="utf-8")
        (p / "arrays.npz").write_bytes(b"dummy")
        return None

    def run_reg_sweep(self, reg_grid, **kwargs):
        self.run_reg_sweep_called += 1
        reg_grid = np.asarray(reg_grid, dtype=float).reshape(-1)
        out = {
            "reg_grid": reg_grid,
            "mse_train": np.linspace(1.0, 0.1, reg_grid.size),
            "mse_test": np.linspace(1.2, 0.2, reg_grid.size),
        }
        self.reg_sweep_ = out
        return out

    def save_reg_sweep_artifacts(self, out_dir, formats=("pdf",), **kwargs):
        self.save_artifacts_called += 1
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        # create “expected-like” outputs
        csv_path = out_dir / "reg_sweep_metrics.csv"
        csv_path.write_text("functional,split,lambda_reg,mse\n", encoding="utf-8")
        artifacts["csv"] = csv_path
        for fmt in formats:
            fmt = fmt.lower().lstrip(".")
            tp = out_dir / f"reg_sweep_train.{fmt}"
            vp = out_dir / f"reg_sweep_test.{fmt}"
            tp.write_bytes(b"train")
            vp.write_bytes(b"test")
            artifacts[f"train_plot_{fmt}"] = tp
            artifacts[f"test_plot_{fmt}"] = vp
        return artifacts


def test__save_sweep_npz_roundtrip(tmp_path: Path):
    sweep = {
        "reg_grid": [1e-9, 1e-6, 1e-3],
        "mse_train": [1.0, 0.5, 0.25],
        "mse_test": [1.2, 0.6, 0.3],
        "rkhs_norm": [0.1, 0.2, 0.3],
    }
    p = tmp_path / "sweep.npz"
    _save_sweep_npz(sweep, p)

    assert p.exists()
    z = np.load(p)
    assert np.allclose(z["reg_grid"], np.asarray(sweep["reg_grid"], dtype=float))
    assert np.allclose(z["mse_train"], np.asarray(sweep["mse_train"], dtype=float))
    assert np.allclose(z["mse_test"], np.asarray(sweep["mse_test"], dtype=float))
    assert np.allclose(z["rkhs_norm"], np.asarray(sweep["rkhs_norm"], dtype=float))


def test_main_requires_dataset_path():
    cfg = OmegaConf.create(
        {
            "dataset_path": "???",
            "instantiate_functionals": False,
            "num_workers": 1,
            "model": {},
            "reg_sweep": {"enabled": False, "reg_grid": [1e-6]},
            "output": {"model_dir": None, "sweep_path": None, "artifacts_dir": None},
        }
    )
    with pytest.raises(ValueError, match="cfg.dataset_path is required"):
        script_main.__wrapped__(cfg)


def test_main_orchestrates_fit_save_sweep_artifacts_with_dummy_experiment(tmp_path: Path, monkeypatch):
    # Patch the Experiment symbol imported in the script module
    import src.experiment.scripts.run_reg_sweep_experiment as mod

    monkeypatch.setattr(mod, "Experiment", _DummyExperiment, raising=True)

    model_dir = tmp_path / "model"
    sweep_path = tmp_path / "reg_sweep.npz"
    artifacts_dir = tmp_path / "artifacts"

    cfg = OmegaConf.create(
        {
            "dataset_path": str(tmp_path / "dataset_artifact"),
            "instantiate_functionals": False,
            "num_workers": 1,
            "model": {"anything": "ok"},
            "reg_sweep": {"enabled": True, "reg_grid": [1e-9, 1e-6, 1e-3]},
            "output": {
                "model_dir": str(model_dir),
                "sweep_path": str(sweep_path),
                "artifacts_dir": str(artifacts_dir),
                "plot_formats": ["pdf"],
            },
        }
    )

    script_main.__wrapped__(cfg)

    # Check model persistence
    assert (model_dir / "meta.json").exists()
    assert (model_dir / "arrays.npz").exists()

    # Check sweep npz
    assert sweep_path.exists()
    z = np.load(sweep_path)
    assert "reg_grid" in z and "mse_train" in z and "mse_test" in z
    assert z["reg_grid"].shape == (3,)

    # Check artifacts
    assert (artifacts_dir / "reg_sweep_metrics.csv").exists()
    assert (artifacts_dir / "reg_sweep_train.pdf").exists()
    assert (artifacts_dir / "reg_sweep_test.pdf").exists()


def test_main_stops_after_fit_when_reg_sweep_disabled(tmp_path: Path, monkeypatch):
    import src.experiment.scripts.run_reg_sweep_experiment as mod

    class _DummyNoSweep(_DummyExperiment):
        def run_reg_sweep(self, *a, **k):
            raise AssertionError("run_reg_sweep should not be called when reg_sweep.enabled=false")

    monkeypatch.setattr(mod, "Experiment", _DummyNoSweep, raising=True)

    cfg = OmegaConf.create(
        {
            "dataset_path": str(tmp_path / "dataset_artifact"),
            "instantiate_functionals": False,
            "num_workers": 1,
            "model": {},
            "reg_sweep": {"enabled": False, "reg_grid": [1e-6]},
            "output": {
                "model_dir": str(tmp_path / "model"),
                "sweep_path": str(tmp_path / "reg_sweep.npz"),
                "artifacts_dir": str(tmp_path / "artifacts"),
                "plot_formats": ["pdf"],
            },
        }
    )

    script_main.__wrapped__(cfg)

    # Should have saved model (because model_dir provided) but no sweep/artifacts
    assert (tmp_path / "model").exists()
    assert not (tmp_path / "reg_sweep.npz").exists()
    assert not (tmp_path / "artifacts").exists()


# ---------------------------------------------------------------------
# Integration / end-to-end (fast): real dataset artifact + real Experiment,
# but monkeypatch heavy model-building + slow tuner.
# ---------------------------------------------------------------------

class CountingFeaturizer:
    """Small deterministic featurizer used to keep the test fast."""
    def __init__(self, D: int = 8, seed: int = 0):
        self.D = int(D)
        self.seed = int(seed)
        self.calls = 0
        self._W = None

    def transform(self, X, num_workers: int = 1):
        self.calls += 1
        X = np.asarray(X, dtype=float)
        N = X.shape[0]
        flat = X.reshape(N, -1)
        if self._W is None:
            rng = np.random.default_rng(self.seed)
            self._W = rng.normal(size=(flat.shape[1], self.D))
        return flat @ self._W


def _fast_grid_tuner(Phi_tr, y_tr, **kwargs):
    # Minimal return contract expected by QRCMaternKRRRegressor.fit():
    reg = float(kwargs.get("reg", 1e-6))
    best = {"xi": 1.0, "nu": 1.5, "reg": reg}
    return best, {}  # second output is usually diagnostics (unused in our path)


def test_run_reg_sweep_script_end_to_end_small_dataset(tmp_path: Path, monkeypatch):
    # --- Prepare a REAL dataset artifact on disk ---
    from src.data.factory import save_dataset
    from src.data.generate.base import WindowsDataset

    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 5, 3)).astype(float)  # (N,w,d)
    z = X.mean(axis=(1, 2))
    y = np.stack([np.sin(z), np.cos(1.25 * z)], axis=0).astype(float)  # (L=2, N)

    ds = WindowsDataset(X=X, y=y, label_functionals=[], meta={"functionals_kind": "unit"})

    save_cfg = OmegaConf.create(
        {
            "output": {
                "save_dir": str(tmp_path / "dataset"),
                "name": "toy",
                "format": "npz",
                "overwrite": True,
                "save_meta": True,
                "save_config": False,
            }
        }
    )
    art = save_dataset(ds, save_cfg)

    # --- Patch slow tuner ---
    import src.models.kernel as kernel_mod
    monkeypatch.setattr(kernel_mod, "tune_matern_grid_train_val", _fast_grid_tuner, raising=True)

    # --- Patch model construction from config to use our lightweight featurizer ---
    import src.models.qrc_matern_krr as qrc_mod

    holder: dict[str, Any] = {}

    def _from_config(cls, cfg):
        feat = CountingFeaturizer(D=10, seed=123)
        holder["featurizer"] = feat
        model = qrc_mod.QRCMaternKRRRegressor(
            feat,
            standardize=True,
            test_ratio=0.25,
            split_seed=123,
            tuning={
                "strategy": "grid",
                "val_ratio": 0.2,
                "seed": 0,
                "reg": 1e-6,
                "nu_grid": (1.5,),
                "xi_maxiter": 2,
            },
        )
        holder["model"] = model
        return model

    monkeypatch.setattr(qrc_mod.QRCMaternKRRRegressor, "from_config", classmethod(_from_config), raising=True)

    # --- Run script main (in-process), bypassing Hydra wrapper ---
    model_dir = tmp_path / "out_model"
    sweep_path = tmp_path / "out_reg_sweep.npz"
    artifacts_dir = tmp_path / "out_artifacts"

    cfg = OmegaConf.create(
        {
            "dataset_path": str(art.root),  # directory containing toy.npz
            "instantiate_functionals": False,
            "num_workers": 1,
            "model": {},  # ignored by our monkeypatched from_config
            "reg_sweep": {"enabled": True, "reg_grid": [1e-9, 1e-6, 1e-3]},
            "output": {
                "model_dir": str(model_dir),
                "sweep_path": str(sweep_path),
                "artifacts_dir": str(artifacts_dir),
                "plot_formats": ["pdf"],
            },
        }
    )

    script_main.__wrapped__(cfg)

    # --- Assertions: model persisted ---
    assert (model_dir / "meta.json").exists()
    assert (model_dir / "arrays.npz").exists()

    # --- Assertions: sweep npz ---
    assert sweep_path.exists()
    z = np.load(sweep_path)
    assert z["reg_grid"].shape == (3,)
    assert "mse_train" in z and "mse_test" in z

    # --- Assertions: artifacts exist (produced by Experiment.save_reg_sweep_artifacts) ---
    csv_path = artifacts_dir / "reg_sweep_metrics.csv"
    train_pdf = artifacts_dir / "reg_sweep_train.pdf"
    test_pdf = artifacts_dir / "reg_sweep_test.pdf"
    assert csv_path.exists()
    assert train_pdf.exists()
    assert test_pdf.exists()

    # CSV should contain: 2 splits * L functionals * R regs rows (+ header)
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].strip() == "functional,split,lambda_reg,mse"
    L, R = 2, 3
    assert len(lines) == 1 + 2 * L * R

    # --- Assertions: no extra featurizer transform calls during sweep ---
    # fit should call transform once; sweep should reuse cached features
    assert holder["featurizer"].calls == 1
