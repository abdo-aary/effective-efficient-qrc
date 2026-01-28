from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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
        feats = np.stack([m, s, m ** 2, s ** 2, np.sin(m), np.cos(m), np.tanh(s)], axis=1).astype(float)
        return feats[:, : self.D]


def _fast_grid_tuner(Phi: np.ndarray, y: np.ndarray, **kwargs):
    reg = float(kwargs.get("reg", 1e-6))
    xi = 0.8 + abs(float(np.mean(y)))
    nu = 1.5
    return {"xi": xi, "nu": nu, "reg": reg}, 0.0


def _cfg_for_saving_dataset(tmp_path: Path):
    """Minimal config structure required by save_dataset."""
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "seed": 0,
            "sampling": {"N": 40, "w": 5, "d": 3, "s": 10},
            "process": {"kind": "unit"},
            "functionals": {"kind": "unit", "items": []},
            "output": {
                "save_dir": str(tmp_path),
                "name": "exp_ds",
                "format": "npz",
                "overwrite": True,
                "save_meta": True,
                "save_config": False,
            },
        }
    )


def test_experiment_from_paths_fit_and_save_model(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(mk, "tune_matern_grid_train_val", _fast_grid_tuner, raising=True)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5, 3)).astype(float)
    z = X.mean(axis=(1, 2))
    y = np.stack([np.sin(z), np.cos(1.5 * z)], axis=0).astype(float)  # (L,N)

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
    assert exp.dataset_artifact is not None
    assert exp.dataset_artifact.root == art.root

    exp.fit(num_workers=1)
    yhat_ref = exp.model.predict()
    assert featurizer.calls == 1  # features computed once
    assert exp.model.Phi_full_ is not None

    metrics = exp.compute_metrics()
    # When functionals aren't instantiated, names are meta-based
    assert set(metrics.keys()) == {"unit[0]", "unit[1]"}
    assert metrics["unit[0]"] == {}

    model_dir = tmp_path / "model_artifact"
    exp.save_model(model_dir)
    assert (model_dir / "arrays.npz").exists()
    assert (model_dir / "meta.json").exists()

    # --- load_model roundtrip (no featurizer rerun) ---
    # Build a fresh Experiment (with a dummy model) and load the persisted model into it.
    dummy_model = mk.QRCMaternKRRRegressor(
        CountingFeaturizer(D=6),
        standardize=True,
        test_ratio=0.25,
        split_seed=123,
        tuning={"strategy": "grid", "val_ratio": 0.2, "seed": 0, "reg": 1e-6, "nu_grid": (1.5,), "xi_maxiter": 3},
    )
    exp2 = Experiment.from_paths(dataset_path=art.root, model=dummy_model, instantiate_functionals=False)
    loaded = exp2.load_model(model_dir, featurizer=None)

    assert loaded.X_train_features_ is not None
    assert loaded.best_params_ == exp.model.best_params_
    assert np.allclose(loaded.alpha_, exp.model.alpha_, atol=0.0, rtol=0.0)

    Phi_te = loaded.Phi_full_[loaded.test_idx_]
    yhat_loaded = loaded.predict_from_features(Phi_te, apply_scaler=True)
    assert np.allclose(yhat_loaded, yhat_ref, atol=1e-10)


def test_experiment_can_construct_model_from_cfg_via_from_config(tmp_path: Path, monkeypatch):
    """Unit-test the config-based branch without building actual QRC objects."""

    # Save a tiny dataset
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 4, 2)).astype(float)
    y = np.sin(X.mean(axis=(1, 2))).astype(float)
    ds = WindowsDataset(X=X, y=y, label_functionals=[], meta={"functionals_kind": "unit"})

    from omegaconf import OmegaConf

    cfg_save = OmegaConf.create(
        {
            "seed": 0,
            "sampling": {"N": 20, "w": 4, "d": 2, "s": 10},
            "process": {"kind": "unit"},
            "functionals": {"kind": "unit", "items": []},
            "output": {
                "save_dir": str(tmp_path),
                "name": "ds_cfg_branch",
                "format": "npz",
                "overwrite": True,
                "save_meta": True,
                "save_config": False,
            },
        }
    )
    art = save_dataset(ds, cfg_save)

    # Patch from_config to return a known fast model
    monkeypatch.setattr(mk, "tune_matern_grid_train_val", _fast_grid_tuner, raising=True)

    def _fake_from_config(_cfg):
        return mk.QRCMaternKRRRegressor(
            CountingFeaturizer(D=6),
            standardize=False,
            test_ratio=0.2,
            split_seed=0,
            tuning={"strategy": "grid", "val_ratio": 0.2, "seed": 0, "reg": 1e-6, "nu_grid": (1.5,), "xi_maxiter": 3},
        )

    monkeypatch.setattr(mk.QRCMaternKRRRegressor, "from_config", staticmethod(_fake_from_config), raising=True)

    dummy_model_cfg = OmegaConf.create({"model": {"dummy": True}})
    exp = Experiment.from_paths(dataset_path=art.data_path, model_cfg=dummy_model_cfg)

    assert isinstance(exp.model, mk.QRCMaternKRRRegressor)
    exp.fit(num_workers=1)
    yhat_ref = exp.model.predict()
    # sanity check that the model is trained
    assert exp.model.alpha_ is not None
