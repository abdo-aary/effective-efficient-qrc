from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import src.models.qrc_matern_krr as mk
from src.data.generate.base import WindowsDataset
from src.experiment.experiment import Experiment


# -----------------------
# Helpers
# -----------------------
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
    """Fast deterministic tuner stub to avoid slow xi/nu search in tests."""
    reg = float(kwargs.get("reg", 1e-6))
    xi = 0.7 + abs(float(np.mean(y)))
    nu = 1.5
    return {"xi": xi, "nu": nu, "reg": reg}, 0.0


def _make_single_output_dataset(seed: int = 0, N: int = 50):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, 6, 3)).astype(float)
    z = X.mean(axis=(1, 2))
    y = (np.sin(z) + 0.05 * rng.normal(size=N)).astype(float)  # (N,)
    ds = WindowsDataset(X=X, y=y, label_functionals=[], meta={"functionals_kind": "unit"})
    return ds


def _make_multi_output_dataset(seed: int = 0, N: int = 60, L: int = 3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, 6, 3)).astype(float)
    z = X.mean(axis=(1, 2))
    ys = []
    for l in range(L):
        ys.append(np.sin((l + 1) * z) + 0.02 * rng.normal(size=N))
    y = np.stack(ys, axis=0).astype(float)  # (L, N)
    ds = WindowsDataset(X=X, y=y, label_functionals=[], meta={"functionals_kind": "unit"})
    return ds


def _make_model(featurizer: CountingFeaturizer, *, split_seed: int = 0):
    return mk.QRCMaternKRRRegressor(
        featurizer,
        standardize=True,
        test_ratio=0.25,
        split_seed=split_seed,
        tuning={
            "strategy": "grid",
            "val_ratio": 0.2,
            "seed": 0,
            "reg": 1e-6,
            "nu_grid": (1.5,),
            "xi_maxiter": 3,
        },
    )


def _ensure_sweep(exp: Experiment, reg_grid):
    """Compatible with either Experiment.run_reg_sweep or manual assignment."""
    if hasattr(exp, "run_reg_sweep"):
        return exp.run_reg_sweep(reg_grid)
    out = exp.model.sweep_regularization(reg_grid)
    exp.reg_sweep_ = out
    return out


# -----------------------
# Tests
# -----------------------
def test_reg_sweep_artifacts_single_output_row_count_and_files(tmp_path: Path, monkeypatch):
    """Single-output guardrail: ensure (R,) MSE sweeps are handled and files are created."""
    monkeypatch.setattr(mk, "tune_matern_grid_train_val", _fast_grid_tuner, raising=True)

    ds = _make_single_output_dataset(seed=1, N=55)
    f = CountingFeaturizer(D=6)
    model = _make_model(f, split_seed=123)
    exp = Experiment(dataset=ds, model=model)

    exp.fit(num_workers=1)
    assert f.calls == 1

    reg_grid = [1e-8, 1e-6, 1e-4, 1e-2]
    _ensure_sweep(exp, reg_grid)

    # 1) Metrics table contract
    rows = exp.build_reg_sweep_metrics_table()
    assert len(rows) == 2 * 1 * len(reg_grid)  # train+test × L=1 × R
    for r in rows[:5]:
        assert set(r.keys()) == {"functional", "split", "lambda_reg", "mse"}
        assert r["split"] in {"train", "test"}

    # 2) Artifacts created
    out_dir = tmp_path / "artifacts_single"
    artifacts = exp.save_reg_sweep_artifacts(out_dir, formats=("png",))

    assert artifacts["csv"].exists()
    assert any(k.endswith("train_plot_png") and p.exists() for k, p in artifacts.items())
    assert any(k.endswith("test_plot_png") and p.exists() for k, p in artifacts.items())

    # Plotting should not re-featurize
    assert f.calls == 1


def test_reg_sweep_metrics_uses_label_functional_class_names(monkeypatch):
    """Naming guardrail: if label_functionals exist, their class names must be used."""
    monkeypatch.setattr(mk, "tune_matern_grid_train_val", _fast_grid_tuner, raising=True)

    class F0: ...
    class F1: ...
    class F2: ...

    ds = _make_multi_output_dataset(seed=2, N=50, L=3)
    ds.label_functionals = [F0(), F1(), F2()]  # must match L

    f = CountingFeaturizer(D=6)
    model = _make_model(f, split_seed=0)
    exp = Experiment(dataset=ds, model=model)
    exp.fit(num_workers=1)

    reg_grid = [1e-6, 1e-4]
    _ensure_sweep(exp, reg_grid)

    rows = exp.build_reg_sweep_metrics_table()
    names = {r["functional"] for r in rows}
    assert names == {"F0", "F1", "F2"}


def test_build_reg_sweep_metrics_table_raises_when_missing_sweep(tmp_path: Path):
    """Failure-mode guardrail: clear error if sweep was not run."""
    ds = _make_single_output_dataset(seed=3, N=20)
    exp = Experiment(dataset=ds, model=object())  # model unused for this failure check

    with pytest.raises(RuntimeError, match="Run `run_reg_sweep"):
        _ = exp.build_reg_sweep_metrics_table()


def test_save_reg_sweep_artifacts_raises_when_missing_sweep(tmp_path: Path):
    """Failure-mode guardrail: clear error if sweep was not run."""
    ds = _make_single_output_dataset(seed=4, N=20)
    exp = Experiment(dataset=ds, model=object())  # model unused for this failure check

    with pytest.raises(RuntimeError, match="Run `run_reg_sweep"):
        _ = exp.save_reg_sweep_artifacts(tmp_path / "artifacts")
