from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import src.experiment.classical_baselines as cb
from src.compute.backend import cupy_available


def _write_tiny_dataset(root: Path, *, N: int = 36, w: int = 5, d: int = 3, L: int = 3) -> Path:
    rng = np.random.default_rng(123)
    root.mkdir(parents=True, exist_ok=True)
    X = rng.normal(size=(N, w, d))
    flat = X.reshape(N, -1)
    y = np.vstack(
        [
            flat[:, 0] + 0.1 * flat[:, 1],
            np.tanh(flat[:, :4]).sum(axis=1),
            flat[:, 2] * flat[:, 3] + 0.2 * flat[:, 4],
        ][:L]
    )
    np.savez_compressed(root / "toy.npz", X=X, y=y)
    (root / "toy.meta.json").write_text(
        json.dumps(
            {
                "N": N,
                "w": w,
                "d": d,
                "L": L,
                "s": 7,
                "process_kind": "varma",
                "functionals_kind": "e2_three" if L == 3 else "toy",
                "seed": 123,
            }
        ),
        encoding="utf-8",
    )
    return root


def _write_fake_reference(root: Path, *, N: int = 36, D: int = 8, L: int = 3) -> Path:
    rng = np.random.default_rng(321)
    root.mkdir(parents=True, exist_ok=True)
    train_idx = np.arange(0, 24)
    test_idx = np.arange(24, N)
    Phi = rng.normal(size=(N, D))
    np.savez_compressed(
        root / "arrays.npz",
        Phi_full=Phi,
        train_idx=train_idx,
        test_idx=test_idx,
        alpha=np.zeros((L, train_idx.size)),
        scaler_mean=np.zeros(D),
        scaler_scale=np.ones(D),
    )
    (root / "meta.json").write_text(
        json.dumps(
            {
                "best_params_": [
                    {"xi": 1.0, "nu": 0.5, "reg": 1e-6},
                    {"xi": 1.0, "nu": 1.5, "reg": 1e-6},
                    {"xi": 1.0, "nu": 0.5, "reg": 1e-6},
                ][:L]
            }
        ),
        encoding="utf-8",
    )
    return root


def _speed_up_grids(monkeypatch):
    monkeypatch.setattr(cb, "RIDGE_ALPHA_GRID", np.asarray([1e-4, 1e-2], dtype=float))
    monkeypatch.setattr(cb, "REG_GRID", np.asarray([1e-4, 1e-2], dtype=float))
    monkeypatch.setattr(cb, "RFF_GAMMA_GRID", np.asarray([1e-2], dtype=float))
    monkeypatch.setattr(cb, "MATERN_NU_GRID", np.asarray([0.5], dtype=float))
    monkeypatch.setattr(cb, "MATERN_XI_BOUNDS", (0.1, 10.0))
    monkeypatch.setattr(cb, "MATERN_XI_MAXITER", 2)
    monkeypatch.setattr(cb, "ESN_SPECTRAL_RADIUS_GRID", np.asarray([0.5], dtype=float))
    monkeypatch.setattr(cb, "ESN_INPUT_SCALE_GRID", np.asarray([0.2], dtype=float))
    monkeypatch.setattr(cb, "ESN_LEAK_RATE_GRID", np.asarray([0.5], dtype=float))


def test_classical_baseline_shapes_and_split_loading(tmp_path: Path, monkeypatch):
    _speed_up_grids(monkeypatch)
    dataset_path = _write_tiny_dataset(tmp_path / "data")
    reference_dir = _write_fake_reference(tmp_path / "reference")

    data = cb.load_benchmark_data(dataset_path)
    split = cb.load_reference_split(reference_dir)
    raw, _ = cb.standardize_flattened_windows(data.X, split.train_idx)
    matched, _ = cb.make_matched_random_features(raw, raw_dim=data.raw_dim, feature_dim=8, seed=0)
    rff = cb.make_rff_features(raw, gamma=0.01, feature_dim=8, seed=0)
    esn = cb.make_esn_features(
        data.X,
        train_idx=split.train_idx,
        feature_dim=8,
        seed=0,
        spectral_radius=0.5,
        input_scale=0.2,
        leak_rate=0.5,
    )

    assert split.train_idx.shape == (24,)
    assert split.test_idx.shape == (12,)
    assert data.task_names == ["one_step_forecast", "exp_fading_linear", "volterra"]
    assert raw.shape == (36, 15)
    assert matched.shape == (36, 8)
    assert rff.shape == (36, 8)
    assert esn.shape == (36, 8)


def test_run_all_classical_baselines_and_aggregate(tmp_path: Path, monkeypatch):
    _speed_up_grids(monkeypatch)
    monkeypatch.setattr(cb, "PROJECT_ROOT", tmp_path)

    try:
        import src.models.qrc_featurizer as qrc_featurizer

        monkeypatch.setattr(
            qrc_featurizer.QRCFeaturizer,
            "transform",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("classical-only tests must not run QRCFeaturizer.transform")
            ),
            raising=False,
        )
    except Exception:
        pass

    dataset_path = _write_tiny_dataset(tmp_path / "data")
    reference_dir = _write_fake_reference(tmp_path / "reference")
    out_dir = tmp_path / "results"

    for method in (
        "raw_matern_krr",
        "raw_ridge",
        "matched_random_features_matern_krr",
        "rff_ridge",
        "esn",
    ):
        run_dir = cb.run_classical_baseline(
            method,
            dataset_path=dataset_path,
            reference_model_dir=reference_dir,
            out_dir=out_dir,
            method_seed=0,
            feature_dim=8,
        )
        assert (run_dir / "metrics.csv").exists()
        assert (run_dir / "predictions.npz").exists()
        assert (run_dir / "best_params.json").exists()
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "run_config.yaml").exists()

    aggregate_dir = cb.aggregate_classical_baselines(
        out_dir,
        dataset_path=dataset_path,
        reference_model_dir=reference_dir,
        include_quark_reference=True,
    )

    long_path = aggregate_dir / "classical_baselines_long.csv"
    wide_path = aggregate_dir / "classical_baselines_wide.csv"
    table_path = aggregate_dir / "classical_baselines_table.md"
    assert long_path.exists()
    assert wide_path.exists()
    assert table_path.exists()

    with long_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    methods = {row["method"] for row in rows}
    assert methods == {
        "raw_matern_krr",
        "raw_ridge",
        "matched_random_features_matern_krr",
        "rff_ridge",
        "esn",
        "quark_reference",
    }
    assert set(rows[0]) == set(cb.METRICS_COLUMNS)
    assert {row["n_train"] for row in rows} == {"24"}
    assert {row["n_test"] for row in rows} == {"12"}
    assert all(row["metric"] == "mse" for row in rows)
    assert all(row["split"] in {"train", "test"} for row in rows)

    response_dir = tmp_path / "docs/rebutal/responses"
    assert (response_dir / "global_view.md").exists()
    assert "matched_random_features_matern_krr" in (response_dir / "MSDS.md").read_text(
        encoding="utf-8"
    )


def test_gpu_classical_baselines_match_cpu_for_core_methods(tmp_path: Path, monkeypatch):
    if not cupy_available():
        return
    _speed_up_grids(monkeypatch)
    dataset_path = _write_tiny_dataset(tmp_path / "data")
    reference_dir = _write_fake_reference(tmp_path / "reference")

    for method in ("raw_ridge", "raw_matern_krr"):
        cpu_dir = cb.run_classical_baseline(
            method,
            dataset_path=dataset_path,
            reference_model_dir=reference_dir,
            out_dir=tmp_path / "cpu",
            method_seed=0,
            feature_dim=8,
            backend="numpy",
        )
        gpu_dir = cb.run_classical_baseline(
            method,
            dataset_path=dataset_path,
            reference_model_dir=reference_dir,
            out_dir=tmp_path / "gpu",
            method_seed=0,
            feature_dim=8,
            backend="cupy",
            device=0,
        )
        with np.load(cpu_dir / "predictions.npz") as cpu, np.load(gpu_dir / "predictions.npz") as gpu:
            np.testing.assert_allclose(gpu["y_test_pred"], cpu["y_test_pred"], atol=1e-8, rtol=1e-8)
            np.testing.assert_allclose(gpu["test_mse"], cpu["test_mse"], atol=1e-8, rtol=1e-8)


def test_gpu_matern_grid_filters_unsupported_nu():
    grid = cb.matern_nu_grid_for_backend("cupy", np.asarray([0.5, 1.5, 2.5, 5.0]))

    np.testing.assert_allclose(grid, np.asarray([0.5, 1.5, 2.5]))
