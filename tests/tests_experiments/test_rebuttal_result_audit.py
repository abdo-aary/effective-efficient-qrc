from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import src.experiment.classical_baselines as cb
import src.experiment.scripts.rebuttal.run_real_quark_temporal_budget_comparison as temporal
import src.experiment.scripts.rebuttal.run_rebuttal_result_audit as audit


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
    monkeypatch.setattr(cb, "RIDGE_ALPHA_GRID", np.asarray([1e-4], dtype=float))
    monkeypatch.setattr(cb, "REG_GRID", np.asarray([1e-4, 1e-2], dtype=float))
    monkeypatch.setattr(cb, "MATERN_NU_GRID", np.asarray([0.5], dtype=float))
    monkeypatch.setattr(cb, "MATERN_XI_BOUNDS", (0.1, 10.0))
    monkeypatch.setattr(cb, "MATERN_XI_MAXITER", 2)
    monkeypatch.setattr(cb, "ESN_SPECTRAL_RADIUS_GRID", np.asarray([0.5], dtype=float))
    monkeypatch.setattr(cb, "ESN_INPUT_SCALE_GRID", np.asarray([0.2], dtype=float))
    monkeypatch.setattr(cb, "ESN_LEAK_RATE_GRID", np.asarray([0.5], dtype=float))


def _write_metric_file(path: Path, *, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "metric", "value"])
        writer.writeheader()
        writer.writerow({"split": "train", "metric": "nrmse_train_y_std", "value": f"{float(value) + 0.1:.6f}"})
        writer.writerow({"split": "test", "metric": "nrmse_train_y_std", "value": f"{float(value):.6f}"})


def test_esn_matern_saved_params_matches_direct_when_grids_are_singleton(tmp_path: Path, monkeypatch):
    _speed_up_grids(monkeypatch)
    dataset_path = _write_tiny_dataset(tmp_path / "data")
    reference_dir = _write_fake_reference(tmp_path / "reference")

    data = cb.load_benchmark_data(dataset_path)
    split = cb.load_reference_split(reference_dir)
    direct = cb.fit_esn_matern_krr(
        data.X,
        data.y2d,
        split,
        data.task_names,
        method_seed=0,
        feature_dim=8,
    )
    saved = cb.fit_esn_matern_krr_from_saved_params(
        data.X,
        data.y2d,
        split,
        data.task_names,
        method_seed=0,
        feature_dim=8,
        saved_best_params=direct["best_params"],
    )

    np.testing.assert_allclose(saved["y_test_pred"], direct["y_test_pred"], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(saved["y_train_pred"], direct["y_train_pred"], atol=1e-8, rtol=1e-8)
    assert [row["selected_spectral_radius"] for row in saved["best_params"]] == [
        row["selected_spectral_radius"] for row in direct["best_params"]
    ]
    assert [row["selected_input_scale"] for row in saved["best_params"]] == [
        row["selected_input_scale"] for row in direct["best_params"]
    ]
    assert [row["selected_leak_rate"] for row in saved["best_params"]] == [
        row["selected_leak_rate"] for row in direct["best_params"]
    ]


def test_build_final_rows_reconstructs_benchmark_from_raw_metrics(tmp_path: Path, monkeypatch):
    out_root = tmp_path / "results"
    runner = temporal.REAL_WORLD_RUNNER if hasattr(temporal, "REAL_WORLD_RUNNER") else "reservoir_channel_cupy_direct_truncated64"
    final_method = temporal.temporal_final_shadow_method(runner, 0.5, 3000)
    baseline_method = temporal.temporal_baseline_method("esn_matern_krr")
    datasets = ("dataset_a", "dataset_b")
    cards = {
        "dataset_a": {"dataset": "dataset_a", "w": 24, "d": 2, "raw_dim": 48},
        "dataset_b": {"dataset": "dataset_b", "w": 60, "d": 7, "raw_dim": 420},
    }

    monkeypatch.setattr(
        temporal,
        "build_dataset_card_rows",
        lambda data_root, datasets: [cards[str(ds)] for ds in datasets],
    )
    monkeypatch.setattr(
        temporal,
        "run_real_root",
        lambda dataset_path, out_root: (
            Path(out_root) / Path(dataset_path).name / "split=official_tser",
            Path(dataset_path).name,
            "official_tser",
        ),
    )

    dataset_values = {
        "dataset_a": {
            "quark": [0.90, 0.91, 0.92, 0.93, 0.94],
            "esn": [1.00, 1.01, 1.02, 1.03, 1.04],
        },
        "dataset_b": {
            "quark": [1.10, 1.11, 1.12, 1.13, 1.14],
            "esn": [1.00, 1.01, 1.02, 1.03, 1.04],
        },
    }

    for dataset, values in dataset_values.items():
        for seed, value in enumerate(values["quark"]):
            _write_metric_file(
                out_root
                / dataset
                / "split=official_tser"
                / final_method
                / f"seed={seed}"
                / "metrics.csv",
                value=value,
            )
        for seed, value in enumerate(values["esn"]):
            _write_metric_file(
                out_root
                / dataset
                / "split=official_tser"
                / baseline_method
                / f"seed={seed}"
                / "metrics.csv",
                value=value,
            )

    rows = temporal.build_final_rows(
        out_root=out_root,
        data_root=tmp_path / "data",
        datasets=list(datasets),
        runner=runner,
        final_seeds=[0, 1, 2, 3, 4],
        lam0=0.5,
        shots=3000,
        temporal_baseline="esn_matern_krr",
    )

    assert [row["dataset"] for row in rows] == ["dataset_a", "dataset_b"]
    assert rows[0]["num_seeds"] == 5
    assert rows[0]["delta_mean"] < 0
    assert rows[1]["delta_mean"] > 0
    assert rows[0]["quark_seed_wins"] == 5
    assert rows[1]["quark_seed_wins"] == 0


def test_recommend_global_lambda_and_shots_choose_expected_rows():
    lambda_rows = [
        {"lam0": 0.1, "num_pairs": 18, "delta_mean": 0.20, "quark_mean": 1.0},
        {"lam0": 0.5, "num_pairs": 18, "delta_mean": -0.05, "quark_mean": 0.9},
        {"lam0": 0.8, "num_pairs": 12, "delta_mean": -0.10, "quark_mean": 0.8},
    ]
    shot_rows = [
        {"shots": 1000, "mean_delta_to_exact": 0.040, "max_delta_to_exact": 0.090},
        {"shots": 3000, "mean_delta_to_exact": 0.015, "max_delta_to_exact": 0.058},
        {"shots": 5000, "mean_delta_to_exact": 0.010, "max_delta_to_exact": 0.040},
    ]

    lambda_rec = temporal.recommend_global_lambda(lambda_rows)
    shot_rec = temporal.recommend_shots(shot_rows, mean_delta_tol=0.03, max_delta_tol=0.08)

    assert lambda_rec["lam0"] == 0.5
    assert shot_rec["shots"] == 3000


def test_build_varma_architecture_table_filters_to_quark_only(tmp_path: Path, monkeypatch):
    sample_rows = [
        {
            "ablation": "architecture",
            "dataset": "varma_e2_three__N=6000__w=25__d=3__s=100",
            "w": 25,
            "d": 3,
            "task": "volterra",
            "method": "quark_reservoir_channel_cupy_direct_arch_baseline_n5_R3_k2_lam0p1",
            "method_seed": 0,
            "n_train": 5000,
            "n_test": 1000,
            "feature_dim": 315,
            "raw_dim": 75,
            "artifact_dir": str(tmp_path / "baseline"),
            "train_mse": 1e-8,
            "test_mse": 0.629388,
        },
        {
            "ablation": "architecture",
            "dataset": "varma_e2_three__N=6000__w=25__d=3__s=100",
            "w": 25,
            "d": 3,
            "task": "volterra",
            "method": "quark_reservoir_channel_cupy_direct_arch_sweep_n5_R8_k2_lam0p1",
            "method_seed": 0,
            "n_train": 5000,
            "n_test": 1000,
            "feature_dim": 840,
            "raw_dim": 75,
            "artifact_dir": str(tmp_path / "r8"),
            "train_mse": 1e-8,
            "test_mse": 0.5661,
        },
        {
            "ablation": "architecture",
            "dataset": "varma_e2_three__N=6000__w=25__d=3__s=100",
            "w": 25,
            "d": 3,
            "task": "volterra",
            "method": "esn",
            "method_seed": 0,
            "n_train": 5000,
            "n_test": 1000,
            "feature_dim": 315,
            "raw_dim": 75,
            "artifact_dir": str(tmp_path / "esn"),
            "train_mse": 1e-8,
            "test_mse": 0.01,
        },
        {
            "ablation": "varma_wd_scaling",
            "dataset": "varma_e2_three__N=6000__w=50__d=10__s=100",
            "w": 50,
            "d": 10,
            "task": "volterra",
            "method": "quark_reservoir_channel_cupy_direct_varma_grid_fixed_n5_R3_k2_lam0p1",
            "method_seed": 0,
            "n_train": 5000,
            "n_test": 1000,
            "feature_dim": 315,
            "raw_dim": 500,
            "artifact_dir": str(tmp_path / "grid"),
            "train_mse": 1e-8,
            "test_mse": 0.8,
        },
    ]

    monkeypatch.setattr(audit, "read_varma_ablation_metric_rows", lambda root: [{"ignored": True}])
    monkeypatch.setattr(audit, "build_varma_ablation_wide_rows", lambda rows, out_root, metric: sample_rows)

    rows = audit._build_varma_architecture_table(tmp_path / "varma", tmp_path / "out")

    assert len(rows) == 2
    assert all(str(row["method"]).startswith("quark_") for row in rows)
    assert all(str(row["ablation"]) == "architecture" for row in rows)
    assert all(int(row["w"]) == 25 and int(row["d"]) == 3 for row in rows)

    md_text = (tmp_path / "out" / "canonical_architecture_ablation_table.md").read_text(encoding="utf-8")
    assert "quark_reservoir_channel_cupy_direct_arch_baseline_n5_R3_k2_lam0p1" in md_text
    assert "quark_reservoir_channel_cupy_direct_arch_sweep_n5_R8_k2_lam0p1" in md_text
    assert "esn" not in md_text
