from __future__ import annotations

import numpy as np

from src.data.real_tser import (
    HYDRAULIC_SENSOR_SPECS,
    TSER_DATASETS,
    get_tser_spec,
    parse_tser_row,
    parse_tser_ts_file,
    prepare_real_dataset,
    prepare_tser_dataset,
)
from src.experiment.real_world_rebuttal import (
    aggregate_real_world_results,
    build_dataset_card_rows,
    eigensolve_krr_lambda_sweep,
    load_real_benchmark_data,
    load_real_split,
    resolve_real_dataset_path,
    run_cached_quark_readout_retune,
    run_real_classical_baseline,
    run_real_quark,
)
from src.experiment.scripts.rebuttal.run_real_quark_temporal_budget_comparison import (
    main as temporal_budget_main,
    recommend_global_lambda,
    simulate_shadow_feature_maps_from_exact_phi,
)


def _row(offset: float, *, dims: int = 8, length: int = 4, label: float = 0.0) -> str:
    dim_parts = []
    for d in range(dims):
        values = [offset + 0.1 * d + 0.01 * t for t in range(length)]
        if d == 0:
            values[1] = np.nan
        dim_parts.append(",".join("?" if not np.isfinite(v) else f"{v:.3f}" for v in values))
    return ":".join(dim_parts + [f"{label:.3f}"])


def _write_fixture_tser(path, *, n_rows: int, label_offset: float = 0.0) -> None:
    lines = [
        "@problemName BenzeneConcentration",
        "@timestamps false",
        "@missing true",
        "@univariate false",
        "@targetlabel true",
        "@data",
    ]
    lines.extend(_row(float(i), label=label_offset + float(i)) for i in range(n_rows))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_fixture_hydraulic(raw_root, *, n_rows: int = 6, length_seconds: int = 2) -> None:
    raw_root.mkdir(parents=True)
    for sensor_i, (sensor_name, rate) in enumerate(HYDRAULIC_SENSOR_SPECS):
        n_cols = int(rate) * int(length_seconds)
        values = np.arange(n_rows * n_cols, dtype=float).reshape(n_rows, n_cols)
        values = values + 1000.0 * sensor_i
        np.savetxt(raw_root / f"{sensor_name}.txt", values, delimiter="\t", fmt="%.6f")
    profile = np.asarray(
        [
            [100, 100, 0, 130, 0],
            [100, 90, 0, 115, 0],
            [20, 90, 1, 100, 0],
            [20, 80, 1, 90, 0],
            [3, 73, 2, 90, 1],
            [3, 73, 2, 100, 0],
        ],
        dtype=float,
    )[:n_rows]
    np.savetxt(raw_root / "profile.txt", profile, delimiter="\t", fmt="%.6f")


def test_new_triage_tser_specs_resolve_expected_zenodo_files():
    expected = {
        "manganese_concentration": ("11236670", "ManganeseConcentration"),
        "iron_concentration": ("11236615", "IronConcentration"),
        "copper_concentration": ("11235478", "CopperConcentration"),
        "gas_sensor_array_acetone": ("11235589", "GasSensorArrayAcetone"),
        "gas_sensor_array_ethanol": ("11235596", "GasSensorArrayEthanol"),
        "electric_motor_temperature": ("11235562", "ElectricMotorTemperature"),
    }

    for key, (record_id, dataset_name) in expected.items():
        assert key in TSER_DATASETS
        spec = get_tser_spec(key)
        assert spec.record_id == record_id
        assert spec.train_file == f"{dataset_name}_TRAIN.ts"
        assert spec.test_file == f"{dataset_name}_TEST.ts"


def test_parse_tser_ts_file_handles_multivariate_missing_values(tmp_path):
    ts_path = tmp_path / "tiny.ts"
    _write_fixture_tser(ts_path, n_rows=3)

    X, y, meta = parse_tser_ts_file(ts_path)

    assert X.shape == (3, 4, 8)
    assert y.shape == (3,)
    assert meta["problemname"] == "BenzeneConcentration"
    assert np.isnan(X[:, 1, 0]).all()


def test_parse_tser_row_handles_timestamped_values_with_colons():
    row = (
        "(2004-03-10 18:00:00,1.0),(2004-03-10 19:00:00,?)"
        ":"
        "(2004-03-10 18:00:00,2.0),(2004-03-10 19:00:00,3.0)"
        ":42.0"
    )

    dims, label = parse_tser_row(row)

    assert label == 42.0
    assert len(dims) == 2
    assert dims[0][0] == 1.0
    assert np.isnan(dims[0][1])
    assert dims[1] == [2.0, 3.0]


def test_prepare_tser_dataset_writes_windows_and_split_sidecars(tmp_path):
    raw_root = tmp_path / "raw"
    raw_dir = raw_root / "benzene_concentration"
    raw_dir.mkdir(parents=True)
    _write_fixture_tser(raw_dir / "BenzeneConcentration_TRAIN.ts", n_rows=5)
    _write_fixture_tser(raw_dir / "BenzeneConcentration_TEST.ts", n_rows=2, label_offset=10.0)

    out_dir = prepare_tser_dataset(
        "benzene_concentration",
        raw_root=raw_root,
        out_root=tmp_path / "data",
        download=False,
        overwrite=True,
    )
    data = load_real_benchmark_data(out_dir)
    split = load_real_split(out_dir)

    assert data.X.shape == (7, 4, 8)
    assert data.y2d.shape == (1, 7)
    assert split.train_idx.tolist() == [0, 1, 2, 3, 4]
    assert split.test_idx.tolist() == [5, 6]
    assert np.isfinite(data.X).all()


def test_dataset_card_rows_include_shape_raw_dim_and_missing_counts(tmp_path):
    raw_root = tmp_path / "raw"
    raw_dir = raw_root / "benzene_concentration"
    raw_dir.mkdir(parents=True)
    _write_fixture_tser(raw_dir / "BenzeneConcentration_TRAIN.ts", n_rows=5)
    _write_fixture_tser(raw_dir / "BenzeneConcentration_TEST.ts", n_rows=2, label_offset=10.0)

    prepare_tser_dataset(
        "benzene_concentration",
        raw_root=raw_root,
        out_root=tmp_path / "data",
        download=False,
        overwrite=True,
    )

    rows = build_dataset_card_rows(tmp_path / "data", datasets=["benzene_concentration"])

    assert len(rows) == 1
    assert rows[0]["dataset"] == "benzene_concentration"
    assert rows[0]["N"] == 7
    assert rows[0]["w"] == 4
    assert rows[0]["d"] == 8
    assert rows[0]["raw_dim"] == 32
    assert rows[0]["missing_or_imputed"] == 7


def test_prepare_hydraulic_dataset_writes_high_dimensional_windows(tmp_path):
    raw_root = tmp_path / "raw" / "hydraulic_systems"
    _write_fixture_hydraulic(raw_root, n_rows=6, length_seconds=2)

    out_dir = prepare_real_dataset(
        "hydraulic_systems",
        raw_root=raw_root,
        out_root=tmp_path / "data",
        download=False,
        overwrite=True,
        seed=0,
        hydraulic_length_seconds=2,
        hydraulic_test_ratio=0.4,
        hydraulic_split_strategy="chronological",
        hydraulic_target="hydraulic_accumulator_pressure",
        hydraulic_stable_only=True,
    )
    data = load_real_benchmark_data(out_dir)
    split = load_real_split(out_dir)

    assert data.X.shape == (5, 2, 728)
    assert data.raw_dim == 1456
    assert data.task_names == ["hydraulic_accumulator_pressure"]
    assert data.y2d.shape == (1, 5)
    assert split.train_idx.tolist() == [0, 1, 2]
    assert split.test_idx.tolist() == [3, 4]
    assert np.isfinite(data.X).all()


def test_resolve_real_dataset_path_uses_latest_pointer_when_multiple_npz_exist(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    old_path = root / "old__N=10.npz"
    new_path = root / "new__N=20.npz"
    np.savez_compressed(old_path, X=np.zeros((10, 1, 1)), y=np.zeros((1, 10)))
    np.savez_compressed(new_path, X=np.zeros((20, 1, 1)), y=np.zeros((1, 20)))
    np.savez_compressed(root / "split.npz", train_idx=np.arange(8), test_idx=np.arange(8, 10))
    (root / "latest_dataset.txt").write_text(new_path.name + "\n", encoding="utf-8")

    assert resolve_real_dataset_path(root) == new_path


def test_real_world_esn_matern_can_reuse_saved_esn_config(tmp_path):
    raw_root = tmp_path / "raw"
    raw_dir = raw_root / "benzene_concentration"
    raw_dir.mkdir(parents=True)
    _write_fixture_tser(raw_dir / "BenzeneConcentration_TRAIN.ts", n_rows=6)
    _write_fixture_tser(raw_dir / "BenzeneConcentration_TEST.ts", n_rows=3, label_offset=10.0)
    dataset_dir = prepare_tser_dataset(
        "benzene_concentration",
        raw_root=raw_root,
        out_root=tmp_path / "data",
        download=False,
        overwrite=True,
    )

    out_root = tmp_path / "results"
    esn_dir = run_real_classical_baseline(
        "esn",
        dataset_path=dataset_dir,
        out_root=out_root,
        feature_dim=4,
        backend="numpy",
    )
    esn_matern_dir = run_real_classical_baseline(
        "esn_matern_krr",
        dataset_path=dataset_dir,
        out_root=out_root,
        feature_dim=4,
        reuse_esn_source_run=esn_dir,
        backend="numpy",
    )

    assert (esn_matern_dir / "metrics.csv").exists()
    assert (esn_matern_dir / "best_params.json").exists()
    run_cfg = (esn_matern_dir / "run_config.yaml").read_text(encoding="utf-8")
    assert "reuse_esn_source_run:" in run_cfg


def test_temporal_budget_script_dry_run_prints_planned_actions(capsys, tmp_path):
    temporal_budget_main(
        [
            "--datasets",
            "copper_concentration",
            "--stages",
            "all",
            "--data-root",
            str(tmp_path / "data"),
            "--out-root",
            str(tmp_path / "results"),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr().out
    assert "run exact QuaRK datasets=copper_concentration" in captured
    assert "run ESN+Matérn datasets=copper_concentration" in captured
    assert "run cached shot sweep from best exact datasets=copper_concentration" in captured
    assert "run final fixed-configuration rerun datasets=copper_concentration" in captured
    assert "write temporal comparison aggregates" in captured


def test_temporal_budget_script_dry_run_supports_esn_matern_control(capsys, tmp_path):
    temporal_budget_main(
        [
            "--datasets",
            "copper_concentration",
            "--stages",
            "all",
            "--temporal-baseline",
            "esn_matern_krr",
            "--data-root",
            str(tmp_path / "data"),
            "--out-root",
            str(tmp_path / "results"),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr().out
    assert "run ESN+Matérn datasets=copper_concentration" in captured


def test_simulate_shadow_feature_maps_from_exact_phi_preserves_shape_and_bounds():
    exact = np.asarray(
        [
            [-1.0, -0.5, 0.0, 0.5, 1.0],
            [0.25, -0.25, 0.75, -0.75, 0.1],
        ],
        dtype=float,
    )

    approx = simulate_shadow_feature_maps_from_exact_phi(exact, shots=200, seed=0)

    assert approx.shape == exact.shape
    assert np.all(np.isfinite(approx))
    assert np.all(approx <= 1.0 + 1e-12)
    assert np.all(approx >= -1.0 - 1e-12)


def test_recommend_global_lambda_prefers_fuller_and_better_coverage():
    rows = [
        {"lam0": 0.1, "num_pairs": 6, "delta_mean": 0.10, "quark_mean": 0.80},
        {"lam0": 0.5, "num_pairs": 6, "delta_mean": 0.05, "quark_mean": 0.78},
        {"lam0": 0.8, "num_pairs": 5, "delta_mean": 0.01, "quark_mean": 0.76},
    ]

    best = recommend_global_lambda(rows)

    assert best is not None
    assert best["lam0"] == 0.5


def test_real_world_quark_runner_preserves_official_split_on_tiny_fixture(tmp_path):
    raw_root = tmp_path / "raw"
    raw_dir = raw_root / "benzene_concentration"
    raw_dir.mkdir(parents=True)
    _write_fixture_tser(raw_dir / "BenzeneConcentration_TRAIN.ts", n_rows=6)
    _write_fixture_tser(raw_dir / "BenzeneConcentration_TEST.ts", n_rows=3, label_offset=10.0)
    dataset_dir = prepare_tser_dataset(
        "benzene_concentration",
        raw_root=raw_root,
        out_root=tmp_path / "data",
        download=False,
        overwrite=True,
    )

    run_dir = run_real_quark(
        dataset_path=dataset_dir,
        out_root=tmp_path / "results",
        runner="reservoir_channel",
        method_seed=0,
        num_workers=1,
    )

    with np.load(run_dir / "model" / "arrays.npz") as z:
        assert z["train_idx"].tolist() == [0, 1, 2, 3, 4, 5]
        assert z["test_idx"].tolist() == [6, 7, 8]
    assert (run_dir / "metrics.csv").exists()


def test_real_world_quark_runner_accepts_feature_regime_overrides(tmp_path):
    raw_root = tmp_path / "raw"
    raw_dir = raw_root / "benzene_concentration"
    raw_dir.mkdir(parents=True)
    _write_fixture_tser(raw_dir / "BenzeneConcentration_TRAIN.ts", n_rows=6)
    _write_fixture_tser(raw_dir / "BenzeneConcentration_TEST.ts", n_rows=3, label_offset=10.0)
    dataset_dir = prepare_tser_dataset(
        "benzene_concentration",
        raw_root=raw_root,
        out_root=tmp_path / "data",
        download=False,
        overwrite=True,
    )

    run_dir = run_real_quark(
        dataset_path=dataset_dir,
        out_root=tmp_path / "results",
        runner="reservoir_channel",
        method_seed=0,
        num_workers=1,
        backend="numpy",
        method_suffix="exact_k1_R1_lam0p1",
        model_overrides=[
            "model/qrc/features/retriever=exact",
            "model.qrc.features.observables.locality=1",
            "model.qrc.pubs.num_reservoirs=1",
            "model.qrc.pubs.lam_0=0.1",
        ],
    )

    assert run_dir.name == "seed=0"
    assert run_dir.parent.name == "quark_reservoir_channel_exact_k1_R1_lam0p1"
    assert (run_dir / "metrics.csv").exists()
    with np.load(run_dir / "model" / "arrays.npz") as z:
        assert z["Phi_full"].shape == (9, 15)


def test_eigensolve_krr_lambda_sweep_matches_direct_solve():
    K_train = np.asarray([[1.0, 0.2], [0.2, 1.5]], dtype=float)
    K_eval_train = np.asarray([[0.5, 0.1], [0.1, 0.4]], dtype=float)
    y_train = np.asarray([1.0, -0.5], dtype=float)
    y_eval = np.asarray([0.2, -0.1], dtype=float)
    lambda_grid = np.asarray([1e-3, 1.0], dtype=float)

    sweep = eigensolve_krr_lambda_sweep(
        K_train=K_train,
        K_eval_train=K_eval_train,
        y_train=y_train,
        y_eval=y_eval,
        lambda_grid=lambda_grid,
    )

    for i, lam in enumerate(lambda_grid):
        alpha = np.linalg.solve(K_train + float(lam) * np.eye(K_train.shape[0]), y_train)
        np.testing.assert_allclose(sweep["alpha_grid"][i], alpha)
        np.testing.assert_allclose(sweep["eval_mse"][i], np.mean((K_eval_train @ alpha - y_eval) ** 2))


def test_cached_quark_readout_retune_writes_artifacts_and_aggregates(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    raw_dir = raw_root / "benzene_concentration"
    raw_dir.mkdir(parents=True)
    _write_fixture_tser(raw_dir / "BenzeneConcentration_TRAIN.ts", n_rows=8)
    _write_fixture_tser(raw_dir / "BenzeneConcentration_TEST.ts", n_rows=3, label_offset=10.0)
    dataset_dir = prepare_tser_dataset(
        "benzene_concentration",
        raw_root=raw_root,
        out_root=tmp_path / "data",
        download=False,
        overwrite=True,
    )
    data = load_real_benchmark_data(dataset_dir)
    split = load_real_split(dataset_dir)

    source_run = tmp_path / "source" / "quark_reservoir_channel" / "seed=0"
    model_dir = source_run / "model"
    model_dir.mkdir(parents=True)
    Phi = data.X.reshape(data.X.shape[0], -1)[:, :4]
    mean = Phi[split.train_idx].mean(axis=0)
    scale = Phi[split.train_idx].std(axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    np.savez_compressed(
        model_dir / "arrays.npz",
        Phi_full=Phi,
        train_idx=split.train_idx,
        test_idx=split.test_idx,
        alpha=np.zeros(split.train_idx.size),
        scaler_mean=mean,
        scaler_scale=scale,
    )
    (model_dir / "meta.json").write_text(
        """{
  "format_version": 1,
  "artifact": "QRCMaternKRRRegressor.full",
  "standardize": true,
  "test_ratio": 0.2,
  "split_seed": 0,
  "n_outputs_": 1,
  "Phi_shape": [11, 4],
  "alpha_shape": [8],
  "best_params_": {"xi": 1.0, "nu": 0.5, "reg": 1e-6}
}
""",
        encoding="utf-8",
    )

    from src.models.qrc_featurizer import QRCFeaturizer

    monkeypatch.setattr(
        QRCFeaturizer,
        "transform",
        lambda self, X: (_ for _ in ()).throw(AssertionError("retune must not re-featurize")),
    )

    out_root = tmp_path / "results"
    run_dir = run_cached_quark_readout_retune(
        dataset_path=dataset_dir,
        source_run=source_run,
        out_root=out_root,
        val_ratio=0.25,
        split_strategy="chronological",
        lambda_grid=np.asarray([1e-3, 1e-1, 1e1]),
    )

    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "best_params.json").exists()
    assert (run_dir / "predictions.npz").exists()
    assert (run_dir / "readout_retune_curve.csv").exists()
    with np.load(run_dir / "readout_retune_indices.npz") as z:
        assert np.intersect1d(z["val_idx"], split.test_idx).size == 0
        assert set(z["val_idx"].tolist()).issubset(set(split.train_idx.tolist()))

    import json

    params = json.loads((run_dir / "best_params.json").read_text(encoding="utf-8"))
    assert params[0]["selected_lambda"] in [1e-3, 1e-1, 1e1]

    aggregate_dir = aggregate_real_world_results(out_root, response_dir=tmp_path / "responses")
    table = (aggregate_dir / "real_world_table.md").read_text(encoding="utf-8")
    assert "quark_reservoir_channel_readout_retune" in table

    kernel_run_dir = run_cached_quark_readout_retune(
        dataset_path=dataset_dir,
        source_run=source_run,
        out_root=out_root,
        val_ratio=0.25,
        split_strategy="chronological",
        lambda_grid=np.asarray([1e-3, 1e-1]),
        xi_grid=np.asarray([0.5, 1.0]),
        nu_grid=np.asarray([0.5, 1.5]),
        tune_max_inner=4,
        tune_max_val=2,
    )
    kernel_params = json.loads((kernel_run_dir / "best_params.json").read_text(encoding="utf-8"))
    assert kernel_params[0]["selected_xi"] in [0.5, 1.0]
    assert kernel_params[0]["selected_nu"] in [0.5, 1.5]
    curve_header = (kernel_run_dir / "readout_retune_curve.csv").read_text(encoding="utf-8").splitlines()[0]
    assert curve_header == "task,xi,nu,lambda,train_inner_mse,val_mse,selected"
