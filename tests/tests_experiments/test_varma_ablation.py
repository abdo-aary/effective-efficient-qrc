from __future__ import annotations

import csv

import numpy as np
import pytest

from src.experiment.temporal_baselines import METRICS_COLUMNS
from src.experiment.scripts.rebuttal.run_varma_ablation_suite import main as varma_ablation_main
from src.experiment.varma_ablation import (
    DEFAULT_VARMA_ABLATION_RESULTS_ROOT,
    VarmaDatasetSpec,
    aggregate_varma_ablation_results,
    architecture_regimes,
    feature_dim_for_local_paulis,
    prepare_varma_ablation_dataset,
    read_varma_ablation_metric_rows,
    varma_grid_regime,
)
from src.backends.aer.circuits import CircuitFactory
from src.core.legacy_config import RingQRConfig
from src.backends.qiskit_utils import angle_positioning_tanh


def test_prepare_varma_ablation_dataset_writes_split_and_metadata(tmp_path):
    spec = VarmaDatasetSpec(w=4, d=3, N_train=16, N_test=8, s=2, seed=123, split_seed=7)

    out_dir = prepare_varma_ablation_dataset(spec, data_root=tmp_path / "data", overwrite=True)

    assert (out_dir / "split.npz").exists()
    assert (out_dir / "latest_dataset.txt").exists()
    with np.load(out_dir / "split.npz") as z:
        train_idx = z["train_idx"]
        test_idx = z["test_idx"]
    assert train_idx.size == 16
    assert test_idx.size == 8
    assert np.intersect1d(train_idx, test_idx).size == 0

    meta_path = out_dir / f"{spec.artifact_name}.meta.json"
    meta = meta_path.read_text(encoding="utf-8")
    assert '"n_train": 16' in meta
    assert '"n_test": 8' in meta
    assert "one_step_forecast" in meta


def test_varma_ablation_dry_run_contains_architecture_and_24_grid_points(capsys, tmp_path):
    varma_ablation_main(
        [
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
    assert "prepare architecture_shape=w25,d3" in captured
    assert "varma_grid_points=24" in captured
    assert "run architecture regimes=" in captured
    assert "run fixed QuaRK n=5,R=3,k=2,lam0=0.1" in captured
    assert "w100_d500" in captured


def test_varma_grid_regime_uses_fixed_paper_feature_dimension():
    regime = varma_grid_regime()

    assert regime.feature_dim == 315
    assert "model.qrc.cfg.num_qubits=5" in regime.overrides
    assert "model.qrc.pubs.num_reservoirs=3" in regime.overrides
    assert "model.qrc.features.observables.locality=2" in regime.overrides
    assert feature_dim_for_local_paulis(num_qubits=5, num_reservoirs=3, locality=2) == 315


def test_architecture_regimes_are_one_factor_around_baseline():
    regimes = {regime.name: regime for regime in architecture_regimes()}

    assert "arch_baseline_n5_R3_k2_lam0p1" in regimes
    assert "arch_no_jl_identity_pad_n5_R3_k2_lam0p1" in regimes
    assert "arch_zero_dynamics_n5_R3_k2_lam0p1" in regimes
    assert "model.qrc.cfg.projection_mode=identity_pad" in regimes[
        "arch_no_jl_identity_pad_n5_R3_k2_lam0p1"
    ].overrides
    assert "model.qrc.pubs.dynamics_mode=zero" in regimes[
        "arch_zero_dynamics_n5_R3_k2_lam0p1"
    ].overrides
    assert "model.qrc.cfg.num_qubits=10" in regimes["arch_sweep_n10_R3_k2_lam0p1"].overrides
    assert "model.qrc.pubs.num_reservoirs=8" in regimes["arch_sweep_n5_R8_k2_lam0p1"].overrides
    assert "model.qrc.features.observables.locality=3" in regimes["arch_sweep_n5_R3_k3_lam0p1"].overrides


def test_identity_pad_projection_and_zero_dynamics_guardrails():
    cfg = RingQRConfig(input_dim=3, num_qubits=5, seed=0, projection_mode="identity_pad")

    np.testing.assert_allclose(cfg.projection[:, :3], np.eye(3))
    np.testing.assert_allclose(cfg.projection[:, 3:], 0.0)
    with pytest.raises(ValueError, match="identity_pad"):
        RingQRConfig(input_dim=6, num_qubits=5, seed=0, projection_mode="identity_pad")

    X = np.ones((2, 3, 3), dtype=float)
    pubs = CircuitFactory.create_pubs_dataset_reservoirs_IsingRingSWAP(
        qrc_cfg=cfg,
        angle_positioning=angle_positioning_tanh,
        X=X,
        num_reservoirs=2,
        lam_0=0.1,
        seed=0,
        dynamics_mode="zero",
    )
    vals = pubs[0][1]
    n = cfg.num_qubits
    reservoir_params = vals[:, :, -(len(cfg.topology.edges) + 2 * n + 1) :]
    np.testing.assert_allclose(reservoir_params[:, :, :-1], 0.0)
    np.testing.assert_allclose(reservoir_params[:, 0, -1], 0.1)


def test_varma_ablation_aggregation_separates_architecture_and_scaling(tmp_path):
    out_root = tmp_path / "results"
    rows = [
        _metric_row(
            out_root,
            ablation="architecture",
            dataset="varma_e2_three__N=96__w=25__d=3__s=2",
            method="quark_reservoir_channel_cupy_direct_arch_baseline_n5_R3_k2_lam0p1",
            task="volterra",
            value=0.3,
        ),
        _metric_row(
            out_root,
            ablation="varma_wd_scaling",
            dataset="varma_e2_three__N=96__w=10__d=10__s=2",
            method="quark_reservoir_channel_cupy_direct_varma_grid_fixed_n5_R3_k2_lam0p1",
            task="volterra",
            value=0.2,
        ),
    ]
    for row in rows:
        run_dir = out_root / row["artifact_dir"]
        run_dir.mkdir(parents=True)
        with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=METRICS_COLUMNS)
            writer.writeheader()
            writer.writerow({col: row.get(col, "") for col in METRICS_COLUMNS})

    aggregate_dir = aggregate_varma_ablation_results(out_root)

    assert (aggregate_dir / "architecture_ablation_table.md").exists()
    assert (aggregate_dir / "varma_wd_scaling_table.md").exists()
    assert (aggregate_dir / "ablation_long.csv").exists()
    assert len(read_varma_ablation_metric_rows(out_root)) == 2
    assert "quark_reservoir_channel_cupy_direct_arch_baseline_n5_R3_k2_lam0p1" in (
        aggregate_dir / "architecture_ablation_table.md"
    ).read_text(encoding="utf-8")
    assert "w=10" not in (aggregate_dir / "architecture_ablation_table.md").read_text(encoding="utf-8")
    assert "quark_reservoir" in (aggregate_dir / "varma_wd_scaling_table.md").read_text(encoding="utf-8")


def _metric_row(out_root, *, ablation: str, dataset: str, method: str, task: str, value: float) -> dict[str, str]:
    artifact_rel = f"{ablation}/{dataset}/split=fixture/{method}/seed=0"
    return {
        "experiment_id": f"{dataset}__fixture__{method}",
        "method": method,
        "dataset": dataset,
        "task": task,
        "split": "test",
        "metric": "mse",
        "value": str(float(value)),
        "n_train": "64",
        "n_test": "32",
        "feature_dim": "315",
        "raw_dim": "750",
        "method_seed": "0",
        "split_source": "fixture",
        "artifact_dir": str(out_root / artifact_rel),
    }
