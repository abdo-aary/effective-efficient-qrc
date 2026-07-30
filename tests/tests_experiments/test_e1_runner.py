from __future__ import annotations

import csv
import json

from src.experiment.e1 import aggregate_e1, campaign_plan
from src.experiment.protocol import ChronologicalSplit, E1ProtocolSpec


def test_full_campaign_plan_has_exactly_five_feature_jobs():
    cfg = {
        "protocol_version": "quark-empirical-v1",
        "experiment_contract": "quark-e1-functional-v1",
        "root": 1101,
        "structural_roots": [1101, 1102, 1103, 1104, 1105],
        "data": {"N": 6000, "window_length": 25},
        "program": {"num_qubits": 5, "num_reservoirs": 3, "locality": 2},
    }
    rows = campaign_plan(cfg)
    assert len(rows) == 5
    assert {row["root"] for row in rows} == {1101, 1102, 1103, 1104, 1105}
    assert all(row["feature_shape"] == [6000, 3, 105] for row in rows)
    assert all(row["status"] == "planned-not-run" for row in rows)


def test_configuration_hash_is_stage_neutral_but_protocol_sensitive():
    spec = E1ProtocolSpec(
        protocol_version="quark-empirical-v1",
        experiment_contract="quark-e1-functional-v1",
        scenario="reference-varma-functional",
        method="quark-exact",
        mode="smoke",
        root=91001,
        num_windows=96,
        window_length=25,
        stride=100,
        num_qubits=3,
        num_reservoirs=2,
        locality=1,
        reset_rate=0.5,
        chunk_size=128,
        task_names=("future", "volterra"),
        split=ChronologicalSplit(64, 80, 96),
        prefixes=(40, 80),
        nu_grid=(1.5, 5.0),
        xi_grid=(0.5, 1.0, 2.0),
        lambda_grid=(1e-4, 1e-2, 1.0),
        path_lambda_grid=(1e-4, 1e-2, 1.0),
        tie_tolerance=1e-12,
        path_nu=2.5,
        path_xi=1.0,
        engine="cupy",
    )
    first = spec.configuration_hash(
        dataset_hash="dataset",
        program_fingerprint="program",
        resolved_configuration={
            "stage": "features",
            "allow_full": True,
            "artifacts": {"root": "artifacts/old"},
            "readout": {"path_nu": 2.5},
        },
    )
    analysis = spec.configuration_hash(
        dataset_hash="dataset",
        program_fingerprint="program",
        resolved_configuration={
            "stage": "analyze",
            "allow_full": False,
            "artifacts": {"root": "storage/artifacts/new"},
            "readout": {"path_nu": 2.5},
        },
    )
    changed_protocol = spec.configuration_hash(
        dataset_hash="dataset",
        program_fingerprint="program",
        resolved_configuration={
            "stage": "features",
            "allow_full": True,
            "readout": {"path_nu": 1.5},
        },
    )
    assert first == analysis
    assert first != changed_protocol


def test_aggregate_e1_filters_smoke_from_full_runs(tmp_path):
    method_root = (
        tmp_path
        / "E1"
        / "reference-varma-functional"
        / "quark-exact"
    )
    for root, mode in ((91001, "smoke"), (1101, "full")):
        run = method_root / f"root={root}" / f"hash-{mode}"
        run.mkdir(parents=True)
        documents = {
            "status.json": {"status": "complete"},
            "resolved_config.json": {"mode": mode},
            "manifest.json": {
                "git_commit": "commit",
                "program_fingerprint": f"program-{mode}",
            },
            "execution.json": {
                "feature_shape": [96, 2, 9] if mode == "smoke" else [6000, 3, 105]
            },
            "metrics.json": {"rows": [{"task": "future", "test_nrmse": 1.0}]},
        }
        for filename, document in documents.items():
            (run / filename).write_text(json.dumps(document), encoding="utf-8")

    outputs = aggregate_e1(artifact_root=tmp_path, mode="full")
    with outputs["run_index"].open(newline="", encoding="utf-8") as stream:
        run_rows = list(csv.DictReader(stream))
    with outputs["metrics"].open(newline="", encoding="utf-8") as stream:
        metric_rows = list(csv.DictReader(stream))

    assert [row["root"] for row in run_rows] == ["1101"]
    assert [row["mode"] for row in run_rows] == ["full"]
    assert [row["root"] for row in metric_rows] == ["1101"]
    assert [row["mode"] for row in metric_rows] == ["full"]

