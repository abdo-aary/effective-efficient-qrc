from __future__ import annotations

from src.experiment.e1 import campaign_plan
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


def test_configuration_hash_includes_the_complete_resolved_protocol():
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
        resolved_configuration={"readout": {"path_nu": 2.5}},
    )
    second = spec.configuration_hash(
        dataset_hash="dataset",
        program_fingerprint="program",
        resolved_configuration={"readout": {"path_nu": 1.5}},
    )
    assert first != second
