from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.experiment.cli import main


ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "experiments/empirical_evaluation/manifests/smoke.yaml"


def _manifest(tmp_path: Path) -> tuple[Path, Path]:
    payload = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    artifact_root = tmp_path / "artifacts"
    payload["artifact_root"] = str(artifact_root)
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path, artifact_root


def test_cli_fake_study_run_and_json_csv_comparison_aggregation(tmp_path, capsys):
    manifest, artifact_root = _manifest(tmp_path)
    assert main(
        [
            "run",
            str(manifest),
            "--campaign",
            "campaign_i",
            "--study",
            "memory_vs_lag",
            "--provider",
            "fake",
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["finalization_scope"] == "study"
    assert result["selected_studies"] == ["memory_vs_lag"]

    assert main(
        [
            "aggregate",
            str(artifact_root),
            "--campaign",
            "campaign_i",
            "--study",
            "memory_vs_lag",
            "--format",
            "json",
        ]
    ) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 48
    assert all(row["study_id"] == "memory_vs_lag" for row in rows)

    assert main(
        [
            "aggregate",
            str(artifact_root),
            "--study",
            "memory_vs_lag",
            "--format",
            "csv",
        ]
    ) == 0
    csv_output = capsys.readouterr().out
    assert "comparison_id" in csv_output.splitlines()[0]
    assert len(csv_output.splitlines()) == 49


def test_cli_numerical_aer_preflight_fails_before_artifact_writes(tmp_path):
    manifest, artifact_root = _manifest(tmp_path)
    with pytest.raises(ValueError, match="Aer canary"):
        main(
            [
                "run",
                str(manifest),
                "--campaign",
                "campaign_i",
                "--study",
                "memory_vs_lag",
                "--provider",
                "numerical",
                "--backend",
                "aer",
            ]
        )
    assert not artifact_root.exists()
