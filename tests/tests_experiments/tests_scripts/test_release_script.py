from __future__ import annotations

from src.experiment.scripts.release import main as release_main


def test_release_prepare_real_data_dry_run(capsys):
    release_main(["prepare-real-data", "--dry-run"])

    captured = capsys.readouterr().out
    assert "prepare TSER datasets=" in captured


def test_release_varma_ablation_dry_run(capsys):
    release_main(["run-varma-ablation", "--dry-run"])

    captured = capsys.readouterr().out
    assert "run architecture regimes=" in captured
    assert "run finite-shot sweep" in captured or "shots" in captured


def test_release_real_world_benchmark_dry_run(capsys):
    release_main(["run-real-world-benchmark", "--dry-run"])

    captured = capsys.readouterr().out
    assert "run exact QuaRK datasets=" in captured
    assert "run ESN+Matérn datasets=" in captured
    assert "run cached shot sweep from best exact datasets=" in captured


def test_release_audit_dry_run(capsys):
    release_main(["run-audit", "--dry-run"])

    captured = capsys.readouterr().out
    assert "rebuild canonical VARMA and real-world tables from raw metrics.csv only" in captured
