from __future__ import annotations

from pathlib import Path

import src.experiment.scripts.release as release_module
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


def test_sync_public_release_artifacts_includes_architecture_table(tmp_path, monkeypatch):
    project_root = Path(tmp_path)
    public_root = project_root / "artifacts" / "public_release"
    monkeypatch.setattr(release_module, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(release_module, "PUBLIC_ARTIFACT_ROOT", public_root)

    for rel_path in (
        "storage/results/rebuttal/result_audit/canonical_varma_claim_table.csv",
        "storage/results/rebuttal/result_audit/canonical_varma_claim_table.md",
        "storage/results/rebuttal/result_audit/canonical_architecture_ablation_table.csv",
        "storage/results/rebuttal/result_audit/canonical_architecture_ablation_table.md",
        "storage/results/rebuttal/varma_ablation/aggregate/finite_shot_ablation_table.md",
        "storage/results/rebuttal/result_audit/canonical_real_world/temporal_budget_final_table.csv",
        "storage/results/rebuttal/result_audit/canonical_real_world/temporal_budget_final_table.md",
        "storage/results/rebuttal/result_audit/canonical_real_world/temporal_budget_lambda_selection.csv",
        "storage/results/rebuttal/result_audit/canonical_real_world/temporal_budget_lambda_selection.md",
        "storage/results/rebuttal/result_audit/canonical_real_world/temporal_budget_selection.json",
        "storage/results/rebuttal/result_audit/canonical_real_world/temporal_budget_shot_sweep.csv",
        "storage/results/rebuttal/result_audit/canonical_real_world/temporal_budget_shot_sweep.md",
        "storage/results/rebuttal/result_audit/canonical_summary.json",
        "storage/results/rebuttal/result_audit/code_path_audit.csv",
        "storage/results/rebuttal/result_audit/code_path_audit.md",
        "storage/results/rebuttal/result_audit/claim_ledger.csv",
        "storage/results/rebuttal/result_audit/claim_ledger.md",
        "storage/results/rebuttal/result_audit/rebuttal_claim_checklist.csv",
        "storage/results/rebuttal/result_audit/rebuttal_claim_checklist.md",
        "storage/results/rebuttal/result_audit/aggregate_comparison.csv",
        "storage/results/rebuttal/result_audit/aggregate_comparison.md",
        "storage/results/rebuttal/result_audit/selective_recompute_report.csv",
        "storage/results/rebuttal/result_audit/selective_recompute_report.md",
    ):
        path = project_root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stub\n", encoding="utf-8")

    release_module.sync_public_release_artifacts()

    assert (public_root / "varma_ablation" / "canonical_architecture_ablation_table.csv").exists()
    assert (public_root / "varma_ablation" / "canonical_architecture_ablation_table.md").exists()
