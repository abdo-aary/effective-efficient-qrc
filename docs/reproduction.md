# Reproduction Guide

The public release is intentionally organized around a single wrapper CLI:

```bash
python -m src.experiment.scripts.release <subcommand>
```

The wrapper is the only command path documented for the public release. It syncs canonical outputs into `artifacts/public_release/`.

## 1. Prepare The Retained TSER Datasets

```bash
python -m src.experiment.scripts.release prepare-real-data
```

Dry run:

```bash
python -m src.experiment.scripts.release prepare-real-data --dry-run
```

This prepares the retained 10 real-world datasets used by the temporal benchmark.

## 2. Run The QuaRK-Only VARMA Ablation

```bash
python -m src.experiment.scripts.release run-varma-ablation
```

Dry run:

```bash
python -m src.experiment.scripts.release run-varma-ablation --dry-run
```

This retained ablation covers the QuaRK mechanism story only. The public release does not document or expose the broader classical VARMA benchmark surface.

Canonical outputs are synced into:

- `artifacts/public_release/varma_ablation/canonical_varma_claim_table.csv`
- `artifacts/public_release/varma_ablation/canonical_varma_claim_table.md`
- `artifacts/public_release/varma_ablation/finite_shot_ablation_table.md`

## 3. Run The Real-World Temporal Benchmark

```bash
python -m src.experiment.scripts.release run-real-world-benchmark
```

Dry run:

```bash
python -m src.experiment.scripts.release run-real-world-benchmark --dry-run
```

The retained benchmark setting is fixed:

- QuaRK: `n=5`, `R=3`, `k=2`, `lambda_0=0.5`, `3000` shots
- Baseline: `ESN+Matérn` with `D=315`

Canonical outputs are synced into:

- `artifacts/public_release/real_world_temporal_benchmark/temporal_budget_final_table.csv`
- `artifacts/public_release/real_world_temporal_benchmark/temporal_budget_final_table.md`
- `artifacts/public_release/real_world_temporal_benchmark/temporal_budget_lambda_selection.csv`
- `artifacts/public_release/real_world_temporal_benchmark/temporal_budget_shot_sweep.csv`
- `artifacts/public_release/real_world_temporal_benchmark/temporal_budget_selection.json`
- `artifacts/public_release/real_world_temporal_benchmark/dataset_manifest.csv`
- `artifacts/public_release/real_world_temporal_benchmark/dataset_manifest.md`

## 4. Run The Release Audit

```bash
python -m src.experiment.scripts.release run-audit
```

Dry run:

```bash
python -m src.experiment.scripts.release run-audit --dry-run
```

This rebuilds canonical tables from raw metrics, checks the current release claims, and performs a small selective recomputation pass.

Canonical outputs are synced into:

- `artifacts/public_release/audit/canonical_summary.json`
- `artifacts/public_release/audit/code_path_audit.md`
- `artifacts/public_release/audit/claim_ledger.md`
- `artifacts/public_release/audit/rebuttal_claim_checklist.md`
- `artifacts/public_release/audit/selective_recompute_report.md`

## 5. Run Everything

```bash
python -m src.experiment.scripts.release run-all
```

Dry run:

```bash
python -m src.experiment.scripts.release run-all --dry-run
```

## Notes

- The canonical public artifact tree is `artifacts/public_release/`.
- The release documentation intentionally does not depend on `storage/results/rebuttal`.
- The notebook is optional and not part of the documented reproduction path.
