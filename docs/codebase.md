# Codebase Guide

The public release is intentionally narrow. The codebase should be read as three retained layers plus one wrapper:

## 1. QuaRK Core

The QuaRK implementation remains in the existing `src/` packages:

- `src/qrc/` for circuit construction, runners, and feature retrieval
- `src/models/` for the QuaRK feature-to-readout model stack
- `src/experiment/` for experiment orchestration

## 2. Temporal Baseline

The retained classical comparison is the temporal baseline surface:

- `src/experiment/temporal_baselines.py`

This public surface is intentionally restricted to:

- ESN state generation
- ESN+ridge support when needed internally
- ESN+Matérn as the retained public baseline

The broader historical classical-baseline collection is not part of the public release story.

## 3. Retained Experiment Drivers

The retained experiment drivers are:

- `src/experiment/scripts/rebuttal/prepare_real_tser.py`
- `src/experiment/scripts/rebuttal/run_varma_ablation_suite.py`
- `src/experiment/scripts/rebuttal/run_real_quark_temporal_budget_comparison.py`
- `src/experiment/scripts/rebuttal/run_rebuttal_result_audit.py`

These are wrapped by the single public entrypoint:

- `src/experiment/scripts/release.py`

Public docs should reference the wrapper, not the lower-level scripts directly.

## 4. Canonical Artifact Tree

The public release artifacts live under:

- `artifacts/public_release/varma_ablation/`
- `artifacts/public_release/real_world_temporal_benchmark/`
- `artifacts/public_release/audit/`

These files are the public source of truth for the released empirical claims.

Within `artifacts/public_release/varma_ablation/`, the retained public VARMA surface is:

- the canonical claim table used in the rebuttal text,
- the fixed `w=25, d=3` QuaRK-only architecture sweep table,
- and the finite-shot ablation table.

## 5. What Is Intentionally Not Part Of The Public Release Story

The following are intentionally excluded from the public-facing workflow:

- exploratory rebuttal-era triage
- overnight search helpers
- hydraulic-only sweep helpers
- standalone launcher scripts for deleted classical baselines
- draft rebuttal notes under `docs/rebuttal/`

Those paths may still exist locally during development, but they are not part of the documented release contract.
