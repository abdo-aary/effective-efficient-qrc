# Codebase Guide

The public release is intentionally narrow. The codebase should be read as three retained layers plus one wrapper:

## 1. QuaRK Core and Backends

The refactored implementation lives directly in the existing `src/` package:

- `src/core/` contains immutable mathematical programs, resources, seeds,
  capabilities, requests, and structured results.
- `src/estimators/` defines exact expectation and local-Pauli CSMoM semantics.
- `src/backends/aer/` is the independent density-matrix correctness oracle.
- `src/backends/nvidia/` is the production CuPy implementation.
- `src/backends/ibm/` implements stochastic reset trajectories, Runtime jobs,
  and hardware provenance.
- `src/features/` owns canonical `(N,R,K)` ordering and readout flattening.
- `src/artifacts/` writes checksum-validated `quark.run/v1` artifacts.
- `src/models/` and `src/experiment/` consume these typed APIs.

The former `src/qrc/` hierarchy has been retired. Legacy implementations that
remain necessary for artifact compatibility live behind explicit
`legacy_*` modules within their owning layer.

| Estimator | Aer CPU | NVIDIA GPU | IBM QPU |
|---|---:|---:|---:|
| Exact expectations | yes | yes | no |
| Local-Pauli CSMoM | yes | yes | yes |

Exact execution never enables history truncation or branch pruning. The legacy
finite-history runner is explicitly named `TruncatedReservoirChannelRunner`.

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
