# QuaRK Public Release

This repository is the public source of truth for the released QuaRK code and the retained empirical results:

- the QuaRK implementation,
- the matched-budget temporal baseline `ESN+Matérn`,
- the QuaRK-only VARMA ablation,
- the real-world matched-budget temporal benchmark,
- and the audit bundle that verifies the reported release numbers.

This release is intentionally narrower than the broader internal rebuttal workspace. It does **not** present a general classical-baseline zoo, triage experiments, or exploratory search scripts. The public experiment story is limited to:

1. **Controlled VARMA ablation**
   QuaRK-only mechanism validation on the fixed synthetic setting.
2. **Real-world temporal benchmark**
   Fixed-budget QuaRK versus `ESN+Matérn`.
3. **Audit bundle**
   Canonical tables and verification reports used to check that the released metrics are grounded in saved artifacts and selective recomputation.

## Results Included In This Release

- `artifacts/public_release/varma_ablation/`
  Canonical VARMA claim tables, the fixed architecture sweep table, and the retained finite-shot ablation table.
- `artifacts/public_release/real_world_temporal_benchmark/`
  Final 10-dataset QuaRK vs `ESN+Matérn` table, lambda/shot selection summaries, and the dataset manifest.
- `artifacts/public_release/audit/`
  Canonical summary, code-path audit, claim ledger, rebuttal checklist, and selective recomputation report.

## Public Reproduction Contract

Use the single public wrapper:

```bash
python -m src.experiment.scripts.release <subcommand>
```

Supported subcommands:

- `prepare-real-data`
- `run-varma-ablation`
- `run-real-world-benchmark`
- `run-audit`
- `run-all`

The wrapper keeps the public workflow small and syncs canonical release artifacts into `artifacts/public_release/`.

## Minimal Reproduction

Prepare the retained real datasets:

```bash
python -m src.experiment.scripts.release prepare-real-data
```

Run the retained QuaRK-only VARMA ablation:

```bash
python -m src.experiment.scripts.release run-varma-ablation
```

Run the retained real-world temporal benchmark:

```bash
python -m src.experiment.scripts.release run-real-world-benchmark
```

Run the verification audit:

```bash
python -m src.experiment.scripts.release run-audit
```

Run the full release workflow:

```bash
python -m src.experiment.scripts.release run-all
```

## Fixed Benchmark Setting

The retained real-world temporal benchmark uses:

- QuaRK: `n=5`, `R=3`, `k=2`, `lambda_0=0.5`, `3000` shots
- Temporal baseline: `ESN+Matérn` with `D=315`

## Where To Look Next

- `docs/reproduction.md`
- `docs/codebase.md`

## License

Released under the MIT License. See `LICENSE`.
