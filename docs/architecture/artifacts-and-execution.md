# Artifacts and Execution

The local runner executes a validated plan synchronously:

```text
prepare -> acquire -> analyze -> finalize
```

Every node identity contains its semantic specification, plan identity, and
upstream artifact digests. Nodes are written to temporary sibling directories,
checksummed, and atomically published. Existing nodes are reused only after
their identity and checksums validate. Failures are append-only attempt records
outside the successful run tree.

## CLI

```bash
quark-experiment validate experiments/empirical_evaluation/manifests/smoke.yaml
quark-experiment plan experiments/empirical_evaluation/manifests/smoke.yaml
quark-experiment run experiments/empirical_evaluation/manifests/smoke.yaml \
  --campaign campaign_i --repetition 0
quark-experiment aggregate storage/artifacts/empirical_evaluation
```

`run` uses deterministic fake providers in this milestone. Production
trajectory, grouped-measurement, Ivanov-readout, and baseline providers must
implement the public protocols without changing plan semantics.

## Extension rule

New numerical providers may change payload formats and compute engines, but
they must not infer pairing, selection, or cache reuse from naming conventions.
Those relationships belong in the validated plan.

