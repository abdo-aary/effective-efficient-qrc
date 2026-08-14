# Artifacts and Execution

The local runner executes a validated plan synchronously:

```text
prepare -> acquire -> analyze -> finalize
```

Every node identity contains its semantic specification, plan identity,
versioned provider identity, and upstream artifact digests. Nodes are written
to temporary sibling directories, checksummed, and atomically published.
Existing nodes are reused only after their identity and checksums validate.
Failures are append-only attempt records outside the successful run tree.

## Array payloads

A `NodePayload` contains JSON metadata and named `ArrayAsset` values. Owned
arrays are immutable, uncompressed `.npy` files written in the node transaction.
Reload uses `numpy.load(..., mmap_mode="r", allow_pickle=False)`, validates the
file checksum, dtype, base shape, selector, and source node digest, and exposes
a read-only view.

Derived prefixes store relative descriptors pointing to upstream arrays. They
compose positive slice selectors and do not copy the base data. Paths are
relative to the referencing node, so moving an entire run root preserves the
graph. Pickle and object arrays are forbidden.

## Study execution

`ExperimentRunner.run(..., study="memory_vs_lag")` filters every stage by the
explicit `study_id` without changing experiment or node identities. Its
study-finalization artifact depends on every selected comparison. Calling the
runner without a study performs separate global finalization.

Numerical preflight runs before the artifact store is created. The current
numerical provider rejects every study except `memory_vs_lag`; Aer is a bounded
small correctness oracle, while NVIDIA/CuPy complex128 is the locked production
path. NVIDIA execution may shard the window axis across explicit GPU IDs. Each
shard computes all locked `tau_plus` values from one exact pure-history
evolution, and shards are concatenated in the original window order before the
node is written atomically.

## CLI

```bash
quark-experiment validate experiments/empirical_evaluation/manifests/smoke.yaml
quark-experiment plan experiments/empirical_evaluation/manifests/smoke.yaml

quark-experiment run \
  experiments/empirical_evaluation/manifests/smoke.yaml \
  --campaign campaign_i \
  --study memory_vs_lag \
  --provider numerical \
  --backend nvidia \
  --gpu-ids 0 1 2 3

quark-experiment aggregate storage/artifacts/empirical_evaluation \
  --campaign campaign_i \
  --study memory_vs_lag \
  --format csv
```

The smoke manifest exercises software and performance and is not a reportable
paper result. A reportable launch requires a versioned production manifest with
every `REQUIRED` value resolved.

## Extension rule

New numerical providers may add payload assets and compute engines, but they
must not infer pairing, study membership, selection, or cache reuse from naming
conventions. Those relationships belong in the validated plan. Provider
algorithm versions must change whenever artifact semantics change.
