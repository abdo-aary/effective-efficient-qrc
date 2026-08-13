# Implementation contract

`experiment_contract.yaml` is a machine-readable mirror of the normative
`../IMPLEMENTATION_CONTRACT.md`.

Before running a benchmark, copy it to a versioned run manifest and replace
every `REQUIRED` value under `pre_run_required`.  The resulting manifest should
be committed alongside the experiment scripts and stored with every output
cache.  Fields under `theory_locked` and `campaign_fixed_axes` must not be
changed unless the manuscript itself is revised.
