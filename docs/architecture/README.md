# QuaRK Architecture

This directory owns the engineering documentation for the empirical evaluation
framework:

- [Experiment framework](experiment-framework.md): immutable domain model,
  pairing semantics, and campaign-local cache rules.
- [Implementation contract](implementation-contract.md): normative
  theorem-to-code conventions.
- [Artifacts and execution](artifacts-and-execution.md): lifecycle, integrity,
  resume behavior, CLI, and provider extension boundary.

The executable mirror is
`experiments/empirical_evaluation/experiment_contract.yaml`. Before a
production run, create a versioned manifest from `production.template.yaml`
and resolve every `REQUIRED` value. Locked contract fields may change only with
a corresponding manuscript revision.
