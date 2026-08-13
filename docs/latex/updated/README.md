# QuaRK TMLR manuscript

Self-contained anonymous TMLR LaTeX project for

> **QuaRK: Quantum Reservoir Kernels for Temporal Learning**

This revision uses the notation layer in `style/`, including all new
width-resolved and finite-pool objects in `style/quark_commands.tex`.

## Build

On a standard TeX installation:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

In the validation container, `/usr/bin/bibtex` is not configured, so the
checked PDF was built with:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
/usr/bin/bibtex.original main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Project structure

- `main.tex`: TMLR entry point and abstract.
- `preamble.tex`: imports package, notation, and theorem layers.
- `style/quark_commands.tex`: QuaRK-specific notation, including the
  projection-information class/floor, good-design mass, and representation
  tolerance.
- `sections/problem_setting.tex`: causal FMP regression, gapped windows, and
  beta mixing.
- `sections/quark_family.tex`: coupled Gaussian preprocessing,
  topology-aware frozen quantum reservoirs, graph-local operator spreading,
  grouped observables, feature maps, RMS-normalized Mat\'ern geometry, and risks.
- `sections/model_selection.tex`: finite frozen pool and full-sample joint ERM.
- `sections/theory.tex`: fidelity terms, dependent-data uniform deviation and
  geometric-mixing gap rule, width-resolved approximation, positive
  good-design mass, finite-pool coverage, ERM oracle inequality,
  resource-to-performance theorem, and the complementary finite-data JL
  result.
- `sections/experiments.tex`: three empirical campaigns aligned with the same
  resource roles.
- `appendices/proofs.tex`: proofs of all theorem-section results.
- `appendices/technical_background.tex`: imported tools and attribution.
- `appendices/empirical_materials.tex`: structured resource grids, task atlas, nested cache
  construction, baselines, and reporting details.
- `THEORY_AUDIT.md`: concise audit of the revised theorem dependency chain.
- `IMPLEMENTATION_CONTRACT.md`: theorem-to-code contract freezing data indexing,
  initialization, measurement, readout, selection, paired randomness, and
  empirical-gap semantics.
- `implementation/experiment_contract.yaml`: machine-readable companion manifest
  distinguishing theory-locked conventions from run-level choices that must be
  frozen before the first benchmark run.

## Central theorem chain

The revised paper deliberately states the ingredients in dependency order:

1. finite-window, finite-shot, and dependent-data deviations, together with a
   logarithmic-in-`N` sufficient gap rule under geometric mixing;
2. a projection-information floor `\projFloor_n` at width `n`;
3. width-resolved QuaRK approximation down to that floor;
4. positive probability mass `\goodMass_\arch` around a strictly good support
   design;
5. exact pool miss probability `(1-\hitMass)^S`;
6. a full-sample finite-pool ERM oracle inequality; and
7. an operational excess-risk certificate for the predictor actually returned.

The final bound has the form

```text
operational excess risk
<= projection-information floor
 + finite representation slack
 + 2 * dependent-data generalization
 + finite-shot error
 + finite-window error,
```

with failure probability bounded by

```text
delta + (1 - hit-mass)^S.
```

Thus `S` has both an exponentially improving representation-coverage role and
a logarithmic statistical selection price.  The manuscript deliberately does
not invert this relation into a numerical pool-size prescription because it
proves positivity, but not a useful task-independent lower bound, for the hit
mass.

## Mat\'ern geometry and physical resource accounting

The readout kernel uses the root-mean-square feature distance
`||z-z'||_2 / sqrt(p)` at feature dimension `p`.  This is equivalent to a
fixed Mat\'ern kernel on `z/sqrt(p)`, preserves universality at every finite
dimension, and prevents architecture-dependent feature count from
mechanically inflating kernel distances.  The resulting window and grouped-shot
risk bounds use the corresponding `1/sqrt(p)` section-map sensitivity; for the
cycle cover, the worst-case shot-risk term scales as `M^{-1/2}` rather than
`sqrt(R n / M)`.  The physical acquisition burden is still explicit: the
complete measured pool uses `9 N M S R` independent state preparations/circuit
runs.

The run count uses the manuscript's reusable-`n`-qubit hardware convention and
is intentionally distinct from per-run circuit resources.  One branch
processing a length-`w` window uses `O(n)` qubits and `O(w)` logical trajectory
depth under the declared parallel gate schedule.  Physically parallelizing all
`R` branches instead gives `O(Rn)` width and can trade up to the factor `R`
from wall-clock executions into hardware parallelism.


## Mixer and observable-bank role

The identity support point remains useful for asymptotic universality, but the
nontrivial mixer now has a separate finite-resource theorem.  Product input
injection cannot enlarge Pauli support; the identity mixer therefore leaves
each raw observable coordinate confined to its seed qubits.  A generic
edge-local cycle mixer activates higher-support Pauli components, with reach
controlled by graph neighborhoods.  Vertex and edge observables seed different
direct interaction orders.  Campaign I ablates identity, local-only, and full
cycle mixers under vertex-only and complete observable banks and reports Pauli
spreading plus exact-feature effective rank alongside NEMSE.

## Width and JL are intentionally separated

Under the canonical Gaussian prefix coupling, the projection-information floor
is nonincreasing in `n`; it vanishes almost surely once `n >= d`.  This is the
population approximation role of width.  The Johnson--Lindenstrauss proposition
is retained separately: it certifies finite observed pairwise geometry at a
width logarithmic in the number of observed raw inputs, but is not used as a
prediction theorem.

## Empirical scope

The manuscript specifies, but does not yet report, three campaigns:

1. representational phase diagrams and a 36-task noiseless Legendre atlas;
2. empirical finite-shot, finite-window, and dependent-sample fidelity gaps,
   with the burned-in causal proxy carrying an explicit contraction residual;
3. nested frozen-pool search, comparing full-sample ERM selection with the
   oracle-best candidate in the same prefix.

The qubit-width sweep is the deliberate exception to the otherwise dyadic resource grids: it uses the dense interval `n in {3,4,5,6,7,8,9,10,11,12}`.  The widths share prefixes of one 12-row Gaussian master matrix, with the `1/sqrt(n)` normalization restored at every width.
