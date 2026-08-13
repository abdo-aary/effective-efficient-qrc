# QuaRK theorem-to-implementation contract

This file is the normative implementation companion to the manuscript.  It
freezes every convention that is already determined by the mathematics and the
empirical protocol.  An implementation is conforming only if it follows the
items marked **LOCKED** literally.  Items marked **PRE-RUN** are not fixed by a
theorem; they must nevertheless be assigned once in the versioned run manifest
before the corresponding benchmark campaign is executed and must never be
retuned per architecture or after inspecting test performance.

## 1. Randomness hierarchy

**LOCKED**

For each paired repetition use independent random streams for

1. the temporal training process;
2. the independent held-out/test process;
3. the master Gaussian preprocessor rows;
4. frozen reservoir designs/candidates; and
5. training and held-out measurement outcomes.

The shared Gaussian projection is drawn independently of the data and is shared
by every branch and every candidate in the same run.  Conditional on that
projection, candidate designs are i.i.d. from the declared design law.  Test
labels and held-out measurement outcomes are never used for fitting or design
selection.

Within a paired resource comparison, reuse the same temporal realization,
preprocessor rows, and eligible design seeds exactly as specified by the prefix
rules.  Pairing is variance reduction only and must not alter any marginal law.

## 2. Temporal data and window indexing

**LOCKED**

For dependence parameter `rho_in`, initialize the stationary Gaussian driver
with `G_0 ~ N(0,I_d)` and evolve

```text
G_t = rho_in * G_{t-1} + sqrt(1-rho_in^2) * E_t,
E_t ~ iid N(0,I_d).
```

Transform coordinatewise by

```text
X_{t,j} = 2 Phi(G_{t,j}) - 1.
```

Labels are generated exactly from the declared deterministic target, with no
observation noise.  Hence the benchmark Bayes risk is zero.

For window length `w` and raw-time gap `g`, set

```text
Delta = w + g
W_j = (X_{j Delta-w+1 : j Delta}, Y_{j Delta}).
```

A finite-window reservoir state uses exactly the `w` inputs in that slice,
ordered from oldest to newest.  Training and held-out/test trajectories are
independent realizations of the same stationary process.

Reference settings, unless an axis is explicitly varied, are

```text
eta_ref = (n=8, R=16, tau_+=32),  lambda_+ = exp(-1/32)
N_0 = 1024
w_ref = 512
g_ref = 0
S = 1
features = exact
input drive = iid
```

The mixer-mechanism ablation is the declared exception: `(n,R,tau_+)=(8,8,8)`.

## 3. Gaussian preprocessing

**LOCKED**

Draw one master matrix `G` with iid standard-normal rows.  At width `n`, use

```text
Pi_n = G[1:n,:] / sqrt(n).
```

Thus each fixed-width matrix has iid `N(0,1/n)` entries.  Width sweeps use the
same master rows and restore the `1/sqrt(n)` normalization at each width.  The
quantum mixer is not interpreted as a nested circuit across different widths.

The angle injected on qubit `j` is

```text
alpha_j(x) = pi * tanh((Pi_n x)_j).
```

and the input unitary is the tensor product of `R_y(alpha_j)` rotations.

## 4. Frozen branch dynamics and initialization

**LOCKED**

Every finite-window branch starts from

```text
rho_reset = |+><+|^{\otimes n}.
```

For a window `x_{-w+1:0}`, apply the channels in chronological order

```text
rho <- T_{r,x_{-w+1}}(rho)
...
rho <- T_{r,x_0}(rho)
```

where

```text
T_{r,x}(rho)
 = lambda_r W_r U_in(x) rho U_in(x)^dag W_r^dag
   + (1-lambda_r) rho_reset.
```

The branch rates are sampled by

```text
U_r ~ Uniform(0,1)
lambda_r(lambda_+) = lambda_0 * (lambda_+/lambda_0) ** U_r
lambda_0 = exp(-1).
```

The same `U_r` is reused across a paired `lambda_+` sweep.  An architecture with
`R` branches uses the first `R` branch seeds.

**PRE-RUN**

The theorem permits any angle distribution satisfying the manuscript support
assumptions.  The experiment manifest must therefore freeze the numerical
`gamma` and the exact continuous angle distribution once before the runs.  The
axis distribution is uniform on the unit sphere and the matching-layer order is
uniform over permutations, as specified by the method.

## 5. Exact and measured features

**LOCKED**

The complete bank is

```text
O_V = {X_v,Y_v,Z_v}
O_E = {P_u P'_v : {u,v} is a cycle edge, P,P' in {X,Y,Z}}
O_n = O_V union O_E
q_n = 12 n
m_eta = R q_n.
```

Exact features are terminal expectation values `Tr(O rho)` concatenated branch
by branch in a single deterministic observable ordering.  The ordering must be
saved in metadata and reused in every cache and readout.

The nine global Pauli settings are generated from one proper three-colouring
`c(v)` of the cycle by

```text
s_{a,b}(v) = a + b c(v) mod 3,  a,b in {0,1,2},
```

with `{0,1,2}` mapped to `{X,Y,Z}` in a fixed recorded order.  Each setting has
exactly `M` independent terminal-state preparations.  A global shot returns one
`+/-1` value per qubit; every compatible observable is computed from that same
global outcome.  Do **not** add independent coordinatewise Gaussian/Bernoulli
noise.

A one-qubit observable is estimated by pooling its compatible outcomes from the
three compatible settings (`3M` samples).  An edge Pauli product is estimated
from its unique compatible setting (`M` samples).  Smaller `M` values use the
first `M` outcomes of every setting, preserving within-shot correlations.

Training measurements are cached once.  Any empirical operational evaluation
uses a fresh measurement cache independent of training measurements.

## 6. RMS Matérn readout

**LOCKED**

At feature dimension `p`, use

```text
d_rms(z,z') = ||z-z'||_2 / sqrt(p).
```

The kernel is exactly

```text
k(z,z') = 2^(1-nu)/Gamma(nu)
          * (sqrt(2 nu) d_rms(z,z') / xi)^nu
          * K_nu(sqrt(2 nu) d_rms(z,z') / xi),
k(z,z)=1,
```

where `K_nu` is the modified Bessel function of the second kind.  This is the
standard Matérn parameterization used by the section-stability proof.  Do not
replace it by another library convention without verifying parameter
identity.

Every method, including the direct raw-window and classical-reservoir baselines,
uses its own feature dimension `p` in the same RMS rule.  There is no
architecture-specific length-scale retuning.

The readout solves the Ivanov problem literally:

```text
min_h  (1/N) sum_j (h(z_j)-y_j)^2
s.t.   ||h||_H <= Lambda.
```

In representer coefficients `a`, this is

```text
min_a  ||K a-y||_2^2 / N
s.t.   a^T K a <= Lambda^2.
```

Use a deterministic numerical solver/tolerance and record both.  Candidate ties
are broken by the smallest candidate index; within a tied candidate choose the
minimum-RKHS-norm minimizer.

**PRE-RUN**

The numerical values of `nu>1`, `xi>0`, and `Lambda>0` must be written to the
run manifest before the architecture sweeps and then held fixed across all
QuaRK architectures and matched baselines.  They must not be retuned after
inspecting architecture-specific kernel-similarity diagnostics.

## 7. Full-sample finite-pool ERM

**LOCKED**

For each candidate `s`, fit an Ivanov readout on that candidate's training
features.  Select jointly by the smallest measured empirical training risk:

```text
(s_hat,h_hat) in argmin_{s<=S, ||h||<=Lambda} Rhat_{D_s,w}(h).
```

No validation fold is used.  No gate, angle, memory rate, or preprocessor
parameter is optimized by this rule.

In Campaign III's exact-feature main study, apply the identical selection rule
with exact features in place of measured features.  This is the zero-shot-error
special case used to isolate `S`.  In the finite-shot ranking check, return to
measured training features and use fresh held-out measurements for operational
evaluation.

For the oracle curve, each candidate readout is fitted **once on training data
only**.  The oracle is the minimum held-out/test risk among those already-fitted
candidate predictors.  Never refit or tune a readout using test labels.

## 8. Certified causal proxy

**LOCKED**

For a label time `t`, initialize the causal proxy at `rho_reset` and apply the
`B` inputs `X_{t-B+1:t}` in chronological order.  The default is `B=1024`.

For the finite-window comparator at the same label time, initialize independently
at `rho_reset` and replay exactly `X_{t-w+1:t}`.  Both sides therefore share the
same final `w` inputs.

The proxy is never called exactly causal in code or figures.  Record

```text
state residual <= 2 lambda_+^B
epsilon_burn = 4 Lambda (Lambda+Upsilon) L_kappa lambda_+^B
```

beside every empirical window-gap result.

## 9. Empirical gap semantics

**LOCKED**

All paired gap evaluations use the same held-out windows and labels on both
sides and the same fitted readout.

### Shot gap

1. Fit `h_M` on measured **training** features at shot count `M`.
2. On one common held-out set, compute
   - fresh measured finite-window features using an independent `M`-shot cache;
   - exact finite-window features.
3. Report

```text
Delta_hat_shot(M)
 = | Rhat_op_heldout(h_M) - Rhat_exact_heldout(h_M) |.
```

### Window gap

1. Fit `h_w` on exact finite-window **training** features at window length `w`.
2. On the same held-out label times, evaluate `h_w` on
   - exact finite-window features reset `w` inputs before the label;
   - exact `B`-burned-in causal-proxy features.
3. Report their absolute risk difference and the separate `epsilon_burn` bound.

### Generalization gap

Use exact features and the declared reference architecture.  Fit `h` on the
full gapped training sample and report

```text
Delta_hat_gen
 = | Rhat_training(h) - Rhat_independent_test(h) |.
```

This is the sole theory-facing gap whose first term is deliberately a training
risk.

### Selection regret

For every candidate, freeze its training-fitted readout.  Then

```text
Rhat_oracle(S) = min_{s<=S} Rhat_test(D_s,h_s)
Delta_hat_sel(S) = Rhat_test(D*_S,h*_S) - Rhat_oracle(S).
```

## 10. Metrics and diagnostics

**LOCKED**

NEMSE uses the variance of the common held-out/test labels for the corresponding
task/repetition:

```text
NEMSE = Rhat_test / Varhat_test(Y).
```

The same denominator is reused across all methods/configurations in the paired
comparison.  Do not use training-label variance.

For the mixer diagnostic, center each feature column on the common diagnostic
sample and use the stable rank

```text
r_stab(F_c) = ||F_c||_F^2 / ||F_c||_2^2,
r_stab(0)=0.
```

For Pauli spreading, coefficients are with respect to the Pauli basis
orthonormal under `2^{-n} Tr(P^dag Q)`; save the coefficient normalization in
metadata.  The reported spread score is exactly the manuscript's sum of
squared coefficient mass whose support leaves the seed support.

## 11. Resource accounting

**LOCKED**

The reported acquisition count uses reusable `n`-qubit hardware:

```text
C_Q(N,M,S,R) = 9 N M S R
```

independent branch-state preparations/circuit runs for a measured pool.

This is not a gate-count metric.  One branch trajectory has `O(n)` quantum
width and `O(w)` logical depth because the declared per-input-step schedule has
constant depth.  If all `R` branches are physically instantiated in parallel,
width becomes `O(Rn)` and up to a factor `R` can be traded from wall-clock runs.
The manuscript's plotted `C_Q` remains the reusable-hardware convention.

## 12. Nested-cache rules

**LOCKED**

- Campaign I: draw `R_max=64` branch seeds once; use prefixes `1:R`.
- Campaign II shot sweep: draw `M_max=8192` global outcomes per setting/branch/
  window; use prefixes `1:M` in each setting.
- Campaign III: draw `S_max=256` candidates, each with
  `R_search,max=32` exact-feature branches; use candidate prefixes `1:S` and
  branch prefixes `1:R`.
- Width sweep: draw one 12-row master Gaussian matrix, use every prefix `n=3,...,12`, and set `Pi_n=G[1:n]/sqrt(n)`.
- Memory-endpoint sweep: reuse the same uniforms `U_r` in the inverse-CDF draw.

Do not construct one `R_max*S_max*M_max` Cartesian cache.

## 13. Campaign-specific fixed axes

**LOCKED**

- Memory endpoint vs lag: vary `(tau_+,L)`, fix `(n,R)=(8,16)`.
- Multiplexing vs simultaneous modes: vary `(R,H)`, fix `n=8, tau_+=32`.
- Width vs spatial load: vary `n in {3,4,5,6,7,8,9,10,11,12}` and the declared `d_act` grid, fix `R=16, tau_+=32`.  This dense width interval is the sole deliberate exception to the otherwise dyadic resource grids.
- Mixer mechanism: `(n,R,tau_+)=(8,8,8)` with identity/local/full mixers and
  vertex-only/complete banks.
- Shot fidelity: vary `M` and the declared `(n,R)` architecture sequence, fix
  `tau_+=32`.
- Window fidelity: vary `(w,tau_+)`, fix `(n,R)=(8,16)`.
- Dependence study: vary `(tau_dep,g,N)` as declared, use `eta_ref` and exact
  features.
- Frozen-pool study: vary `(S,R)`, fix `n=8, tau_+=32`.

## 14. Run-level values that must be frozen before execution

**PRE-RUN**

The manuscript intentionally defines families rather than arbitrary numerical
software defaults.  A reproducible experiment release therefore needs one
versioned run manifest fixing at least:

- mixer `gamma` and exact angle distribution;
- Matérn `nu`, `xi`, and Ivanov `Lambda`;
- the exact master seed list;
- held-out/test sample size and any numerical burn-in used solely for generating
  a stationary test driver (not the reservoir causal proxy);
- the exact number of paired repetitions (the paper requires no fewer than
  16 for architecture/fidelity and 32 for pool curves);
- the finite-shot `M` used in Campaign III's representative ranking check;
- the exact frozen classical multiscale-reservoir dynamics and random-weight
  distributions;
- solver tolerances, floating precision, and any numerical jitter used in
  kernel linear algebra.

These are **not** missing theorem assumptions.  They are empirical implementation
choices.  They must be fixed once, logged, and kept identical wherever the
manuscript says a resource is being held constant.  Any change after examining
test performance constitutes a new experimental specification and must be
reported as such.
