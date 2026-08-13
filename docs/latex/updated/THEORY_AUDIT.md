# QuaRK theory audit - current revision

## 0. Mixer role at finite resources

The product input encoder is qubit-local and cannot enlarge Pauli support.  With
`W_n = I`, every raw vertex observable depends directly on one projected input
stream and every edge observable on at most two.  The balanced graph-local
mixer has a deterministic light cone and, under the declared continuous random
axis/angle law, activates selected weight-2 and weight-3 Pauli components almost
surely.  Hence the mixer and edge bank enrich the raw frozen representation
without being claimed to enlarge the asymptotic Matérn hypothesis class or to
monotonically improve task risk.

This note records the logical dependency chain of the current manuscript.  It
is not part of the paper.

## 1. Fixed-representation fidelity

For any fixed frozen design and Ivanov-bounded Matérn readout:

- strict contraction yields the finite-window term `epsilon_win`;
- the nine-setting grouped measurement schedule yields `epsilon_shot`;
- beta-mixing coupling plus RKHS Rademacher complexity yields a uniform
  measured-to-operational deviation `epsilon_gen` simultaneously over all
  `S` frozen candidates and all readouts in the Ivanov ball.

The last uniformity is what permits full-sample finite-pool ERM without a
separate validation set.  Its selection price is logarithmic in `S`.

The Matérn kernel is evaluated on the RMS feature metric
`||z-z'||_2 / sqrt(p)`.  Consequently its section-map Lipschitz constant with
respect to the raw `p`-dimensional feature vector contains `1/sqrt(p)`.  This
cancels pure feature-count growth from the worst-case window bound and, for the
cycle measurement cover, gives `epsilon_shot = O(M^{-1/2})` rather than
`O(sqrt(R n / M))`.  This is a risk-sensitivity statement, not a claim that
larger architectures are physically free: the measured pool still needs
`9 N M S R` independent circuit runs.

## 2. Width-resolved population approximation

For a fixed realized Gaussian projection `Pi_n`, define the class of continuous
fading-memory predictors that depend on the original history only through the
complete projected history.  The projection-information floor `I_n^Pi` is the
best squared-error excess risk in that class.

The input encoder and all reservoir dynamics depend on the raw input only
through `Pi_n x`, so no QuaRK predictor can beat this floor.  Conversely,
projected-history point separation, Stone--Weierstrass spatial multiplexing,
and Matérn universality show that finite-multiplex QuaRK predictors with
finite-norm readouts can approach the floor arbitrarily closely.

Under the canonical Gaussian prefix coupling, width `n+1` contains the
information of width `n`, hence `I_{n+1}^Pi <= I_n^Pi`.  If `n >= d`, the
Gaussian map has full column rank almost surely and the floor is zero.  No
strict improvement is claimed for every target at every width.

## 3. Support existence to positive random-design mass

If a support design has approximation error strictly below
`I_n^Pi + epsilon_rep`, continuity of the contractive causal reservoir map in
its frozen rate/circuit parameters gives an open neighborhood with the same
strict inequality.  Because the center lies in the support of the declared
random-design law, this neighborhood has positive probability.

The width-resolved approximation theorem guarantees that for every positive
representation slack there exist finite `R` and finite readout radius for which
such a strictly good support design exists.  The paper therefore proves
positivity of the target-dependent hit mass, but not a distribution-free
numerical lower bound.

## 4. Finite-pool coverage

Conditional on the shared projection, candidate designs are i.i.d.  If the
single-design hit mass is `p_hit`, then

```text
P[pool misses every epsilon_rep-good design] = (1 - p_hit)^S.
```

This is exact.  It supplies the representation-search role of `S`.

## 5. Full-sample ERM oracle inequality

On the simultaneous uniform deviation event, joint ERM over candidate index
and Ivanov readout satisfies

```text
R_op(selected)
<= min_s inf_h R_op(D_s,h) + 2 epsilon_gen.
```

Thus using the same sample to fit the readout and choose the frozen design is
controlled rather than assumed harmless.

## 6. End-to-end resource-to-performance theorem

Intersecting pool coverage with the ERM oracle event and then inserting the
shot and finite-window links gives

```text
R_op(selected) - R_star
<= I_n^Pi
 + epsilon_rep
 + 2 epsilon_gen
 + epsilon_shot
 + epsilon_win
```

with conditional failure probability at most

```text
delta + (1 - p_hit)^S.
```

A second displayed bound gives the causal exact-feature risk by paying the shot
and window terms once more.

## 7. Finite-data JL result

The Gaussian JL proposition is intentionally separate.  It certifies pairwise
geometry of the finite set of raw inputs appearing in the observed windows at
`n = O(epsilon^{-2} log(Nw/delta_Pi))`.  It is not used to infer approximation
or predictive performance; that role belongs to the projection-information
floor.

## 8. Explicit limitations retained

- `p_hit` is positive under the theorem's support condition but is not assigned
  a universal numerical lower bound; accordingly no a-priori numerical
  calibration of `S` is claimed.
- The beta-mixing bound is sufficient and can require increasing gap as sample
  size grows; under `beta_Z(q) <= C_beta exp(-c_beta q)`, the manuscript gives
  an explicit sufficient `g = O(log N)` rule at fixed confidence.
- Width monotonicity concerns retained information, not monotone finite-sample
  test risk.
- No theorem gives quantum advantage over classical reservoirs.

## 9. Numerical causal proxy

Campaign II does not treat a finite burn-in as literally causal.  At burn-in
`B`, each branch is within `2 lambda_+^B` in trace norm of the infinite-past
state and the corresponding readout-risk residual is bounded explicitly.  At
the largest planned memory horizon (`tau_+ = 64`, `B = 1024`) the branch-state
residual is `2 exp(-16) ~= 2.25e-7`.

## 10. Circuit runs versus per-run depth and width

The acquisition count is the number of independent branch-state
preparations/circuit runs under the reusable-`n`-qubit hardware convention.
One branch processing a window of length `w` has logical trajectory depth
`O(w)` because the declared per-step block has constant scheduled depth, and
physical width `O(n)`.  Instantiating all `R` branches in parallel instead gives
width `O(Rn)` and can trade up to the factor `R` from wall-clock executions
into hardware parallelism.
