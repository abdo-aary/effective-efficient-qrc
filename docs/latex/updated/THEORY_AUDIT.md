# QuaRK theory audit

This note records the final internal audit of the theory-only portion of the TMLR manuscript. It is not a guarantee that no reviewer will request further clarification; it records the checks performed and the remaining scope restrictions.

## Central results checked

- The reset operation is defined as a linear CPTP replacement channel, with an explicit Kraus representation.
- Exact trace-norm contraction, existence and uniqueness of the causal state, measurability, and finite-initialization forgetting are proved directly.
- Multiplexed reservoirs are treated as a tuple of states followed by concatenated observable features; no tensor-product contraction is used.
- The finite-window feature, Matérn prediction, and squared-risk truncation constants were recomputed.
- The Matérn spectral density and covariance were checked against the manuscript parameterization; the stability result requires nu > 1.
- The interpolation theorem is conditional on pairwise-distinct exact features, proves strict positive definiteness, constructs the unique minimum-norm interpolant, and handles the zero-label case.
- The window-process beta-mixing index was audited, including the conservative one-step offset.
- The dependent-data theorem is a one-sided specialization of Mohri and Rostamizadeh (2008), with N = 2 mu a, the correct coupling remainder, and the source paper's absolute-value empirical Rademacher convention.
- The squared-loss Rademacher calculation retains the label-only offset required by that convention. The closed-form constant is deliberately conservative.
- The finite-candidate result is proved by applying the dependent-data theorem once to a fixed finite union and bounding its empirical Rademacher complexity with a bounded-difference/log-sum-exp argument.
- The geometric-mixing corollary retains the finite-window term and states the divisibility and nonempty-retained-sample conditions.
- The local-Pauli shadow requirement uses the 3^k squared-shadow-norm scaling and explicitly accounts for the union over windows and reservoirs.
- The measurement perturbation result is deterministic and uniform over the RKHS ball; it does not cover retraining on noisy features or physical gate/reset noise.

## Imported results and primary sources

The manuscript now identifies imported or adapted results in theorem headings or in the technical-background provenance subsection. Primary sources include the finite-set JL lemma, RKHS/representer results, Bochner's theorem, the non-i.i.d. Rademacher theorem, bounded differences, classical shadows, ARMA complete regularity, finite-state Markov contraction, and GARCH stationarity/mixing.

## Remaining scope restrictions

The main learning theorem requires a stationary beta-mixing supervised process, bounded labels, a fixed exact featurizer, a normalized Matérn kernel with nu > 1, and a fixed RKHS radius. The finite-family corollary permits selection only from a predeclared finite measurable family. Continuous/adaptive hyperparameter search, nonstationarity, unbounded labels, noisy-feature retraining, physical device noise, and quantum advantage are outside the theorem.

The HMM and VARMA synthetic constructions can be initialized exactly in stationarity. The GARCH implementation uses a long burn-in as a numerical approximation to the stationary law; the mathematical result concerns the ideal stationary solution.
