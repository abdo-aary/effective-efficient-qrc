# QuaRK empirical-specification audit

## Reviewed state

- Repository: `abdo-aary/quark`
- Reviewed commit: `34e8d910169c729ab5e7ae07d87aac4cccf761f3`
- Commit date: 2026-07-29
- Manuscript source revised in this deliverable: `docs/latex/updated`-equivalent project supplied in the conversation
- Protocol version introduced here: `quark-empirical-v1`

No numerical result was invented or run as part of this task. The revised manuscript specifies the experiments and leaves numerical conclusions absent until checksum-validated artifacts exist.

## A. Repository-support audit

| Experiment | Required capability | Current implementation path | Status at reviewed commit | Missing implementation | Estimated implementation effort |
|---|---|---|---|---|---|
| E1 functional hierarchy | Stable VARMA; genuine future target; exponential, Volterra, delay and cross-lag targets; exact GPU features; finite KRR grid; conditioning/RKHS metrics | `src/data/generate/varma.py`; `src/data/label/functionals.py`; `src/backends/nvidia/backend.py`; `src/models/qrc_matern_krr.py` | **Partial**. VARMA, exponential and Volterra exist. Exact NVIDIA features exist. The class called `OneStepForecastFunctional` currently uses `X_win[-1]`, so it is current-point decoding, not future forecasting. | Add future-context target interface; delayed recall; cross-lag; finite-grid KRR selector (current “grid” continuously optimizes xi); E1 driver and diagnostics. | Moderate, 3-5 developer days plus smoke runs. |
| E2 memory and cross-family | HMM/GARCH generators, stationary initialization/burn-in, filtering and volatility teachers, contraction sweep | VARMA support exists; family definitions currently only in LaTeX | **Mostly unsupported beyond VARMA**. | Add HMM and GARCH generators, context artifacts, stationary/initialization checks, target implementations, E2 matrix driver. | Large, 7-12 days. |
| E3 ambient dimension | Shared latent process, sparse/distributed embeddings, nested nuisance streams, JL distortion, projection/runtime metrics | Existing `src/experiment/varma_ablation.py` contains d sweeps and controls | **Partial**. Existing sweep generates full d-dimensional VARMA and does not implement the revised latent-embedding design or distortion statistic. | Add latent embedding dataset builder; pair sampler; distortion metric; width-sensitive driver; timing/memory instrumentation. | Moderate-large, 5-8 days. |
| E4 classical calibration | ESN, matched random map, classical JL, raw references, common readout grid, artifact aggregation | `src/experiment/classical_baselines.py`; retained rebuttal artifacts | **Methods implemented; protocol aggregation missing**. Existing grids and split logic are not exactly the new finite grid. | Normalize baseline API, use common finite KRR grid, add seven-scenario aggregator and matching audit fields. | Moderate, 3-5 days. |
| E5 resource sweeps | One-factor n/R/k/lambda sweeps, exact GPU timing, peak memory, Pareto extraction | `src/experiment/varma_ablation.py`; `src/backends/nvidia/backend.py` | **Largely implemented in legacy experiment form**. | Move to typed backend-neutral configuration; deduplicate by program fingerprint; add synchronized timing, memory, resource fields, Pareto script. | Moderate, 3-5 days. |
| E6 finite shots | Exact reference; NVIDIA CSMoM; Aer shadow-only diagnostic; frozen/noisy readouts; repeated named seeds; feature metrics | `src/estimators/csmom.py`; `src/backends/nvidia/csmom.py`; `src/backends/aer/backend.py`; deterministic seed streams | **Core estimator and both execution semantics implemented**. | Add staged E6 driver, paired structural/measurement roots, feature-error metrics, cached exact readouts, noisy-training mode and artifact aggregation. | Moderate-large, 5-8 days. |
| E7 chaos | Lorenz and Mackey-Glass generators, chronological guards, persistence skill | Only manuscript definitions were found | **Unsupported**. | Add two generators, numerical-integrity tests, horizon target builder, E7 driver and plots. | Moderate-large, 5-8 days. |
| E8 real sanity | Existing real datasets/splits, exact API execution, controls, guardrails | `src/data/real_tser.py`; `src/experiment/real_world_rebuttal.py`; retained real benchmark artifacts | **Data and legacy runners implemented; exact requested datasets retained**. | Add typed API driver for Beijing PM2.5, live fuel moisture and hydraulic systems; common grid; resource guardrail and new aggregate. | Moderate, 3-6 days. |
| E9 IBM hardware | CSMoM only; pre-sampled trajectories; local ladder; Runtime submit/resume; backend/layout selection; chunking/retry; provenance | `src/backends/ibm/backend.py`, `trajectories.py`, `csmom.py`, `jobs.py`; `tests/tests_api/test_backends.py` | **Strong core, incomplete experiment lifecycle**. Exact-on-IBM is correctly rejected. Local trajectory convergence is tested. | Add backend/pair selector, fixed-layout transpilation, deterministic <=100-PUB chunking, multi-handle resume/retry, richer calibration capture, E9 driver and plots. No mitigation in v1. | Moderate-large, 5-8 days plus QPU access. |

### Cross-cutting implementation findings

1. `src/api.py` exports the intended backend-neutral program, estimators and three adapters.
2. `QuaRKProgram` is immutable and fingerprinted; `FeatureBatch` preserves `(N,R,K)` and explicit ordering.
3. `SeedBundle` already provides the named streams required by this protocol, including independent reset-trajectory and shadow-basis streams.
4. The Aer oracle uses an independent SWAP dilation and is deliberately bounded to smoke-scale workloads.
5. The NVIDIA exact backend is the appropriate production path and directly returns expectations; it is currently CuPy-only.
6. NVIDIA CSMoM samples a pure-state branch from the exact channel ensemble and then samples local-Pauli measurements. Thus it includes trajectory and measurement variance.
7. IBM CSMoM groups identical `(window,reservoir,last-reset-suffix,basis)` snapshots and records suffix/reset provenance.
8. `src/artifacts/run.py` provides immutable `quark.run/v1` feature artifacts with checksums, but a protocol-level run schema is still needed for failed/incomplete runs, candidate selection, multi-job hardware aggregation and long-form metrics.
9. Hydra includes new backend and estimator groups, but legacy runner/retriever groups remain in the default model tree. Migration is therefore transitional rather than complete.
10. `QRCMaternKRRRegressor` still contains continuous xi optimization and top-level Qiskit coupling. Main protocol runs need the new finite grid and should not rely on the continuous tuner.

## B. Experimental master matrix

The normative values are in manuscript Appendix C (“Implementation-Ready Experimental Protocol”). The compact execution view is:

| ID | Feature workload | Data/task roots | Exact/CSMoM | Controls | Approximate new feature-run count |
|---|---:|---|---|---|---:|
| E1 | Reference `n=5,R=3,k=2,lambda=.5,w=25` | 5 VARMA roots; 8 tasks/readouts share features | Exact NVIDIA | Controls only for three reference tasks via E4 | 5 exact feature tensors |
| E2a | 2 persistence x 5 lambda | 5 roots; 3 delays share features | Exact NVIDIA | None in sweep | 50 exact tensors |
| E2b | 3 families x 3 persistence | 5 roots; shared/family targets share features | Exact NVIDIA | ESN and random map | 45 QuaRK tensors + control fits |
| E3 | 2 observation types x 5 d plus width checks | 5 roots; two targets share features | Exact NVIDIA | JL, random, raw ridge; raw Matérn d<=30 | 90 QuaRK tensors + control fits |
| E4 | Reuse seven scenarios | Reuses E1-E3 | None new | Consolidates all retained methods | 0 feature tensors |
| E5 | 11 unique one-factor programs | 5 roots; two targets share features | Exact NVIDIA | None | 55 exact tensors, each timed after warm-up |
| E6 frozen | 5 S values | 3 structural x 5 measurement roots; 256 windows | NVIDIA CSMoM | exact reference; 16-window Aer shadow-only | 75 large CSMoM estimates + 15 Aer diagnostics |
| E6 noisy train | 3 S values | 3 structural x 3 measurement roots; 896 windows total | NVIDIA CSMoM | exact reference | 27 train/eval estimates |
| E7 | 6 Lorenz and 12 MG trajectory/noise settings | 3 roots; horizons reuse features | Exact NVIDIA | ESN and random map | 18 QuaRK feature tensors |
| E8 | 3 retained datasets | 5 representation roots | Exact NVIDIA | ESN, random, raw ridge | 15 QuaRK tensors + control fits |
| E9 | 8 windows, 192 snapshots | one structural plan; two hardware periods | Aer exact, Aer CSMoM, IBM-local CSMoM, IBM QPU CSMoM | Frozen exact readout | <=360 groups/period; 1536 QPU shots/period |

### Shared selection rules

- Structural roots: `1101,1102,1103,1104,1105`.
- Synthetic split: 4000 inner train / 1000 validation / 1000 test; refit on 5000 outer train.
- Common finite Matérn grid: 3 nu x 6 xi x 5 lambda = 90 candidates.
- No test selection, adaptive grid expansion or result-driven exclusions.
- All stochastic methods pair data/split/task roots; method-specific randomness has its own named stream.

## C. Main-paper display plan

| Display | Experiment/artifacts | Exact content | Scientific role |
|---|---|---|---|
| Figure 3, “Functional and sample-complexity profile” | E1 aggregate | (a) KRR regularization path with RKHS norm; (b) NRMSE vs training prefix; (c) NRMSE vs delay | Separates readout capacity, sample size and explicit memory difficulty. |
| Figure 4, “Memory and dependence map” | E2a/E2b | (a) lambda x delay/persistence heat map; (b) paired NRMSE differences across three families | Shows whether memory setting changes with task/process and whether findings transfer. |
| Figure 5, “Fixed-width ambient-dimension study” | E3 | (a) NRMSE vs d for sparse/distributed signal; (b) q95 JL distortion vs d/n | Tests fixed encoded width without claiming d-independent end-to-end cost. |
| Table 2, “Classical calibration” | E4 | Seven scenario rows; method NRMSE mean/std, paired difference, p, candidate count, time, memory | Gives transparent representation-level comparison without a benchmark zoo. |
| Figure 6, “Resource and measurement operating points” | E5/E6 | (a) NRMSE vs feature runtime, Pareto points; (b) frozen-readout degradation vs snapshots | Connects resources and finite measurement to predictive behavior. |
| Figure 7 + compact table, “IBM transfer ladder” | E9 | Feature RMSE by ladder layer; exact vs QPU predictions; backend/layout/depth/count/shot/status table | Compact hardware feasibility and perturbation decomposition. |

E7 and E8 remain appendix displays unless their results materially change a central conclusion. This keeps the main paper theory-led.

## D. Appendix display plan

- E1: all task/seed/path rows, Gram condition numbers and prediction traces.
- E2: complete low/medium/high family tables, every lambda/delay cell and selected parameters.
- E3: n=3/5/8 details, pairwise distortion distributions, projection/feature/readout timings, host/device memory.
- E4: long-form paired results and explicit matching/candidate-count audit.
- E5: all 11 configurations, resource-exhausted rows, runtime repetitions and Pareto derivation.
- E6: coordinate/per-example feature errors, all shadow replicates, Aer shadow-only diagnostic, noisy-feature training, retained raw-snapshot audit replicates.
- E7: every horizon/noise/root curve, persistence baseline and numerical-integrity ledger.
- E8: dataset-level predictions, all method roots, guardrail decisions and no average rank.
- E9: structural-plan checksum, per-observable bias, per-window errors, transpilation rows, physical layouts, backend properties, Runtime handles, chunk/retry ledger, missing-result accounting and both calibration periods separately.
- Global: all failed/incomplete runs and protocol deviations.

## F. Implementation backlog

### Required before full runs

1. **Protocol schema and configs** (moderate)
   - Add `quark-empirical-v1` typed configuration objects and Hydra mappings.
   - Add common finite KRR grid and lexicographic tie rule.
   - Add protocol-level run status/aggregate schema on top of `quark.run/v1`.
   - Tests: config round-trip, hash stability, no test-index access during selection.

2. **Targets and context interface** (moderate)
   - Add immutable teacher context with future observation, HMM filter and GARCH variance fields.
   - Correct the current forecast/current-point mismatch.
   - Add delay and cross-lag targets.
   - Tests: no future leakage, boundedness, fixed directions, deterministic context checksums.

3. **HMM and GARCH generators** (large)
   - Implement exact stationary HMM initialization and forward-filter labels.
   - Implement GARCH burn-in/stationarity diagnostics and latent conditional-variance labels.
   - Tests against analytical moments/transition frequencies and deterministic seeds.

4. **Latent high-dimensional dataset builder** (moderate)
   - Add sparse/distributed observation embeddings and nested nuisance streams.
   - Add JL pair sampler/distortion metrics.
   - Test that latent teacher and first three sparse coordinates are identical across d.

5. **Experiment drivers E1-E6** (large)
   - Add shared feature cache, finite-grid readout runner, timings/memory, long-form metrics and resumability.
   - Reuse E2/E3 artifacts in E4; forbid re-extraction.

6. **Chaos generators and E7** (moderate-large)
   - Add fixed integrators/checksums/guard intervals, persistence metrics and plots.

7. **Typed E8 driver** (moderate)
   - Use retained datasets and official splits; implement raw-Matérn guardrail.

8. **IBM E9 lifecycle** (moderate-large)
   - Backend/pair selector; fixed initial layout; ordered <=100-PUB chunks; multi-job handle artifact; one retry; exact snapshot accounting; richer backend properties.
   - Tests with fake Runtime jobs and local Aer before QPU submission.

9. **Plot/table generation** (moderate)
   - Pure scripts that consume aggregate artifacts only.
   - Snapshot/golden tests for schemas and displayed row counts.

### Inexpensive optional enhancements

- Add complex64-versus-complex128 E5 precision diagnostic after exact conformance is verified.
- Add one optional current-point decoding smoke task, clearly not called forecasting.
- Add measurement mitigation as a versioned E9 secondary study only after backend support and tests exist.
- Add bootstrap utilities shared across E1-E9.

### Future work not required for TMLR

- Broad IBM backend/device study.
- Deep-circuit or many-qubit hardware scaling.
- Automatic continuous architecture search.
- Full real-world TSER benchmark.
- Formal analysis of noisy-feature retraining or hardware noise.
- cuStateVec engine unless measured benchmarks justify it.

## G. Validation and unresolved items

The final build report is generated after LaTeX compilation. Scientifically unresolved until implementation:

- HMM/GARCH/chaos generators are not in the reviewed repository.
- Genuine future forecasting, delay, cross-lag and teacher-context targets are missing.
- The main KRR tuner is not yet the finite 90-candidate grid.
- Current E3 code does not use the revised latent embedding.
- IBM backend does not yet select a physical pair, force a fixed layout, chunk multiple Runtime jobs, retry missing chunks or apply mitigation.
- No IBM QPU result exists; E9 is a protocol only.
- No numerical empirical conclusion should appear until all required artifacts pass audit.
