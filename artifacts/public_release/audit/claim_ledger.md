# Claim Ledger

| claim | status | expected | actual | evidence |
|---|---|---|---|---|
| varma_default_volterra | verified | 0.629 | 0.629 | storage/results/rebuttal/varma_ablation/architecture/varma_e2_three__N=6000__w=25__d=3__s=100/split=deterministic_random_Ntr=5000_Nte=1000_seed=0/quark_reservoir_channel_cupy_direct_arch_baseline_n5_R3_k2_lam0p1/seed=0/metrics.csv |
| varma_no_jl_volterra | verified | 0.606 | 0.606 | storage/results/rebuttal/varma_ablation/architecture/varma_e2_three__N=6000__w=25__d=3__s=100/split=deterministic_random_Ntr=5000_Nte=1000_seed=0/quark_reservoir_channel_cupy_direct_arch_no_jl_identity_pad_n5_R3_k2_lam0p1/seed=0/metrics.csv |
| varma_best_n6_volterra | verified | 0.479 | 0.479 | storage/results/rebuttal/varma_ablation/architecture/varma_e2_three__N=6000__w=25__d=3__s=100/split=deterministic_random_Ntr=5000_Nte=1000_seed=0/quark_reservoir_channel_cupy_direct_arch_sweep_n6_R3_k2_lam0p1_kernel_readout_retune/seed=0/metrics.csv |
| varma_shot_100 | verified | 1.048 | 1.048 | storage/results/rebuttal/varma_ablation/finite_shots/varma_e2_three__N=6000__w=25__d=3__s=100/split=deterministic_random_Ntr=5000_Nte=1000_seed=0/quark_reservoir_channel_cupy_direct_truncated64_shots_100_n5_R3_k2_lam0p1/seed=0/metrics.csv |
| varma_shot_10000 | verified | 0.701 | 0.701 | storage/results/rebuttal/varma_ablation/finite_shots/varma_e2_three__N=6000__w=25__d=3__s=100/split=deterministic_random_Ntr=5000_Nte=1000_seed=0/quark_reservoir_channel_cupy_direct_truncated64_shots_10000_n5_R3_k2_lam0p1/seed=0/metrics.csv |
| pilot_lambda | verified | 0.5 | 0.5 | storage/results/rebuttal/result_audit/canonical_real_world/temporal_budget_lambda_selection.csv |
| pilot_shots | verified | 3000 | 3000 | storage/results/rebuttal/result_audit/canonical_real_world/temporal_budget_shot_sweep.csv |
| pilot_shot_mean_delta | verified | 0.012 | 0.012 | storage/results/rebuttal/result_audit/canonical_real_world/temporal_budget_shot_sweep.csv |
| pilot_shot_max_delta | verified | 0.064 | 0.064 | storage/results/rebuttal/result_audit/canonical_real_world/temporal_budget_shot_sweep.csv |
| real_better_count | verified | 5/10 | 5/10 | storage/results/rebuttal/result_audit/canonical_real_world/temporal_budget_final_table.csv |
| real_w_range | verified | 24..7500 | 24..7500 | storage/results/rebuttal/result_audit/canonical_real_world/temporal_budget_final_table.csv |
| real_d_range | verified | 1..728 | 1..728 | storage/results/rebuttal/result_audit/canonical_real_world/temporal_budget_final_table.csv |
| ethanol_pair | verified | 0.958 vs 1.045 | 0.958 vs 1.045 | storage/results/rebuttal/result_audit/canonical_real_world/temporal_budget_final_table.csv |
| copper_pair | verified | 1.000 vs 1.069 | 1.000 vs 1.069 | storage/results/rebuttal/result_audit/canonical_real_world/temporal_budget_final_table.csv |
| live_fuel_pair | verified | 0.938 vs 0.907 | 0.938 vs 0.907 | storage/results/rebuttal/result_audit/canonical_real_world/temporal_budget_final_table.csv |
