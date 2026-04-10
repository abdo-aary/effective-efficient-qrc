# Canonical VARMA Claim Table

| claim | test_mse | evidence |
|---|---:|---|
| varma_default_volterra_exact | 0.629388 | `storage/results/rebuttal/varma_ablation/architecture/varma_e2_three__N=6000__w=25__d=3__s=100/split=deterministic_random_Ntr=5000_Nte=1000_seed=0/quark_reservoir_channel_cupy_direct_arch_baseline_n5_R3_k2_lam0p1/seed=0/metrics.csv` |
| varma_no_jl_volterra_exact | 0.606072 | `storage/results/rebuttal/varma_ablation/architecture/varma_e2_three__N=6000__w=25__d=3__s=100/split=deterministic_random_Ntr=5000_Nte=1000_seed=0/quark_reservoir_channel_cupy_direct_arch_no_jl_identity_pad_n5_R3_k2_lam0p1/seed=0/metrics.csv` |
| varma_best_n6_volterra_retuned | 0.478788 | `storage/results/rebuttal/varma_ablation/architecture/varma_e2_three__N=6000__w=25__d=3__s=100/split=deterministic_random_Ntr=5000_Nte=1000_seed=0/quark_reservoir_channel_cupy_direct_arch_sweep_n6_R3_k2_lam0p1_kernel_readout_retune/seed=0/metrics.csv` |
| varma_shots_100_volterra | 1.047676 | `storage/results/rebuttal/varma_ablation/finite_shots/varma_e2_three__N=6000__w=25__d=3__s=100/split=deterministic_random_Ntr=5000_Nte=1000_seed=0/quark_reservoir_channel_cupy_direct_truncated64_shots_100_n5_R3_k2_lam0p1/seed=0/metrics.csv` |
| varma_shots_500_volterra | 0.901627 | `storage/results/rebuttal/varma_ablation/finite_shots/varma_e2_three__N=6000__w=25__d=3__s=100/split=deterministic_random_Ntr=5000_Nte=1000_seed=0/quark_reservoir_channel_cupy_direct_truncated64_shots_500_n5_R3_k2_lam0p1/seed=0/metrics.csv` |
| varma_shots_1000_volterra | 0.834551 | `storage/results/rebuttal/varma_ablation/finite_shots/varma_e2_three__N=6000__w=25__d=3__s=100/split=deterministic_random_Ntr=5000_Nte=1000_seed=0/quark_reservoir_channel_cupy_direct_truncated64_shots_1000_n5_R3_k2_lam0p1/seed=0/metrics.csv` |
| varma_shots_5000_volterra | 0.744970 | `storage/results/rebuttal/varma_ablation/finite_shots/varma_e2_three__N=6000__w=25__d=3__s=100/split=deterministic_random_Ntr=5000_Nte=1000_seed=0/quark_reservoir_channel_cupy_direct_truncated64_shots_5000_n5_R3_k2_lam0p1/seed=0/metrics.csv` |
| varma_shots_10000_volterra | 0.701464 | `storage/results/rebuttal/varma_ablation/finite_shots/varma_e2_three__N=6000__w=25__d=3__s=100/split=deterministic_random_Ntr=5000_Nte=1000_seed=0/quark_reservoir_channel_cupy_direct_truncated64_shots_10000_n5_R3_k2_lam0p1/seed=0/metrics.csv` |
