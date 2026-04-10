# Finite-Shot Ablation

| w | d | task | method | test_mse | train_mse | feature_dim | raw_dim | n_train | n_test |
|---|---|---|---|---|---|---|---|---|---|
| 25 | 3 | exp_fading_linear | quark_reservoir_channel_cupy_direct_truncated64_shots_10000_n5_R3_k2_lam0p1 | 0.587 | 5.357e-09 | 315 | 75 | 5000 | 1000 |
| 25 | 3 | exp_fading_linear | quark_reservoir_channel_cupy_direct_truncated64_shots_1000_n5_R3_k2_lam0p1 | 0.6895 | 1.381e-12 | 315 | 75 | 5000 | 1000 |
| 25 | 3 | exp_fading_linear | quark_reservoir_channel_cupy_direct_truncated64_shots_100_n5_R3_k2_lam0p1 | 0.9599 | 1.543e-12 | 315 | 75 | 5000 | 1000 |
| 25 | 3 | exp_fading_linear | quark_reservoir_channel_cupy_direct_truncated64_shots_5000_n5_R3_k2_lam0p1 | 0.6123 | 2.974e-12 | 315 | 75 | 5000 | 1000 |
| 25 | 3 | exp_fading_linear | quark_reservoir_channel_cupy_direct_truncated64_shots_500_n5_R3_k2_lam0p1 | 0.7472 | 1.462e-12 | 315 | 75 | 5000 | 1000 |
| 25 | 3 | one_step_forecast | quark_reservoir_channel_cupy_direct_truncated64_shots_10000_n5_R3_k2_lam0p1 | 0.001843 | 1.263e-12 | 315 | 75 | 5000 | 1000 |
| 25 | 3 | one_step_forecast | quark_reservoir_channel_cupy_direct_truncated64_shots_1000_n5_R3_k2_lam0p1 | 0.01065 | 7.013e-13 | 315 | 75 | 5000 | 1000 |
| 25 | 3 | one_step_forecast | quark_reservoir_channel_cupy_direct_truncated64_shots_100_n5_R3_k2_lam0p1 | 0.0789 | 5.133e-11 | 315 | 75 | 5000 | 1000 |
| 25 | 3 | one_step_forecast | quark_reservoir_channel_cupy_direct_truncated64_shots_5000_n5_R3_k2_lam0p1 | 0.002894 | 3.441e-12 | 315 | 75 | 5000 | 1000 |
| 25 | 3 | one_step_forecast | quark_reservoir_channel_cupy_direct_truncated64_shots_500_n5_R3_k2_lam0p1 | 0.01899 | 1.499e-12 | 315 | 75 | 5000 | 1000 |
| 25 | 3 | volterra | quark_reservoir_channel_cupy_direct_truncated64_shots_10000_n5_R3_k2_lam0p1 | 0.7015 | 1.191e-10 | 315 | 75 | 5000 | 1000 |
| 25 | 3 | volterra | quark_reservoir_channel_cupy_direct_truncated64_shots_1000_n5_R3_k2_lam0p1 | 0.8346 | 1.517e-12 | 315 | 75 | 5000 | 1000 |
| 25 | 3 | volterra | quark_reservoir_channel_cupy_direct_truncated64_shots_100_n5_R3_k2_lam0p1 | 1.048 | 2.473e-12 | 315 | 75 | 5000 | 1000 |
| 25 | 3 | volterra | quark_reservoir_channel_cupy_direct_truncated64_shots_5000_n5_R3_k2_lam0p1 | 0.745 | 1.598e-12 | 315 | 75 | 5000 | 1000 |
| 25 | 3 | volterra | quark_reservoir_channel_cupy_direct_truncated64_shots_500_n5_R3_k2_lam0p1 | 0.9016 | 1.827e-12 | 315 | 75 | 5000 | 1000 |
