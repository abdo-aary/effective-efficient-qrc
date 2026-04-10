# Final Rebuttal Table

| Dataset | w | d | raw_dim | seeds | QuaRK mean +- std | ESN+Matérn mean +- std | delta mean | QuaRK wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| iron_concentration | 1716 | 1 | 1716 | 5 | 1.0617 +- 0.0058 | 1.1552 +- 0.0508 | -0.0935 | 4 |
| gas_sensor_array_ethanol | 7500 | 1 | 7500 | 5 | 0.9579 +- 0.0263 | 1.0446 +- 0.1064 | -0.0867 | 3 |
| gas_sensor_array_acetone | 7500 | 1 | 7500 | 5 | 0.9779 +- 0.0249 | 1.0620 +- 0.0954 | -0.0840 | 3 |
| copper_concentration | 2542 | 1 | 2542 | 5 | 1.0003 +- 0.0084 | 1.0693 +- 0.1219 | -0.0691 | 2 |
| manganese_concentration | 1716 | 1 | 1716 | 5 | 0.9593 +- 0.0152 | 1.0278 +- 0.0884 | -0.0685 | 3 |
| live_fuel_moisture | 365 | 7 | 2555 | 5 | 0.9383 +- 0.0056 | 0.9074 +- 0.0088 | +0.0309 | 0 |
| electric_motor_temperature | 60 | 6 | 360 | 5 | 0.5314 +- 0.0207 | 0.3742 +- 0.0178 | +0.1572 | 0 |
| beijing_pm25 | 24 | 9 | 216 | 5 | 0.9327 +- 0.0223 | 0.7120 +- 0.0042 | +0.2207 | 0 |
| hydraulic_systems | 60 | 728 | 43680 | 5 | 0.9389 +- 0.0406 | 0.3222 +- 0.0429 | +0.6167 | 0 |
| benzene_concentration | 240 | 8 | 1920 | 5 | 0.9051 +- 0.0921 | 0.0231 +- 0.0029 | +0.8820 | 0 |

QuaRK uses fixed n=5, R=3, k=2, D=315 with global lambda_0=0.5 and 3000 simulated shadow shots; ESN+Matérn uses 315 hidden states/features.
