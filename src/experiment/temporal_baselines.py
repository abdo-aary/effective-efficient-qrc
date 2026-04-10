"""Public temporal-baseline surface for the QuaRK release.

This module intentionally exposes only the retained classical temporal baseline
used in the public release: ESN state generation with an optional Matérn-KRR
readout.  The wider historical baseline collection remains an internal
implementation detail.
"""

from __future__ import annotations

from src.experiment.classical_baselines import (
    ESN_DENSITY,
    ESN_INPUT_SCALE_GRID,
    ESN_LEAK_RATE_GRID,
    ESN_SPECTRAL_RADIUS_GRID,
    FEATURE_DIM_METHODS,
    MATERN_NU_GRID,
    MATERN_TUNING_REG,
    MATERN_XI_BOUNDS,
    MATERN_XI_MAXITER,
    MATERN_TUNE_MAX_TRAIN,
    MATERN_TUNE_MAX_VAL,
    METRICS_COLUMNS,
    RIDGE_ALPHA_GRID,
    REG_GRID,
    BenchmarkData,
    SplitData,
    build_markdown_table,
    fit_esn_matern_krr,
    fit_esn_matern_krr_from_saved_params,
    fit_esn_ridge,
    fit_matern_krr_features,
    fit_ridge_features,
    json_safe,
    make_esn_features,
    make_metric_rows,
    matern_nu_grid_for_backend,
    mean_squared_error,
    method_run_dir,
    standardize_features,
    standardize_flattened_windows,
    write_baseline_plot,
    write_dict_csv,
    write_metrics_csv,
    write_run_artifacts,
)


TEMPORAL_BASELINE_METHODS = {
    "esn",
    "esn_matern_krr",
}

__all__ = [
    "BenchmarkData",
    "ESN_DENSITY",
    "ESN_INPUT_SCALE_GRID",
    "ESN_LEAK_RATE_GRID",
    "ESN_SPECTRAL_RADIUS_GRID",
    "FEATURE_DIM_METHODS",
    "MATERN_NU_GRID",
    "MATERN_TUNING_REG",
    "MATERN_XI_BOUNDS",
    "MATERN_XI_MAXITER",
    "MATERN_TUNE_MAX_TRAIN",
    "MATERN_TUNE_MAX_VAL",
    "METRICS_COLUMNS",
    "RIDGE_ALPHA_GRID",
    "REG_GRID",
    "SplitData",
    "TEMPORAL_BASELINE_METHODS",
    "build_markdown_table",
    "fit_esn_matern_krr",
    "fit_esn_matern_krr_from_saved_params",
    "fit_esn_ridge",
    "fit_matern_krr_features",
    "fit_ridge_features",
    "json_safe",
    "make_esn_features",
    "make_metric_rows",
    "matern_nu_grid_for_backend",
    "mean_squared_error",
    "method_run_dir",
    "standardize_features",
    "standardize_flattened_windows",
    "write_baseline_plot",
    "write_dict_csv",
    "write_metrics_csv",
    "write_run_artifacts",
]
