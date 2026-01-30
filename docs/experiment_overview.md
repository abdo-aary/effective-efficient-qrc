# Experiment package overview

The `src/experiment/` package provides the *orchestration layer* used to reproduce the numerical validation experiments for the paper. It sits “above” the data, QRC, and model packages: it loads a dataset artifact from disk, constructs a model (typically a QRC feature extractor + Matérn KRR), runs training and evaluation, and then saves the resulting metrics and plots as reproducible artifacts.

## What the Experiment wrapper does

The central object is `src.experiment.experiment.Experiment`. Conceptually, it encapsulates the end-to-end flow:

1. **Load a dataset artifact** produced by `src.data` (e.g., synthetic VARMA windows with one or more labeling functionals).
2. **Instantiate (or receive) a model** from `src.models` (most commonly `QRCMaternKRRRegressor`).
3. **Fit once** to select kernel hyperparameters (e.g., Matérn \(\nu\), \(\xi\)) and to cache the computed feature maps so that follow-up evaluations do not recompute QRC features.
4. **Run a regularization sweep** over a grid of KRR regularization values \(\lambda_{\mathrm{reg}}\) and compute train/test metrics for each functional output.
5. **Save artifacts** (NPZ results, CSV metrics table, plots) into an experiment output directory.

This structure keeps the model implementation focused on learning/inference (in `src.models`), while the Experiment wrapper provides reproducible *experiment runs* driven by Hydra configuration.

## Configuration layout (Hydra)

Experiment runs are configured with Hydra configs stored under `src/experiment/conf/`.

Common patterns:

- `src/experiment/conf/data/…` controls dataset generation and the dataset artifact layout.
- `src/experiment/conf/model/…` controls model construction (QRC featurizer, runner, feature-map retriever, Matérn KRR tuning settings, preprocessing).
- A top-level experiment config (e.g., `reg_sweep_experiment.yaml`) composes `data` and `model` defaults and defines experiment-specific knobs such as the regularization grid and output locations.

Important conventions used throughout configs:

- `PROJECT_ROOT` is expected to be set (via environment variable) so that paths can be expressed relative to the repository root.
- Hydra’s runtime output directory (`${hydra:runtime.output_dir}`) is used as the default location to write experiment artifacts.

## Regularization sweep experiment

The regularization sweep is used to produce the Section 5.1 figure: train/test MSE as a function of \(\lambda_{\mathrm{reg}}\), with one curve per functional output.

### High-level flow

1. **Fit** the model once to select kernel hyperparameters (Matérn \(\nu,\xi\)) using the tuning procedure (train/val split).
2. **Sweep** \(\lambda_{\mathrm{reg}}\) on fixed features and fixed kernel hyperparameters, recomputing only the KRR dual weights \(\alpha\) per grid value and evaluating metrics.
3. **Export**:
   - raw sweep results (`.npz`)
   - a tidy metrics table (`.csv`)
   - train/test sweep plots (`.pdf` / `.png`)

### Relevant entry points

- **Script:** `src/experiment/scripts/run_reg_sweep_experiment.py`
- **Config:** `src/experiment/conf/reg_sweep_experiment.yaml`
- **Wrapper method:** `Experiment.run_reg_sweep_from_cfg(cfg.reg_sweep)` (calls the model’s `sweep_regularization(...)` internally)
- **Artifact helpers:** `Experiment.save_reg_sweep_artifacts(...)`

### Outputs

A typical run writes:

- `reg_sweep.npz` — raw outputs from the sweep (grid, train/test MSE arrays, per-output best indices, etc.).
- `reg_sweep_metrics.csv` — tabular view with one row per (functional, reg) pair.
- `reg_sweep_train.(pdf|png)` — train MSE vs reg with one curve per functional.
- `reg_sweep_test.(pdf|png)` — test MSE vs reg with one curve per functional.

Curve names are derived from the dataset metadata (when available) so plots are immediately readable (e.g., *Single step forecasting*, *Exponential fading*, *Volterra*).

## Typical usage

### 1) Generate a synthetic dataset (example)

The dataset generation entry point is `src/experiment/scripts/generate_data.py` and is driven by the `data` config tree. A typical run:

```bash
export PROJECT_ROOT=$(pwd)

python -m src.experiment.scripts.generate_data \
  data.sampling.N=16 data.sampling.w=6 data.sampling.d=2 data.sampling.s=2
```

This produces a dataset artifact under `storage/data/...` with the data file plus metadata/config for reproducibility.

### 2) Run the regularization sweep on an existing dataset

```bash
export PROJECT_ROOT=$(pwd)

python -m src.experiment.scripts.run_reg_sweep_experiment \
  dataset_path="/abs/path/to/storage/data/.../N=16__w=6__d=2__s=2"
```

Use Hydra overrides to adjust outputs or the sweep grid, for example:

```bash
python -m src.experiment.scripts.run_reg_sweep_experiment \
  dataset_path="/abs/path/to/dataset_artifact" \
  reg_sweep.reg_grid='[1e-12,1e-9,1e-6,1e-3]' \
  output.save_dir="${oc.env:PROJECT_ROOT}/storage/results"
```

## Extending experiments

The Experiment wrapper is intentionally small: extending the project typically means adding a new script/config pair and a new `Experiment.run_*` method that calls into existing model capabilities, then exporting the results in a consistent artifact format.

Examples of natural extensions:

- sweeping QRC complexity parameters (\(n\), number of reservoirs \(R\), observable locality),
- comparing QRC–Matérn KRR with classical baselines (e.g., ESN),
- logging additional metrics (NLPD, calibration, per-functional errors) into the exported CSV.
