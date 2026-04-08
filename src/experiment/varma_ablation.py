"""VARMA rebuttal ablation utilities.

This module prepares controlled VARMA datasets with explicit train/test
sidecars, then reuses the rebuttal baseline and QuaRK runners on those
artifacts.  The two intended studies are:

- architecture ablations at fixed ``w=25,d=3``;
- ``w,d`` scaling ablations at fixed ``n=5,R=3``.
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.data.factory import generate_and_save_dataset, load_windows_dataset
from src.experiment.classical_baselines import (
    CLASSICAL_METHODS,
    FEATURE_DIM_METHODS,
    METRICS_COLUMNS,
    TASK_NAMES_E2_THREE,
    build_markdown_table,
    build_wide_rows,
    write_dict_csv,
    write_metrics_csv,
)
from src.experiment.real_world_rebuttal import (
    READOUT_RETUNE_LAMBDA_GRID,
    READOUT_RETUNE_XI_GRID,
    aggregate_real_world_results,
    read_real_metric_rows,
    run_cached_quark_readout_retune,
    run_real_classical_baseline,
    run_real_quark,
)
from src.settings import PROJECT_ROOT_PATH


PROJECT_ROOT = Path(PROJECT_ROOT_PATH)
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))

DEFAULT_VARMA_ABLATION_DATA_ROOT = PROJECT_ROOT / "storage/data/rebuttal/varma_ablation"
DEFAULT_VARMA_ABLATION_RESULTS_ROOT = PROJECT_ROOT / "storage/results/rebuttal/varma_ablation"

VARMA_ABLATION_CLASSICAL_METHODS = (
    "raw_ridge",
    "raw_matern_krr",
    "matched_random_features_matern_krr",
    "esn",
)
VARMA_W_GRID = (10, 25, 50, 100)
VARMA_D_GRID = (3, 10, 30, 100, 300, 500)
VARMA_SHOT_GRID = (100, 500, 1000, 5000, 10000)
VARMA_BASELINE_FEATURE_DIM = 315


@dataclass(frozen=True)
class VarmaDatasetSpec:
    """One generated VARMA dataset shape."""

    w: int
    d: int
    N_train: int = 5000
    N_test: int = 1000
    s: int = 100
    seed: int = 12345
    split_seed: int = 0

    @property
    def N(self) -> int:
        return int(self.N_train) + int(self.N_test)

    @property
    def dataset_id(self) -> str:
        return f"varma_e2_three__N={self.N}__w={int(self.w)}__d={int(self.d)}__s={int(self.s)}"

    @property
    def dataset_dir_name(self) -> str:
        return (
            f"Ntrain={int(self.N_train)}__Ntest={int(self.N_test)}__"
            f"w={int(self.w)}__d={int(self.d)}__s={int(self.s)}__seed={int(self.seed)}"
        )

    @property
    def artifact_name(self) -> str:
        return f"varma__e2_three__N={self.N}__w={int(self.w)}__d={int(self.d)}__s={int(self.s)}__seed={int(self.seed)}"

    @property
    def split_source(self) -> str:
        return f"deterministic_random_Ntr={int(self.N_train)}_Nte={int(self.N_test)}_seed={int(self.split_seed)}"


@dataclass(frozen=True)
class VarmaQuarkRegime:
    """A named QuaRK ablation regime."""

    name: str
    overrides: tuple[str, ...]
    runner: str = "reservoir_channel_cupy_direct"
    feature_dim: int = VARMA_BASELINE_FEATURE_DIM


def architecture_regimes() -> list[VarmaQuarkRegime]:
    """Return one-factor QuaRK architecture ablations around the paper baseline."""

    base = (
        "model/qrc/features/retriever=exact",
        "model.qrc.cfg.num_qubits=5",
        "model.qrc.features.observables.locality=2",
        "model.qrc.pubs.num_reservoirs=3",
        "model.qrc.pubs.lam_0=0.1",
    )

    regimes = [
        VarmaQuarkRegime("arch_baseline_n5_R3_k2_lam0p1", base, feature_dim=315),
        VarmaQuarkRegime(
            "arch_no_jl_identity_pad_n5_R3_k2_lam0p1",
            base + ("model.qrc.cfg.projection_mode=identity_pad",),
            feature_dim=315,
        ),
        VarmaQuarkRegime(
            "arch_zero_dynamics_n5_R3_k2_lam0p1",
            base + ("model.qrc.pubs.dynamics_mode=zero",),
            feature_dim=315,
        ),
    ]

    for n in (3, 6, 8, 10):
        feature_dim = feature_dim_for_local_paulis(num_qubits=n, num_reservoirs=3, locality=2)
        regimes.append(
            VarmaQuarkRegime(
                f"arch_sweep_n{n}_R3_k2_lam0p1",
                replace_override(base, "model.qrc.cfg.num_qubits", n),
                feature_dim=feature_dim,
            )
        )
    for R in (1, 5, 8):
        feature_dim = feature_dim_for_local_paulis(num_qubits=5, num_reservoirs=R, locality=2)
        regimes.append(
            VarmaQuarkRegime(
                f"arch_sweep_n5_R{R}_k2_lam0p1",
                replace_override(base, "model.qrc.pubs.num_reservoirs", R),
                feature_dim=feature_dim,
            )
        )
    for k in (1, 3):
        feature_dim = feature_dim_for_local_paulis(num_qubits=5, num_reservoirs=3, locality=k)
        regimes.append(
            VarmaQuarkRegime(
                f"arch_sweep_n5_R3_k{k}_lam0p1",
                replace_override(base, "model.qrc.features.observables.locality", k),
                feature_dim=feature_dim,
            )
        )
    for lam in (0.01, 0.05, 0.3, 0.5, 0.8, 0.95):
        regimes.append(
            VarmaQuarkRegime(
                f"arch_sweep_n5_R3_k2_lam{format_float_token(lam)}",
                replace_override(base, "model.qrc.pubs.lam_0", lam),
                feature_dim=315,
            )
        )
    return regimes


def varma_grid_regime(*, runner: str = "reservoir_channel_cupy_direct") -> VarmaQuarkRegime:
    return VarmaQuarkRegime(
        "varma_grid_fixed_n5_R3_k2_lam0p1",
        (
            "model/qrc/features/retriever=exact",
            "model.qrc.cfg.num_qubits=5",
            "model.qrc.features.observables.locality=2",
            "model.qrc.pubs.num_reservoirs=3",
            "model.qrc.pubs.lam_0=0.1",
        ),
        runner=runner,
        feature_dim=315,
    )


def shot_regimes(shots: Sequence[int] = VARMA_SHOT_GRID) -> list[VarmaQuarkRegime]:
    regimes: list[VarmaQuarkRegime] = []
    for shot_count in shots:
        regimes.append(
            VarmaQuarkRegime(
                f"shots_{int(shot_count)}_n5_R3_k2_lam0p1",
                (
                    "model/qrc/features/retriever=cs",
                    "model.qrc.cfg.num_qubits=5",
                    "model.qrc.features.observables.locality=2",
                    "model.qrc.pubs.num_reservoirs=3",
                    "model.qrc.pubs.lam_0=0.1",
                    f"model.qrc.features.retriever.kwargs.shots={int(shot_count)}",
                ),
                feature_dim=315,
            )
        )
    return regimes


def feature_dim_for_local_paulis(*, num_qubits: int, num_reservoirs: int, locality: int) -> int:
    from math import comb

    n = int(num_qubits)
    k = min(int(locality), n)
    per_reservoir = sum(comb(n, order) * (3**order) for order in range(1, k + 1))
    return int(num_reservoirs) * int(per_reservoir)


def replace_override(overrides: Sequence[str], key: str, value: Any) -> tuple[str, ...]:
    prefix = f"{key}="
    out = [str(item) for item in overrides if not str(item).startswith(prefix)]
    out.append(f"{key}={value}")
    return tuple(out)


def format_float_token(value: float) -> str:
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return text


def architecture_dataset_spec(
    *,
    N_train: int = 5000,
    N_test: int = 1000,
    s: int = 100,
    seed: int = 12345,
    split_seed: int = 0,
) -> VarmaDatasetSpec:
    return VarmaDatasetSpec(w=25, d=3, N_train=N_train, N_test=N_test, s=s, seed=seed, split_seed=split_seed)


def varma_grid_specs(
    *,
    w_grid: Sequence[int] = VARMA_W_GRID,
    d_grid: Sequence[int] = VARMA_D_GRID,
    N_train: int = 5000,
    N_test: int = 1000,
    s: int = 100,
    seed: int = 12345,
    split_seed: int = 0,
) -> list[VarmaDatasetSpec]:
    return [
        VarmaDatasetSpec(w=int(w), d=int(d), N_train=N_train, N_test=N_test, s=s, seed=seed, split_seed=split_seed)
        for w in w_grid
        for d in d_grid
    ]


def prepare_varma_ablation_dataset(
    spec: VarmaDatasetSpec,
    *,
    data_root: str | Path = DEFAULT_VARMA_ABLATION_DATA_ROOT,
    overwrite: bool = False,
) -> Path:
    """Generate one VARMA ablation dataset and its deterministic split sidecars."""

    data_root = Path(data_root)
    dataset_dir = data_root / spec.dataset_dir_name
    dataset_path = dataset_dir / f"{spec.artifact_name}.npz"
    if dataset_path.exists() and not overwrite:
        ensure_varma_split_sidecars(dataset_dir, spec)
        return dataset_dir

    config_dir = PROJECT_ROOT / "src/experiment/conf/data"
    save_dir_text = str(dataset_dir).replace("\\", "\\\\").replace("'", "\\'")
    overrides = [
        f"sampling.N={int(spec.N)}",
        f"sampling.w={int(spec.w)}",
        f"sampling.d={int(spec.d)}",
        f"sampling.s={int(spec.s)}",
        f"seed={int(spec.seed)}",
        f"output.save_dir='{save_dir_text}'",
        "output.name=auto",
        "output.format=npz",
        f"output.overwrite={str(bool(overwrite)).lower()}",
    ]
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    _ds, artifact = generate_and_save_dataset(cfg)
    ensure_varma_split_sidecars(artifact.root, spec)
    return artifact.root


def ensure_varma_split_sidecars(dataset_dir: Path, spec: VarmaDatasetSpec) -> None:
    dataset_path = dataset_dir / f"{spec.artifact_name}.npz"
    if not dataset_path.exists():
        candidates = sorted(dataset_dir.glob("*.npz"))
        if len(candidates) == 1:
            dataset_path = candidates[0]
        else:
            raise FileNotFoundError(f"Expected one VARMA dataset artifact under {dataset_dir}, got {candidates}.")
    ds, _artifact = load_windows_dataset(dataset_path, instantiate_functionals=False)
    if tuple(np.asarray(ds.X).shape[:3]) != (int(spec.N), int(spec.w), int(spec.d)):
        raise ValueError(
            f"Existing VARMA artifact {dataset_path} has X shape {np.asarray(ds.X).shape}, "
            f"expected {(int(spec.N), int(spec.w), int(spec.d))}."
        )
    train_idx, test_idx = make_deterministic_split(spec.N, spec.N_train, spec.N_test, seed=spec.split_seed)
    np.savez_compressed(dataset_dir / "split.npz", train_idx=train_idx, test_idx=test_idx)
    (dataset_dir / "latest_dataset.txt").write_text(dataset_path.name + "\n", encoding="utf-8")

    meta_path = dataset_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.update(
        {
            "dataset_id": spec.dataset_id,
            "task_names": list(TASK_NAMES_E2_THREE),
            "N": int(spec.N),
            "n_train": int(spec.N_train),
            "n_test": int(spec.N_test),
            "w": int(spec.w),
            "d": int(spec.d),
            "s": int(spec.s),
            "split_source": spec.split_source,
            "split_seed": int(spec.split_seed),
            "description": "Generated VARMA rebuttal ablation dataset.",
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def make_deterministic_split(N: int, N_train: int, N_test: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    N = int(N)
    N_train = int(N_train)
    N_test = int(N_test)
    if N_train <= 0 or N_test <= 0 or N_train + N_test > N:
        raise ValueError(f"Invalid split sizes: N={N}, N_train={N_train}, N_test={N_test}.")
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(N)
    train_idx = np.sort(idx[:N_train])
    test_idx = np.sort(idx[N_train : N_train + N_test])
    if np.intersect1d(train_idx, test_idx).size:
        raise RuntimeError("Generated train/test split overlaps.")
    return train_idx, test_idx


def run_varma_ablation_classical(
    *,
    dataset_dir: str | Path,
    out_root: str | Path,
    ablation: str,
    methods: Sequence[str] = VARMA_ABLATION_CLASSICAL_METHODS,
    feature_dim: int = VARMA_BASELINE_FEATURE_DIM,
    method_seed: int = 0,
    backend: str = "cupy",
    device: int | None = 0,
) -> list[Path]:
    run_dirs: list[Path] = []
    for method in methods:
        if method not in CLASSICAL_METHODS:
            raise ValueError(f"Unknown classical method={method!r}.")
        run_dirs.append(
            run_real_classical_baseline(
                str(method),
                dataset_path=dataset_dir,
                out_root=Path(out_root) / ablation,
                method_seed=method_seed,
                feature_dim=feature_dim,
                backend=backend,
                device=device,
            )
        )
    return run_dirs


def run_varma_ablation_quark(
    *,
    dataset_dir: str | Path,
    out_root: str | Path,
    ablation: str,
    regime: VarmaQuarkRegime,
    method_seed: int = 0,
    backend: str = "cupy",
    device: int | None = 0,
    num_workers: int = 1,
    retune_readout: bool = True,
    tune_max_inner: int | None = 1000,
    tune_max_val: int | None = 300,
) -> list[Path]:
    run_dir = run_real_quark(
        dataset_path=dataset_dir,
        out_root=Path(out_root) / ablation,
        method_seed=method_seed,
        runner=regime.runner,
        num_workers=num_workers,
        experiment_name=f"{ablation}_{regime.name}",
        backend=backend,
        device=device,
        method_suffix=regime.name,
        model_overrides=regime.overrides,
    )
    out = [run_dir]
    if retune_readout:
        method = f"quark_{regime.runner}_{regime.name}_kernel_readout_retune"
        out.append(
            run_cached_quark_readout_retune(
                dataset_path=dataset_dir,
                source_run=run_dir,
                out_root=Path(out_root) / ablation,
                method_seed=method_seed,
                val_ratio=0.2,
                split_strategy="random",
                lambda_grid=READOUT_RETUNE_LAMBDA_GRID,
                xi_grid=READOUT_RETUNE_XI_GRID,
                nu_grid=np.asarray([0.5, 1.5, 2.5], dtype=float),
                tune_max_inner=tune_max_inner,
                tune_max_val=tune_max_val,
                method=method,
                backend=backend,
                device=device,
            )
        )
    return out


def aggregate_varma_ablation_results(
    out_root: str | Path = DEFAULT_VARMA_ABLATION_RESULTS_ROOT,
) -> Path:
    out_root = Path(out_root)
    aggregate_dir = out_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    rows = read_varma_ablation_metric_rows(out_root)
    write_metrics_csv(aggregate_dir / "ablation_long.csv", rows)
    wide_rows = build_varma_ablation_wide_rows(rows, out_root=out_root, metric="mse")
    write_dict_csv(aggregate_dir / "ablation_wide.csv", wide_rows)

    architecture_rows = [row for row in wide_rows if row.get("ablation") == "architecture"]
    scaling_rows = [row for row in wide_rows if row.get("ablation") == "varma_wd_scaling"]
    shot_rows = [row for row in wide_rows if row.get("ablation") == "finite_shots"]

    (aggregate_dir / "architecture_ablation_table.md").write_text(
        build_varma_markdown_table(architecture_rows, title="Architecture Ablation"),
        encoding="utf-8",
    )
    (aggregate_dir / "varma_wd_scaling_table.md").write_text(
        build_varma_markdown_table(scaling_rows, title="VARMA w,d Scaling"),
        encoding="utf-8",
    )
    (aggregate_dir / "finite_shot_ablation_table.md").write_text(
        build_varma_markdown_table(shot_rows, title="Finite-Shot Ablation"),
        encoding="utf-8",
    )
    (aggregate_dir / "seed_errorbar_table.md").write_text(
        build_seed_errorbar_table(wide_rows),
        encoding="utf-8",
    )
    write_varma_scaling_heatmap(
        aggregate_dir / "varma_wd_scaling_volterra_heatmap.pdf",
        scaling_rows,
    )
    write_varma_response_drafts(aggregate_dir)
    return aggregate_dir


def read_varma_ablation_metric_rows(out_root: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for metrics_path in sorted(Path(out_root).glob("**/metrics.csv")):
        if "aggregate" in metrics_path.parts:
            continue
        with metrics_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({col: row.get(col, "") for col in METRICS_COLUMNS})
    return rows


def build_varma_ablation_wide_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    out_root: str | Path,
    metric: str = "mse",
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    out_root = Path(out_root)
    for row in rows:
        if row.get("metric") != metric:
            continue
        artifact_dir = Path(str(row.get("artifact_dir", "")))
        ablation = infer_ablation_name(artifact_dir, out_root)
        dataset = str(row.get("dataset", ""))
        shape = parse_varma_shape(dataset)
        key = (
            ablation,
            dataset,
            str(row.get("task", "")),
            str(row.get("method", "")),
            str(row.get("method_seed", "")),
        )
        out = grouped.setdefault(
            key,
            {
                "ablation": ablation,
                "dataset": dataset,
                "w": shape.get("w", ""),
                "d": shape.get("d", ""),
                "task": row.get("task", ""),
                "method": row.get("method", ""),
                "method_seed": row.get("method_seed", ""),
                "n_train": row.get("n_train", ""),
                "n_test": row.get("n_test", ""),
                "feature_dim": row.get("feature_dim", ""),
                "raw_dim": row.get("raw_dim", ""),
                "artifact_dir": row.get("artifact_dir", ""),
            },
        )
        out[f"{row.get('split')}_{metric}"] = row.get("value", "")
    return sorted(
        grouped.values(),
        key=lambda row: (
            str(row.get("ablation", "")),
            int(row.get("w") or 0),
            int(row.get("d") or 0),
            str(row.get("task", "")),
            str(row.get("method", "")),
        ),
    )


def infer_ablation_name(artifact_dir: Path, out_root: Path) -> str:
    try:
        rel = artifact_dir.resolve().relative_to(out_root.resolve())
    except Exception:
        return ""
    return rel.parts[0] if rel.parts else ""


def parse_varma_shape(dataset_id: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in ("N", "w", "d", "s"):
        match = re.search(rf"(?:^|__){key}=([0-9]+)", str(dataset_id))
        if match:
            out[key] = int(match.group(1))
    return out


def build_varma_markdown_table(rows: Sequence[Mapping[str, Any]], *, title: str) -> str:
    if not rows:
        return f"# {title}\n\n_No rows available._\n"
    fields = [
        "w",
        "d",
        "task",
        "method",
        "test_mse",
        "train_mse",
        "feature_dim",
        "raw_dim",
        "n_train",
        "n_test",
    ]
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("|" + "|".join("---" for _ in fields) + "|")
    for row in rows:
        lines.append("| " + " | ".join(format_varma_cell(row.get(field, "")) for field in fields) + " |")
    lines.append("")
    return "\n".join(lines)


def build_seed_errorbar_table(rows: Sequence[Mapping[str, Any]]) -> str:
    by_key: dict[tuple[str, str, str, str], list[float]] = {}
    for row in rows:
        try:
            value = float(row.get("test_mse", ""))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            key = (
                str(row.get("ablation", "")),
                str(row.get("dataset", "")),
                str(row.get("task", "")),
                str(row.get("method", "")),
            )
            by_key.setdefault(key, []).append(value)
    summary = []
    for (ablation, dataset, task, method), values in sorted(by_key.items()):
        if len(values) < 2:
            continue
        arr = np.asarray(values, dtype=float)
        summary.append(
            {
                "ablation": ablation,
                "dataset": dataset,
                "task": task,
                "method": method,
                "n_seeds": int(arr.size),
                "mean_test_mse": float(arr.mean()),
                "std_test_mse": float(arr.std(ddof=1)),
            }
        )
    if not summary:
        return "# Seed Error Bars\n\n_No repeated-seed rows available yet._\n"
    lines = ["# Seed Error Bars", ""]
    keys = list(summary[0].keys())
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("|" + "|".join("---" for _ in keys) + "|")
    for row in summary:
        lines.append("| " + " | ".join(format_varma_cell(row.get(key, "")) for key in keys) + " |")
    lines.append("")
    return "\n".join(lines)


def write_varma_scaling_heatmap(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        path.with_suffix(".plot_error.txt").write_text(str(exc), encoding="utf-8")
        return

    candidates = [
        row
        for row in rows
        if str(row.get("task", "")) == "volterra" and "quark_" in str(row.get("method", ""))
    ]
    if not candidates:
        path.with_suffix(".plot_skipped.txt").write_text("No Volterra QuaRK scaling rows available.", encoding="utf-8")
        return
    best: dict[tuple[int, int], float] = {}
    for row in candidates:
        try:
            key = (int(row.get("w", 0)), int(row.get("d", 0)))
            value = float(row.get("test_mse", ""))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            best[key] = min(value, best.get(key, float("inf")))
    if not best:
        path.with_suffix(".plot_skipped.txt").write_text("No finite Volterra QuaRK scaling rows available.", encoding="utf-8")
        return
    w_vals = sorted({key[0] for key in best})
    d_vals = sorted({key[1] for key in best})
    grid = np.full((len(w_vals), len(d_vals)), np.nan, dtype=float)
    for i, w in enumerate(w_vals):
        for j, d in enumerate(d_vals):
            grid[i, j] = best.get((w, d), np.nan)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    im = ax.imshow(grid, aspect="auto")
    ax.set_xticks(np.arange(len(d_vals)), labels=[str(d) for d in d_vals])
    ax.set_yticks(np.arange(len(w_vals)), labels=[str(w) for w in w_vals])
    ax.set_xlabel("d")
    ax.set_ylabel("w")
    ax.set_title("Best QuaRK Volterra test MSE")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_varma_response_drafts(aggregate_dir: Path) -> None:
    architecture = (aggregate_dir / "architecture_ablation_table.md").read_text(encoding="utf-8")
    scaling = (aggregate_dir / "varma_wd_scaling_table.md").read_text(encoding="utf-8")
    shots = (aggregate_dir / "finite_shot_ablation_table.md").read_text(encoding="utf-8")
    for name, lead in {
        "ablation_response_MSDS.md": "This ablation separates architecture sensitivity from VARMA w,d scaling at fixed N_train=5000.",
        "ablation_response_kc2J.md": "This ablation directly tests JL/no-JL, reservoir size, spatial reuse, and high-dimensional VARMA scaling.",
        "ablation_response_vKFL.md": "This ablation reports finite-shot behavior and systematic design sensitivity on the same VARMA family.",
    }.items():
        (aggregate_dir / name).write_text(
            f"# {name.removesuffix('.md')}\n\n{lead}\n\n{architecture}\n\n{scaling}\n\n{shots}\n",
            encoding="utf-8",
        )


def format_varma_cell(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}" if np.isfinite(float(value)) else ""
    text = str(value)
    try:
        val = float(text)
    except (TypeError, ValueError):
        return text
    return f"{val:.4g}" if np.isfinite(val) else ""


def build_varma_dry_run_actions(
    *,
    stages: set[str],
    specs: Sequence[VarmaDatasetSpec],
    methods: Sequence[str],
    out_root: str | Path,
    backend: str,
    retune_readout: bool,
) -> list[str]:
    actions: list[str] = []
    architecture = architecture_dataset_spec(
        N_train=specs[0].N_train if specs else 5000,
        N_test=specs[0].N_test if specs else 1000,
        s=specs[0].s if specs else 100,
        split_seed=specs[0].split_seed if specs else 0,
    )
    grid = [spec for spec in specs if not (spec.w == architecture.w and spec.d == architecture.d)]
    if "prepare" in stages:
        actions.append(
            f"prepare architecture_shape=w25,d3 and varma_grid_points={len(specs)} "
            f"N_train={architecture.N_train} N_test={architecture.N_test}"
        )
    if "classical" in stages:
        actions.append(
            f"run classical methods={','.join(methods)} feature_dim={VARMA_BASELINE_FEATURE_DIM} "
            f"architecture_anchor=1 varma_grid_points={len(specs)} backend={backend}"
        )
    if "quark_architecture" in stages:
        actions.append(f"run architecture regimes={len(architecture_regimes())} fixed_shape=w25,d3 retune={retune_readout}")
    if "quark_varma_grid" in stages:
        actions.append(
            f"run fixed QuaRK n=5,R=3,k=2,lam0=0.1 varma_grid_points={len(specs)} "
            f"retune={retune_readout}"
        )
    if "shots" in stages:
        actions.append(f"run finite-shot regimes shots={','.join(map(str, VARMA_SHOT_GRID))} fixed_shape=w25,d3")
    if "aggregate" in stages:
        actions.append(f"aggregate outputs out_root={Path(out_root)}")
    actions.append("grid=" + ",".join(f"w{spec.w}_d{spec.d}" for spec in specs))
    if grid:
        actions.append("non_anchor_grid=" + ",".join(f"w{spec.w}_d{spec.d}" for spec in grid))
    return actions


def expand_varma_ablation_stages(stages: Sequence[str]) -> set[str]:
    if "all" in stages:
        return {"prepare", "classical", "quark_architecture", "quark_varma_grid", "shots", "aggregate"}
    if "quark" in stages:
        out = set(stages)
        out.remove("quark")
        out.update({"quark_architecture", "quark_varma_grid"})
        return out
    return {str(stage) for stage in stages}
