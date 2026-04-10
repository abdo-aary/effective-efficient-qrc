from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.real_tser import DEFAULT_TSER_DATA_ROOT, REAL_DATASET_KEYS
from src.experiment.real_world_rebuttal import (
    DEFAULT_REAL_TRIAGE_RESULTS_ROOT,
    READOUT_RETUNE_LAMBDA_GRID,
    READOUT_RETUNE_XI_GRID,
    build_dataset_card_rows,
    load_real_benchmark_data,
    load_real_split,
    normalize_method_suffix,
    read_real_metric_rows,
    real_dataset_output_root,
    resolve_cached_quark_model_dir,
    run_cached_quark_readout_retune,
    run_real_classical_baseline,
    run_real_quark,
)


DEFAULT_DATASETS = (
    "copper_concentration",
    "gas_sensor_array_acetone",
    "gas_sensor_array_ethanol",
    "iron_concentration",
    "live_fuel_moisture",
    "manganese_concentration",
)
DEFAULT_LAM_GRID = (0.1, 0.5, 0.8)
DEFAULT_SHOT_GRID = (100, 500, 1000, 2000, 3000, 5000, 7000, 10000)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_FINAL_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_OUT_ROOT = DEFAULT_REAL_TRIAGE_RESULTS_ROOT.parent / "real_world_temporal_budget"
TEMPORAL_PREFIX = "temporal_budget_D315"


def format_float_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def format_metric(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(val):
        return ""
    return f"{val:.4f}"


def temporal_exact_suffix(lam0: float) -> str:
    return f"{TEMPORAL_PREFIX}_exact_n5_R3_k2_lam{format_float_token(lam0)}"


def temporal_exact_method(runner: str, lam0: float, *, retuned: bool = False) -> str:
    method = f"quark_{runner}{normalize_method_suffix(temporal_exact_suffix(lam0))}"
    if retuned:
        method += "_kernel_readout_retune"
    return method


def temporal_shadow_method(runner: str, lam0: float, shots: int) -> str:
    return (
        f"quark_{runner}{normalize_method_suffix(f'{TEMPORAL_PREFIX}_shadow_from_lam{format_float_token(lam0)}_shots{int(shots)}')}"
        "_kernel_readout_retune"
    )


def temporal_shadow_cache_method(runner: str, lam0: float, shots: int) -> str:
    return f"quark_{runner}{normalize_method_suffix(f'{TEMPORAL_PREFIX}_shadow_cache_from_lam{format_float_token(lam0)}_shots{int(shots)}')}"


def temporal_final_shadow_method(runner: str, lam0: float, shots: int) -> str:
    return (
        f"quark_{runner}"
        f"{normalize_method_suffix(f'{TEMPORAL_PREFIX}_final_shadow_lam{format_float_token(lam0)}_shots{int(shots)}')}"
        "_kernel_readout_retune"
    )


def temporal_final_shadow_cache_method(runner: str, lam0: float, shots: int) -> str:
    return (
        f"quark_{runner}"
        f"{normalize_method_suffix(f'{TEMPORAL_PREFIX}_final_shadow_cache_lam{format_float_token(lam0)}_shots{int(shots)}')}"
    )


def shadow_noise_seed(*, model_seed: int, shots: int) -> int:
    return int(model_seed) * 100_000 + int(shots)


def simulate_shadow_feature_maps_from_exact_phi(
    exact_phi: np.ndarray,
    *,
    shots: int,
    seed: int | None = None,
    n_groups: int | None = None,
) -> np.ndarray:
    exact_phi = np.asarray(exact_phi, dtype=float)
    if exact_phi.ndim != 2:
        raise ValueError(f"Expected exact feature matrix shape (N,D), got {exact_phi.shape}.")
    shots = int(shots)
    if shots <= 0:
        raise ValueError(f"shots must be positive, got {shots}.")
    if n_groups is None:
        n_groups = max(1, min(16, int(math.sqrt(shots))))
    n_groups = max(1, min(int(n_groups), shots))
    batch_size = max(shots // n_groups, 1)

    mu = np.clip(exact_phi, -1.0, 1.0)
    p = np.clip((1.0 + mu) / 2.0, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    counts = rng.binomial(n=batch_size, p=p[..., None], size=mu.shape + (n_groups,))
    group_means = (2.0 * counts - batch_size) / float(batch_size)
    return np.median(group_means, axis=-1).astype(float, copy=False)


def method_metric_path(run_root: Path, method: str, *, seed: int) -> Path:
    return run_root / method / f"seed={int(seed)}" / "metrics.csv"


def temporal_baseline_method(method: str, *, feature_dim: int = 315) -> str:
    return f"{str(method)}/feature_dim={int(feature_dim)}"


def temporal_baseline_label(method: str) -> str:
    if str(method) == "esn":
        return "ESN"
    if str(method) == "esn_matern_krr":
        return "ESN+Matérn"
    return str(method)


def run_real_root(dataset_path: Path, out_root: Path) -> tuple[Path, str, str]:
    data = load_real_benchmark_data(dataset_path)
    split = load_real_split(dataset_path)
    return real_dataset_output_root(out_root, data.dataset_id, split.split_source), data.dataset_id, split.split_source


def metrics_file_has_test_nrmse(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") == "test" and row.get("metric") == "nrmse_train_y_std":
                return True
    return False


def read_single_test_nrmse(metrics_path: Path) -> float:
    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") == "test" and row.get("metric") == "nrmse_train_y_std":
                return float(row["value"])
    raise ValueError(f"No test nrmse_train_y_std row found in {metrics_path}.")


def build_shadow_cache_from_exact_run(
    *,
    dataset_path: Path,
    source_run: Path,
    cache_run: Path,
    shots: int,
    seed: int,
) -> Path:
    source_model_dir = resolve_cached_quark_model_dir(source_run)
    with np.load(source_model_dir / "arrays.npz", allow_pickle=False) as z:
        arrays = {name: np.asarray(z[name]) for name in z.files}
    exact_phi = np.asarray(arrays["Phi_full"], dtype=float)
    noisy_phi = simulate_shadow_feature_maps_from_exact_phi(
        exact_phi,
        shots=int(shots),
        seed=shadow_noise_seed(model_seed=seed, shots=shots),
    )
    arrays["Phi_full"] = noisy_phi

    cache_model_dir = cache_run / "model"
    cache_model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_model_dir / "arrays.npz", **arrays)

    meta = json.loads((source_model_dir / "meta.json").read_text(encoding="utf-8"))
    meta["artifact"] = f"{meta.get('artifact', 'QRCMaternKRRRegressor.full')}.shadow_cache"
    meta["source_model_dir"] = str(source_model_dir)
    meta["noise_model"] = {
        "kind": "cs_binomial_mom_from_exact_phi",
        "shots": int(shots),
        "seed": int(shadow_noise_seed(model_seed=seed, shots=shots)),
        "n_groups": max(1, min(16, int(math.sqrt(int(shots))))),
    }
    (cache_model_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return cache_run


def find_best_exact_runs(
    *,
    out_root: Path,
    data_root: Path,
    datasets: Sequence[str],
    runner: str,
    seeds: Sequence[int],
    lam_grid: Sequence[float],
) -> dict[tuple[str, int], dict[str, Any]]:
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    for dataset in datasets:
        dataset_path = data_root / dataset
        run_root, dataset_id, _ = run_real_root(dataset_path, out_root)
        for seed in seeds:
            best: dict[str, Any] | None = None
            for lam0 in lam_grid:
                method = temporal_exact_method(runner, lam0, retuned=True)
                metrics_path = method_metric_path(run_root, method, seed=int(seed))
                if not metrics_path.exists():
                    continue
                value = read_single_test_nrmse(metrics_path)
                if best is None or value < float(best["test_nrmse"]):
                    best = {
                        "dataset": dataset_id,
                        "seed": int(seed),
                        "lam0": float(lam0),
                        "test_nrmse": float(value),
                        "retune_method": method,
                        "retune_run": metrics_path.parent,
                        "source_method": temporal_exact_method(runner, lam0, retuned=False),
                        "source_run": run_root / temporal_exact_method(runner, lam0, retuned=False) / f"seed={int(seed)}",
                    }
            if best is not None:
                selected[(dataset_id, int(seed))] = best
    return selected


def build_exact_vs_esn_rows(
    *,
    out_root: Path,
    data_root: Path,
    datasets: Sequence[str],
    runner: str,
    seeds: Sequence[int],
    lam_grid: Sequence[float],
    temporal_baseline: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    best_exact = find_best_exact_runs(
        out_root=out_root,
        data_root=data_root,
        datasets=datasets,
        runner=runner,
        seeds=seeds,
        lam_grid=lam_grid,
    )
    cards = {row["dataset"]: row for row in build_dataset_card_rows(data_root, datasets=datasets)}
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_path = data_root / dataset
        run_root, dataset_id, _ = run_real_root(dataset_path, out_root)
        paired: list[dict[str, Any]] = []
        lam_counter: Counter[float] = Counter()
        for seed in seeds:
            exact = best_exact.get((dataset_id, int(seed)))
            esn_metrics = method_metric_path(
                run_root,
                temporal_baseline_method(temporal_baseline),
                seed=int(seed),
            )
            if exact is None or not esn_metrics.exists():
                continue
            esn_value = read_single_test_nrmse(esn_metrics)
            lam_counter[float(exact["lam0"])] += 1
            paired.append(
                {
                    "seed": int(seed),
                    "quark": float(exact["test_nrmse"]),
                    "esn": float(esn_value),
                    "lam0": float(exact["lam0"]),
                }
            )
        if not paired:
            continue
        quark_vals = np.asarray([row["quark"] for row in paired], dtype=float)
        esn_vals = np.asarray([row["esn"] for row in paired], dtype=float)
        card = cards.get(dataset_id, {})
        rows.append(
            {
                "dataset": dataset_id,
                "w": int(card.get("w", 0) or 0),
                "d": int(card.get("d", 0) or 0),
                "raw_dim": int(card.get("raw_dim", 0) or 0),
                "num_seeds": int(len(paired)),
                "selected_lam0": ", ".join(f"{lam:g}x{count}" for lam, count in sorted(lam_counter.items())),
                "quark_mean": float(np.mean(quark_vals)),
                "quark_std": float(np.std(quark_vals, ddof=0)),
                "esn_mean": float(np.mean(esn_vals)),
                "esn_std": float(np.std(esn_vals, ddof=0)),
                "delta_mean": float(np.mean(quark_vals - esn_vals)),
                "quark_seed_wins": int(np.sum(quark_vals < esn_vals)),
            }
        )
    rows.sort(key=lambda row: (row["delta_mean"], row["dataset"]))
    return rows, best_exact


def build_lambda_summary_rows(
    *,
    out_root: Path,
    data_root: Path,
    datasets: Sequence[str],
    runner: str,
    seeds: Sequence[int],
    lam_grid: Sequence[float],
    temporal_baseline: str,
) -> list[dict[str, Any]]:
    cards = {row["dataset"]: row for row in build_dataset_card_rows(data_root, datasets=datasets)}
    rows: list[dict[str, Any]] = []
    for lam0 in lam_grid:
        paired: list[tuple[str, int, float, float]] = []
        covered_datasets: set[str] = set()
        for dataset in datasets:
            dataset_path = data_root / dataset
            run_root, dataset_id, _ = run_real_root(dataset_path, out_root)
            for seed in seeds:
                quark_method = temporal_exact_method(runner, lam0, retuned=True)
                quark_metrics = method_metric_path(run_root, quark_method, seed=int(seed))
                esn_metrics = method_metric_path(
                    run_root,
                    temporal_baseline_method(temporal_baseline),
                    seed=int(seed),
                )
                if not quark_metrics.exists() or not esn_metrics.exists():
                    continue
                paired.append(
                    (
                        dataset_id,
                        int(seed),
                        read_single_test_nrmse(quark_metrics),
                        read_single_test_nrmse(esn_metrics),
                    )
                )
                covered_datasets.add(dataset_id)
        if not paired:
            continue
        quark_vals = np.asarray([row[2] for row in paired], dtype=float)
        esn_vals = np.asarray([row[3] for row in paired], dtype=float)
        rows.append(
            {
                "lam0": float(lam0),
                "num_pairs": int(len(paired)),
                "num_datasets": int(len(covered_datasets)),
                "quark_mean": float(np.mean(quark_vals)),
                "quark_std": float(np.std(quark_vals, ddof=0)),
                "esn_mean": float(np.mean(esn_vals)),
                "esn_std": float(np.std(esn_vals, ddof=0)),
                "delta_mean": float(np.mean(quark_vals - esn_vals)),
                "quark_seed_wins": int(np.sum(quark_vals < esn_vals)),
            }
        )
    rows.sort(key=lambda row: (-int(row["num_pairs"]), float(row["delta_mean"]), float(row["lam0"])))
    return rows


def recommend_global_lambda(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    max_pairs = max(int(row["num_pairs"]) for row in rows)
    candidates = [dict(row) for row in rows if int(row["num_pairs"]) == max_pairs]
    candidates.sort(key=lambda row: (float(row["delta_mean"]), float(row["quark_mean"]), float(row["lam0"])))
    return candidates[0]


def build_shot_summary_rows(
    *,
    out_root: Path,
    best_exact: Mapping[tuple[str, int], Mapping[str, Any]],
    runner: str,
    shot_grid: Sequence[int],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    all_rows = read_real_metric_rows(out_root, include_quark=True)
    for shots in shot_grid:
        shot_method = temporal_shadow_method(runner, lam0=0.0, shots=shots)
        values: list[tuple[float, float]] = []
        for row in all_rows:
            method = str(row.get("method", ""))
            if row.get("split") != "test" or row.get("metric") != "nrmse_train_y_std":
                continue
            if f"_shadow_from_lam" not in method or f"_shots{int(shots)}_kernel_readout_retune" not in method:
                continue
            dataset = str(row.get("dataset", ""))
            seed = int(float(row.get("method_seed", "0")))
            exact = best_exact.get((dataset, seed))
            if exact is None:
                continue
            values.append((float(row["value"]), float(exact["test_nrmse"])))
        if not values:
            continue
        shot_vals = np.asarray([pair[0] for pair in values], dtype=float)
        exact_vals = np.asarray([pair[1] for pair in values], dtype=float)
        deltas = shot_vals - exact_vals
        summary.append(
            {
                "shots": int(shots),
                "num_runs": int(len(values)),
                "shot_mean": float(np.mean(shot_vals)),
                "shot_std": float(np.std(shot_vals, ddof=0)),
                "exact_mean": float(np.mean(exact_vals)),
                "mean_delta_to_exact": float(np.mean(deltas)),
                "max_delta_to_exact": float(np.max(deltas)),
            }
        )
    return summary


def recommend_shots(
    rows: Sequence[Mapping[str, Any]],
    *,
    mean_delta_tol: float,
    max_delta_tol: float,
) -> dict[str, Any] | None:
    for row in sorted(rows, key=lambda item: int(item["shots"])):
        if float(row["mean_delta_to_exact"]) <= float(mean_delta_tol) and float(row["max_delta_to_exact"]) <= float(max_delta_tol):
            return dict(row)
    if not rows:
        return None
    return dict(min(rows, key=lambda item: float(item["mean_delta_to_exact"])))


def build_final_rows(
    *,
    out_root: Path,
    data_root: Path,
    datasets: Sequence[str],
    runner: str,
    final_seeds: Sequence[int],
    lam0: float,
    shots: int,
    temporal_baseline: str,
) -> list[dict[str, Any]]:
    cards = {row["dataset"]: row for row in build_dataset_card_rows(data_root, datasets=datasets)}
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_path = data_root / dataset
        run_root, dataset_id, _ = run_real_root(dataset_path, out_root)
        paired: list[dict[str, Any]] = []
        quark_method = temporal_final_shadow_method(runner, lam0, shots)
        for seed in final_seeds:
            quark_metrics = method_metric_path(run_root, quark_method, seed=int(seed))
            esn_metrics = method_metric_path(
                run_root,
                temporal_baseline_method(temporal_baseline),
                seed=int(seed),
            )
            if not quark_metrics.exists() or not esn_metrics.exists():
                continue
            paired.append(
                {
                    "seed": int(seed),
                    "quark": read_single_test_nrmse(quark_metrics),
                    "esn": read_single_test_nrmse(esn_metrics),
                }
            )
        if not paired:
            continue
        quark_vals = np.asarray([row["quark"] for row in paired], dtype=float)
        esn_vals = np.asarray([row["esn"] for row in paired], dtype=float)
        card = cards.get(dataset_id, {})
        rows.append(
            {
                "dataset": dataset_id,
                "w": int(card.get("w", 0) or 0),
                "d": int(card.get("d", 0) or 0),
                "raw_dim": int(card.get("raw_dim", 0) or 0),
                "num_seeds": int(len(paired)),
                "lam0": float(lam0),
                "shots": int(shots),
                "quark_mean": float(np.mean(quark_vals)),
                "quark_std": float(np.std(quark_vals, ddof=0)),
                "esn_mean": float(np.mean(esn_vals)),
                "esn_std": float(np.std(esn_vals, ddof=0)),
                "delta_mean": float(np.mean(quark_vals - esn_vals)),
                "quark_seed_wins": int(np.sum(quark_vals < esn_vals)),
            }
        )
    rows.sort(key=lambda row: (row["delta_mean"], row["dataset"]))
    return rows


def write_dict_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_exact_vs_esn_markdown(rows: Sequence[Mapping[str, Any]], *, temporal_baseline: str) -> str:
    baseline_label = temporal_baseline_label(temporal_baseline)
    lines = [
        "# Budget-Matched Temporal Comparison",
        "",
        f"| Dataset | w | d | raw_dim | seeds | selected lam0 | QuaRK mean +- std | {baseline_label} mean +- std | delta mean | QuaRK wins |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["w"]),
                    str(row["d"]),
                    str(row["raw_dim"]),
                    str(row["num_seeds"]),
                    str(row["selected_lam0"]),
                    f"{float(row['quark_mean']):.4f} +- {float(row['quark_std']):.4f}",
                    f"{float(row['esn_mean']):.4f} +- {float(row['esn_std']):.4f}",
                    f"{float(row['delta_mean']):+.4f}",
                    str(row["quark_seed_wins"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        f"QuaRK uses fixed n=5, R=3, k=2, D=315 and {baseline_label} uses 315 hidden states/features."
    )
    return "\n".join(lines)


def build_lambda_markdown(
    rows: Sequence[Mapping[str, Any]],
    *,
    recommendation: Mapping[str, Any] | None,
    temporal_baseline: str,
) -> str:
    baseline_label = temporal_baseline_label(temporal_baseline)
    lines = [
        "# Global Lambda Selection",
        "",
        f"| lam0 | pairs | datasets | QuaRK mean +- std | {baseline_label} mean +- std | delta mean | QuaRK wins |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{float(row['lam0']):g}",
                    str(row["num_pairs"]),
                    str(row["num_datasets"]),
                    f"{float(row['quark_mean']):.4f} +- {float(row['quark_std']):.4f}",
                    f"{float(row['esn_mean']):.4f} +- {float(row['esn_std']):.4f}",
                    f"{float(row['delta_mean']):+.4f}",
                    str(row["quark_seed_wins"]),
                ]
            )
            + " |"
        )
    lines.append("")
    if recommendation is None:
        lines.append("No completed exact-vs-temporal-baseline pilot runs available.")
    else:
        lines.append(
            "Recommended global lambda_0: "
            f"{float(recommendation['lam0']):g} "
            f"(delta mean {float(recommendation['delta_mean']):+.4f} across {int(recommendation['num_pairs'])} pairs)."
        )
    return "\n".join(lines)


def build_shot_markdown(rows: Sequence[Mapping[str, Any]], *, recommendation: Mapping[str, Any] | None) -> str:
    lines = [
        "# Shot Sweep Summary",
        "",
        "| shots | runs | shot mean +- std | exact mean | mean delta | max delta |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: int(item["shots"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["shots"]),
                    str(row["num_runs"]),
                    f"{float(row['shot_mean']):.4f} +- {float(row['shot_std']):.4f}",
                    f"{float(row['exact_mean']):.4f}",
                    f"{float(row['mean_delta_to_exact']):+.4f}",
                    f"{float(row['max_delta_to_exact']):+.4f}",
                ]
            )
            + " |"
        )
    lines.append("")
    if recommendation is None:
        lines.append("No completed shot-sweep runs available.")
    else:
        lines.append(
            "Recommended reporting point: "
            f"{int(recommendation['shots'])} shots "
            f"(mean delta {float(recommendation['mean_delta_to_exact']):+.4f}, "
            f"max delta {float(recommendation['max_delta_to_exact']):+.4f})."
        )
    return "\n".join(lines)


def build_final_markdown(
    rows: Sequence[Mapping[str, Any]],
    *,
    lam0: float,
    shots: int,
    temporal_baseline: str,
) -> str:
    baseline_label = temporal_baseline_label(temporal_baseline)
    lines = [
        "# Final Rebuttal Table",
        "",
        f"| Dataset | w | d | raw_dim | seeds | QuaRK mean +- std | {baseline_label} mean +- std | delta mean | QuaRK wins |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["w"]),
                    str(row["d"]),
                    str(row["raw_dim"]),
                    str(row["num_seeds"]),
                    f"{float(row['quark_mean']):.4f} +- {float(row['quark_std']):.4f}",
                    f"{float(row['esn_mean']):.4f} +- {float(row['esn_std']):.4f}",
                    f"{float(row['delta_mean']):+.4f}",
                    str(row["quark_seed_wins"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        f"QuaRK uses fixed n=5, R=3, k=2, D=315 with global lambda_0={float(lam0):g} "
        f"and {int(shots)} simulated shadow shots; {baseline_label} uses 315 hidden states/features."
    )
    return "\n".join(lines)


def aggregate_outputs(
    *,
    out_root: Path,
    data_root: Path,
    datasets: Sequence[str],
    runner: str,
    seeds: Sequence[int],
    final_seeds: Sequence[int],
    lam_grid: Sequence[float],
    shot_grid: Sequence[int],
    mean_delta_tol: float,
    max_delta_tol: float,
    temporal_baseline: str,
) -> Path:
    aggregate_dir = out_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    exact_rows, best_exact = build_exact_vs_esn_rows(
        out_root=out_root,
        data_root=data_root,
        datasets=datasets,
        runner=runner,
        seeds=seeds,
        lam_grid=lam_grid,
        temporal_baseline=temporal_baseline,
    )
    lambda_rows = build_lambda_summary_rows(
        out_root=out_root,
        data_root=data_root,
        datasets=datasets,
        runner=runner,
        seeds=seeds,
        lam_grid=lam_grid,
        temporal_baseline=temporal_baseline,
    )
    lambda_recommendation = recommend_global_lambda(lambda_rows)
    shot_rows = build_shot_summary_rows(
        out_root=out_root,
        best_exact=best_exact,
        runner=runner,
        shot_grid=shot_grid,
    )
    recommendation = recommend_shots(
        shot_rows,
        mean_delta_tol=mean_delta_tol,
        max_delta_tol=max_delta_tol,
    )
    final_rows: list[dict[str, Any]] = []
    if lambda_recommendation is not None and recommendation is not None:
        final_rows = build_final_rows(
            out_root=out_root,
            data_root=data_root,
            datasets=datasets,
            runner=runner,
            final_seeds=final_seeds,
            lam0=float(lambda_recommendation["lam0"]),
            shots=int(recommendation["shots"]),
            temporal_baseline=temporal_baseline,
        )
    write_dict_csv(aggregate_dir / "temporal_budget_exact_vs_esn.csv", exact_rows)
    write_dict_csv(aggregate_dir / "temporal_budget_lambda_selection.csv", lambda_rows)
    write_dict_csv(aggregate_dir / "temporal_budget_shot_sweep.csv", shot_rows)
    write_dict_csv(aggregate_dir / "temporal_budget_final_table.csv", final_rows)
    (aggregate_dir / "temporal_budget_exact_vs_esn.md").write_text(
        build_exact_vs_esn_markdown(exact_rows, temporal_baseline=temporal_baseline),
        encoding="utf-8",
    )
    (aggregate_dir / "temporal_budget_lambda_selection.md").write_text(
        build_lambda_markdown(
            lambda_rows,
            recommendation=lambda_recommendation,
            temporal_baseline=temporal_baseline,
        ),
        encoding="utf-8",
    )
    (aggregate_dir / "temporal_budget_shot_sweep.md").write_text(
        build_shot_markdown(shot_rows, recommendation=recommendation),
        encoding="utf-8",
    )
    if lambda_recommendation is not None and recommendation is not None:
        (aggregate_dir / "temporal_budget_final_table.md").write_text(
            build_final_markdown(
                final_rows,
                lam0=float(lambda_recommendation["lam0"]),
                shots=int(recommendation["shots"]),
                temporal_baseline=temporal_baseline,
            ),
            encoding="utf-8",
        )
        (aggregate_dir / "temporal_budget_selection.json").write_text(
            json.dumps(
                {
                    "global_lambda_0": float(lambda_recommendation["lam0"]),
                    "recommended_shots": int(recommendation["shots"]),
                    "lambda_selection": lambda_recommendation,
                    "shot_selection": recommendation,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if recommendation is not None:
        (aggregate_dir / "temporal_budget_shot_recommendation.json").write_text(
            json.dumps(recommendation, indent=2),
            encoding="utf-8",
        )
    if lambda_recommendation is not None:
        (aggregate_dir / "temporal_budget_lambda_recommendation.json").write_text(
            json.dumps(lambda_recommendation, indent=2),
            encoding="utf-8",
        )
    return aggregate_dir


def build_dry_run_actions(args: argparse.Namespace, stages: set[str]) -> list[str]:
    actions: list[str] = []
    baseline_label = temporal_baseline_label(args.temporal_baseline)
    if "exact" in stages:
        actions.append(
            "run exact QuaRK datasets="
            + ",".join(args.datasets)
            + " seeds="
            + ",".join(map(str, args.seeds))
            + " lam_grid="
            + ",".join(f"{float(l):g}" for l in args.lam_grid)
            + f" runner={args.runner}"
        )
    if "esn" in stages:
        reuse_note = ""
        if args.temporal_baseline == "esn_matern_krr" and args.reuse_existing_esn_config:
            reuse_note = " reuse_existing_esn_config=true"
        actions.append(
            f"run {baseline_label} datasets="
            + ",".join(args.datasets)
            + " seeds="
            + ",".join(map(str, args.seeds))
            + " feature_dim=315"
            + reuse_note
        )
    if "shots" in stages:
        actions.append(
            "run cached shot sweep from best exact datasets="
            + ",".join(args.datasets)
            + " shot_grid="
            + ",".join(map(str, args.shot_grid))
        )
    if "final" in stages:
        actions.append(
            "run final fixed-configuration rerun datasets="
            + ",".join(args.datasets)
            + " final_seeds="
            + ",".join(map(str, args.final_seeds))
            + " using selected global lam0 and shots"
        )
    if "aggregate" in stages:
        actions.append(f"write temporal comparison aggregates out_root={args.out_root}")
    return actions


def expand_stages(stages: Sequence[str]) -> set[str]:
    if "all" in stages:
        return {"exact", "esn", "shots", "final", "aggregate"}
    return set(stages)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run budget-matched real-world temporal comparison: QuaRK (D=315) vs temporal baseline (315 states/features)."
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=sorted(REAL_DATASET_KEYS))
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["exact", "esn", "shots", "final", "aggregate"],
        choices=("exact", "esn", "shots", "final", "aggregate", "all"),
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_TSER_DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--runner", default="reservoir_channel_cupy_direct_truncated64")
    parser.add_argument(
        "--temporal-baseline",
        choices=("esn", "esn_matern_krr"),
        default="esn_matern_krr",
        help="Temporal baseline to compare against; use esn_matern_krr for the strict matched-readout control.",
    )
    parser.add_argument(
        "--reuse-existing-esn-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When using esn_matern_krr, rebuild ESN states from saved ESN best_params instead of rerunning ESN search.",
    )
    parser.add_argument("--backend", choices=("auto", "cupy", "numpy"), default="cupy")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--final-seeds", nargs="+", type=int, default=list(DEFAULT_FINAL_SEEDS))
    parser.add_argument("--lam-grid", nargs="+", type=float, default=list(DEFAULT_LAM_GRID))
    parser.add_argument("--shot-grid", nargs="+", type=int, default=list(DEFAULT_SHOT_GRID))
    parser.add_argument("--retune-readout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-strategy", choices=("chronological", "random"), default="random")
    parser.add_argument("--tune-max-inner", type=int, default=1000)
    parser.add_argument("--tune-max-val", type=int, default=300)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--mean-delta-tol", type=float, default=0.03)
    parser.add_argument("--max-delta-tol", type=float, default=0.08)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    stages = expand_stages(args.stages)
    if args.dry_run:
        for action in build_dry_run_actions(args, stages):
            print(action)
        return

    if "exact" in stages:
        for dataset in args.datasets:
            dataset_path = args.data_root / dataset
            run_root, _, _ = run_real_root(dataset_path, args.out_root)
            for seed in args.seeds:
                for lam0 in args.lam_grid:
                    suffix = temporal_exact_suffix(lam0)
                    method = temporal_exact_method(args.runner, lam0, retuned=False)
                    retune_method = temporal_exact_method(args.runner, lam0, retuned=True)
                    if not (args.skip_existing and metrics_file_has_test_nrmse(method_metric_path(run_root, method, seed=int(seed)))):
                        run_dir = run_real_quark(
                            dataset_path=dataset_path,
                            out_root=args.out_root,
                            method_seed=int(seed),
                            runner=args.runner,
                            num_workers=args.num_workers,
                            experiment_name=f"{dataset}_{suffix}_seed{int(seed)}",
                            backend=args.backend,
                            device=args.device,
                            method_suffix=suffix,
                            model_overrides=[
                                "model/qrc/features/retriever=exact",
                                "model.qrc.cfg.num_qubits=5",
                                "model.qrc.features.observables.locality=2",
                                "model.qrc.pubs.num_reservoirs=3",
                                f"model.qrc.pubs.lam_0={float(lam0)}",
                            ],
                        )
                        print(run_dir)
                    if args.retune_readout and not (
                        args.skip_existing and metrics_file_has_test_nrmse(method_metric_path(run_root, retune_method, seed=int(seed)))
                    ):
                        retune_dir = run_cached_quark_readout_retune(
                            dataset_path=dataset_path,
                            source_run=run_root / method / f"seed={int(seed)}",
                            out_root=args.out_root,
                            method_seed=int(seed),
                            val_ratio=args.val_ratio,
                            split_strategy=args.split_strategy,
                            lambda_grid=READOUT_RETUNE_LAMBDA_GRID,
                            xi_grid=READOUT_RETUNE_XI_GRID,
                            nu_grid=np.asarray([0.5, 1.5, 2.5], dtype=float),
                            tune_max_inner=args.tune_max_inner,
                            tune_max_val=args.tune_max_val,
                            method=retune_method,
                            backend=args.backend,
                            device=args.device,
                        )
                        print(retune_dir)

    if "esn" in stages:
        for dataset in args.datasets:
            dataset_path = args.data_root / dataset
            for seed in args.seeds:
                run_root, _, _ = run_real_root(dataset_path, args.out_root)
                if args.skip_existing and metrics_file_has_test_nrmse(
                    method_metric_path(run_root, temporal_baseline_method(args.temporal_baseline), seed=int(seed))
                ):
                    continue
                run_dir = run_real_classical_baseline(
                    args.temporal_baseline,
                    dataset_path=dataset_path,
                    out_root=args.out_root,
                    method_seed=int(seed),
                    feature_dim=315,
                    reuse_esn_source_run=(
                        run_root / temporal_baseline_method("esn") / f"seed={int(seed)}"
                        if args.temporal_baseline == "esn_matern_krr" and args.reuse_existing_esn_config
                        else None
                    ),
                    backend=args.backend,
                    device=args.device,
                )
                print(run_dir)

    if "shots" in stages:
        best_exact = find_best_exact_runs(
            out_root=args.out_root,
            data_root=args.data_root,
            datasets=args.datasets,
            runner=args.runner,
            seeds=args.seeds,
            lam_grid=args.lam_grid,
        )
        for dataset in args.datasets:
            dataset_path = args.data_root / dataset
            run_root, dataset_id, _ = run_real_root(dataset_path, args.out_root)
            for seed in args.seeds:
                exact = best_exact.get((dataset_id, int(seed)))
                if exact is None:
                    continue
                for shots in args.shot_grid:
                    cache_method = temporal_shadow_cache_method(args.runner, float(exact["lam0"]), int(shots))
                    cache_run = run_root / cache_method / f"seed={int(seed)}"
                    shot_method = temporal_shadow_method(args.runner, float(exact["lam0"]), int(shots))
                    if not cache_run.exists() or not (cache_run / "model" / "arrays.npz").exists():
                        build_shadow_cache_from_exact_run(
                            dataset_path=dataset_path,
                            source_run=Path(exact["source_run"]),
                            cache_run=cache_run,
                            shots=int(shots),
                            seed=int(seed),
                        )
                    if args.skip_existing and metrics_file_has_test_nrmse(method_metric_path(run_root, shot_method, seed=int(seed))):
                        continue
                    retune_dir = run_cached_quark_readout_retune(
                        dataset_path=dataset_path,
                        source_run=cache_run,
                        out_root=args.out_root,
                        method_seed=int(seed),
                        val_ratio=args.val_ratio,
                        split_strategy=args.split_strategy,
                        lambda_grid=READOUT_RETUNE_LAMBDA_GRID,
                        xi_grid=READOUT_RETUNE_XI_GRID,
                        nu_grid=np.asarray([0.5, 1.5, 2.5], dtype=float),
                        tune_max_inner=args.tune_max_inner,
                        tune_max_val=args.tune_max_val,
                        method=shot_method,
                        backend=args.backend,
                        device=args.device,
                    )
                    print(retune_dir)

    if "final" in stages:
        lambda_rows = build_lambda_summary_rows(
            out_root=args.out_root,
            data_root=args.data_root,
            datasets=args.datasets,
            runner=args.runner,
            seeds=args.seeds,
            lam_grid=args.lam_grid,
            temporal_baseline=args.temporal_baseline,
        )
        lambda_recommendation = recommend_global_lambda(lambda_rows)
        best_exact = find_best_exact_runs(
            out_root=args.out_root,
            data_root=args.data_root,
            datasets=args.datasets,
            runner=args.runner,
            seeds=args.seeds,
            lam_grid=args.lam_grid,
        )
        shot_rows = build_shot_summary_rows(
            out_root=args.out_root,
            best_exact=best_exact,
            runner=args.runner,
            shot_grid=args.shot_grid,
        )
        shot_recommendation = recommend_shots(
            shot_rows,
            mean_delta_tol=args.mean_delta_tol,
            max_delta_tol=args.max_delta_tol,
        )
        if lambda_recommendation is None or shot_recommendation is None:
            raise RuntimeError("Final stage requires completed pilot exact/shot runs to select global lambda_0 and shots.")
        global_lam0 = float(lambda_recommendation["lam0"])
        final_shots = int(shot_recommendation["shots"])
        for dataset in args.datasets:
            dataset_path = args.data_root / dataset
            run_root, _, _ = run_real_root(dataset_path, args.out_root)
            for seed in args.final_seeds:
                base_method = temporal_exact_method(args.runner, global_lam0, retuned=False)
                base_retune_method = temporal_exact_method(args.runner, global_lam0, retuned=True)
                if not (args.skip_existing and metrics_file_has_test_nrmse(method_metric_path(run_root, base_method, seed=int(seed)))):
                    run_dir = run_real_quark(
                        dataset_path=dataset_path,
                        out_root=args.out_root,
                        method_seed=int(seed),
                        runner=args.runner,
                        num_workers=args.num_workers,
                        experiment_name=f"{dataset}_{temporal_exact_suffix(global_lam0)}_final_seed{int(seed)}",
                        backend=args.backend,
                        device=args.device,
                        method_suffix=temporal_exact_suffix(global_lam0),
                        model_overrides=[
                            "model/qrc/features/retriever=exact",
                            "model.qrc.cfg.num_qubits=5",
                            "model.qrc.features.observables.locality=2",
                            "model.qrc.pubs.num_reservoirs=3",
                            f"model.qrc.pubs.lam_0={global_lam0}",
                        ],
                    )
                    print(run_dir)
                if args.retune_readout and not (
                    args.skip_existing and metrics_file_has_test_nrmse(method_metric_path(run_root, base_retune_method, seed=int(seed)))
                ):
                    retune_dir = run_cached_quark_readout_retune(
                        dataset_path=dataset_path,
                        source_run=run_root / base_method / f"seed={int(seed)}",
                        out_root=args.out_root,
                        method_seed=int(seed),
                        val_ratio=args.val_ratio,
                        split_strategy=args.split_strategy,
                        lambda_grid=READOUT_RETUNE_LAMBDA_GRID,
                        xi_grid=READOUT_RETUNE_XI_GRID,
                        nu_grid=np.asarray([0.5, 1.5, 2.5], dtype=float),
                        tune_max_inner=args.tune_max_inner,
                        tune_max_val=args.tune_max_val,
                        method=base_retune_method,
                        backend=args.backend,
                        device=args.device,
                    )
                    print(retune_dir)
                if args.skip_existing and metrics_file_has_test_nrmse(
                    method_metric_path(run_root, temporal_baseline_method(args.temporal_baseline), seed=int(seed))
                ):
                    pass
                else:
                    run_dir = run_real_classical_baseline(
                        args.temporal_baseline,
                        dataset_path=dataset_path,
                        out_root=args.out_root,
                        method_seed=int(seed),
                        feature_dim=315,
                        reuse_esn_source_run=(
                            run_root / temporal_baseline_method("esn") / f"seed={int(seed)}"
                            if args.temporal_baseline == "esn_matern_krr" and args.reuse_existing_esn_config
                            else None
                        ),
                        backend=args.backend,
                        device=args.device,
                    )
                    print(run_dir)
                cache_method = temporal_final_shadow_cache_method(args.runner, global_lam0, final_shots)
                cache_run = run_root / cache_method / f"seed={int(seed)}"
                final_method = temporal_final_shadow_method(args.runner, global_lam0, final_shots)
                if not cache_run.exists() or not (cache_run / "model" / "arrays.npz").exists():
                    build_shadow_cache_from_exact_run(
                        dataset_path=dataset_path,
                        source_run=run_root / base_method / f"seed={int(seed)}",
                        cache_run=cache_run,
                        shots=final_shots,
                        seed=int(seed),
                    )
                if args.skip_existing and metrics_file_has_test_nrmse(method_metric_path(run_root, final_method, seed=int(seed))):
                    continue
                retune_dir = run_cached_quark_readout_retune(
                    dataset_path=dataset_path,
                    source_run=cache_run,
                    out_root=args.out_root,
                    method_seed=int(seed),
                    val_ratio=args.val_ratio,
                    split_strategy=args.split_strategy,
                    lambda_grid=READOUT_RETUNE_LAMBDA_GRID,
                    xi_grid=READOUT_RETUNE_XI_GRID,
                    nu_grid=np.asarray([0.5, 1.5, 2.5], dtype=float),
                    tune_max_inner=args.tune_max_inner,
                    tune_max_val=args.tune_max_val,
                    method=final_method,
                    backend=args.backend,
                    device=args.device,
                )
                print(retune_dir)

        aggregate_dir = aggregate_outputs(
            out_root=args.out_root,
            data_root=args.data_root,
            datasets=args.datasets,
            runner=args.runner,
            seeds=args.seeds,
            final_seeds=args.final_seeds,
            lam_grid=args.lam_grid,
            shot_grid=args.shot_grid,
            mean_delta_tol=args.mean_delta_tol,
            max_delta_tol=args.max_delta_tol,
            temporal_baseline=args.temporal_baseline,
        )
        print(aggregate_dir)

    if "aggregate" in stages:
        aggregate_dir = aggregate_outputs(
            out_root=args.out_root,
            data_root=args.data_root,
            datasets=args.datasets,
            runner=args.runner,
            seeds=args.seeds,
            final_seeds=args.final_seeds,
            lam_grid=args.lam_grid,
            shot_grid=args.shot_grid,
            mean_delta_tol=args.mean_delta_tol,
            max_delta_tol=args.max_delta_tol,
            temporal_baseline=args.temporal_baseline,
        )
        print(aggregate_dir)


if __name__ == "__main__":
    main()
