"""Artifact-only figures and tables for Experiment E1."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.experiment.e1 import aggregate_e1


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _group_mean_std(
    rows: list[dict[str, Any]], *, x: Callable[[dict[str, Any]], float], y: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[float, list[float]] = {}
    for row in rows:
        grouped.setdefault(float(x(row)), []).append(float(row[y]))
    x_values = np.asarray(sorted(grouped), dtype=float)
    means = np.asarray([np.mean(grouped[value]) for value in x_values])
    stds = np.asarray([np.std(grouped[value], ddof=0) for value in x_values])
    return x_values, means, stds


def _write_latex_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{rrrrrrrr}",
        r"Root & Task & Test MSE & Test NRMSE & $\nu$ & $\xi$ & $\lambda_K$ & RKHS norm \\",
        r"\hline",
    ]
    for row in rows:
        task = str(row["task"]).replace("_", r"\_")
        lines.append(
            f"{row['root']} & {task} & {float(row['test_mse']):.6g} & "
            f"{float(row['test_nrmse']):.6g} & {float(row['nu']):g} & "
            f"{float(row['xi']):g} & {float(row['lambda_K']):.3g} & "
            f"{float(row['rkhs_norm']):.6g} \\\\"
        )
    lines.extend((r"\end{tabular}", ""))
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_e1(
    *, artifact_root: str | Path, scenario: str = "reference-varma-functional"
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    artifact_root = Path(artifact_root)
    aggregate_e1(artifact_root=artifact_root, scenario=scenario)
    method_root = artifact_root / "E1" / scenario / "quark-exact"
    complete_runs = [path.parent for path in sorted(method_root.glob("root=*/*/status.json"))]
    if not complete_runs:
        raise FileNotFoundError("No complete E1 runs are available for plotting.")
    path_rows: list[dict[str, Any]] = []
    learning_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for run_path in complete_runs:
        root_value = int(run_path.parent.name.split("=", 1)[1])
        for filename, destination in (
            ("regularization_path.csv", path_rows),
            ("learning_curve.csv", learning_rows),
            ("metrics.csv", metric_rows),
        ):
            for row in _read_rows(run_path / filename):
                destination.append({"root": root_value, **row})

    delay_rows = [row for row in metric_rows if str(row.get("task", "")).startswith("delay_")]
    panel_count = 3 if delay_rows else 2
    fig, axes = plt.subplots(1, panel_count, figsize=(5.2 * panel_count, 4.2))
    axes = np.atleast_1d(axes)
    norm_axis = axes[0].twinx()
    colors: dict[str, str] = {}
    for index, task in enumerate(sorted({row["task"] for row in path_rows})):
        colors[task] = f"C{index}"
        rows = [row for row in path_rows if row["task"] == task and row.get("status") == "ok"]
        lambdas, validation_mean, validation_std = _group_mean_std(
            rows, x=lambda row: float(row["lambda_K"]), y="validation_mse"
        )
        _, norm_mean, norm_std = _group_mean_std(
            rows, x=lambda row: float(row["lambda_K"]), y="rkhs_norm"
        )
        axes[0].plot(
            lambdas, validation_mean, marker="o", color=colors[task], label=f"{task}: MSE"
        )
        axes[0].fill_between(
            lambdas,
            np.maximum(validation_mean - validation_std, np.finfo(float).tiny),
            validation_mean + validation_std,
            color=colors[task],
            alpha=0.14,
        )
        norm_axis.plot(
            lambdas, norm_mean, linestyle="--", color=colors[task], alpha=0.8,
            label=f"{task}: RKHS norm",
        )
        norm_axis.fill_between(
            lambdas,
            np.maximum(norm_mean - norm_std, np.finfo(float).tiny),
            norm_mean + norm_std,
            color=colors[task],
            alpha=0.08,
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    norm_axis.set_yscale("log")
    axes[0].set_xlabel(r"$\lambda_K$")
    axes[0].set_ylabel("validation MSE")
    norm_axis.set_ylabel("RKHS norm")
    axes[0].set_title("Regularization path")
    handles_a, labels_a = axes[0].get_legend_handles_labels()
    handles_b, labels_b = norm_axis.get_legend_handles_labels()
    axes[0].legend(handles_a + handles_b, labels_a + labels_b, fontsize=7)

    for task in sorted({row["task"] for row in learning_rows}):
        rows = [row for row in learning_rows if row["task"] == task]
        prefixes, means, stds = _group_mean_std(
            rows, x=lambda row: int(row["prefix"]), y="test_nrmse"
        )
        axes[1].plot(prefixes, means, marker="o", label=task)
        axes[1].fill_between(prefixes, means - stds, means + stds, alpha=0.14)
    axes[1].set_xlabel("outer-training prefix")
    axes[1].set_ylabel("test NRMSE")
    axes[1].set_title("Nested sample-size study")
    axes[1].legend(fontsize=8)

    if delay_rows:
        delays, means, stds = _group_mean_std(
            delay_rows,
            x=lambda row: int(str(row["task"]).split("_")[-1]),
            y="test_nrmse",
        )
        axes[2].plot(delays, means, marker="o")
        axes[2].fill_between(delays, means - stds, means + stds, alpha=0.14)
        axes[2].set_xlabel("delay")
        axes[2].set_ylabel("test NRMSE")
        axes[2].set_title("Delayed-recall profile")
    fig.tight_layout()

    output_root = artifact_root / "aggregate" / "E1"
    prefix = "e1_functional_profile" if delay_rows else "e1_smoke_diagnostics"
    png = output_root / f"{prefix}.png"
    pdf = output_root / f"{prefix}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    summary = output_root / "e1_smoke_summary.csv"
    summary.write_text(
        (output_root / "metrics_long.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    latex = output_root / "e1_smoke_summary.tex"
    _write_latex_summary(latex, metric_rows)
    return {"png": png, "pdf": pdf, "summary": summary, "latex": latex}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root", default="artifacts/experiments/quark-empirical-v1"
    )
    parser.add_argument("--scenario", default="reference-varma-functional")
    arguments = parser.parse_args()
    outputs = plot_e1(artifact_root=arguments.artifact_root, scenario=arguments.scenario)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
