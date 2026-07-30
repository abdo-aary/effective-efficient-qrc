"""Non-reportable N=1000 engineering pilot for the E1 finite Matérn grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.core.seeds import SeedBundle
from src.experiment.protocol import ChronologicalSplit
from src.models.matern_grid import FiniteGridMaternKRR


def run_pilot(*, output: str | Path, gpu_id: int = 0) -> Path:
    import cupy as cp

    cp.cuda.Device(gpu_id).use()
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    baseline = int(pool.total_bytes())
    rng = SeedBundle.from_root(91001).generator("model_selection")
    features = rng.normal(size=(1000, 315))
    targets = np.column_stack(
        (
            np.tanh(features[:, 0] + 0.2 * features[:, 1]),
            np.tanh(features[:, 2] * features[:, 3]),
            np.tanh(0.5 * features[:, 4] - features[:, 5]),
        )
    )
    model = FiniteGridMaternKRR(
        nu_grid=(1.5, 2.5, 5.0),
        xi_grid=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
        lambda_grid=(1e-6, 1e-4, 1e-2, 1.0, 1e2),
        backend="cupy",
    )
    started = perf_counter()
    result = model.fit_select_refit(
        features=features,
        targets=targets,
        task_names=("pilot_linear", "pilot_interaction", "pilot_mixed"),
        split=ChronologicalSplit(800, 900, 1000),
    )
    cp.cuda.Stream.null.synchronize()
    elapsed = perf_counter() - started
    solve_seconds = sum(float(row["seconds"]) for row in result.diagnostics["solve_rows"])
    kernel_seconds = sum(float(row["seconds"]) for row in result.diagnostics["kernel_rows"])
    prefix_inner_sizes = (200, 400, 800, 1600, 4000)
    cubic_factor = sum((size / 800.0) ** 3 for size in prefix_inner_sizes)
    quadratic_factor = sum((size / 800.0) ** 2 for size in prefix_inner_sizes)
    payload = {
        "schema_version": "quark.e1-readout-pilot/v1",
        "status": "engineering-only-not-paper-result",
        "data_semantics": "deterministic-gaussian-features",
        "root": 91001,
        "N": 1000,
        "p": 315,
        "tasks": 3,
        "candidate_count": model.candidate_count,
        "gpu_id": int(gpu_id),
        "elapsed_seconds": elapsed,
        "distance_seconds": result.diagnostics["distance_seconds"],
        "kernel_seconds": kernel_seconds,
        "factorization_prediction_seconds": solve_seconds,
        "peak_pool_increment_bytes": max(0, int(pool.total_bytes()) - baseline),
        "projection_model": {
            "prefix_inner_sizes": list(prefix_inner_sizes),
            "cubic_solve_multiplier_per_root": cubic_factor,
            "quadratic_kernel_multiplier_per_root": quadratic_factor,
            "projected_solve_seconds_per_root": solve_seconds * cubic_factor,
            "projected_kernel_seconds_per_root": kernel_seconds * quadratic_factor,
            "qualification": "engineering extrapolation; not an empirical result",
        },
        "kernel_rows": result.diagnostics["kernel_rows"],
        "solve_rows": result.diagnostics["solve_rows"],
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if destination.exists() and destination.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"A different pilot artifact exists at {destination}.")
    destination.write_text(text, encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=(
            "storage/artifacts/experiments/quark-empirical-v1/"
            "aggregate/E1/readout_pilot_n1000.json"
        ),
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    print(run_pilot(output=args.output, gpu_id=args.gpu_id))


if __name__ == "__main__":
    main()
