"""Experiment E1: exact-NVIDIA functional hierarchy vertical slice."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from src.artifacts.experiment import (
    E1ArtifactPaths,
    load_analysis_checkpoint,
    reconstruct_feature_batch,
    records_to_csv,
    save_analysis_checkpoint,
    save_e1_dataset_asset,
    save_failure_artifact,
)
from src.artifacts.run import save_feature_artifact
from src.backends.nvidia import NvidiaBackend
from src.core.capabilities import BackendKind, EstimatorKind, Precision
from src.core.factories import make_protocol_program
from src.core.requests import ExecutionSpec
from src.core.results import ExecutionMetadata, FeatureBatch
from src.core.seeds import SeedBundle
from src.data.generate.stationary_varma import (
    StationaryVARMASpec,
    generate_e1_dataset,
)
from src.estimators.exact import ExactFeatureEstimator
from src.experiment.protocol import ChronologicalSplit, E1ProtocolSpec
from src.models.matern_grid import FiniteGridMaternKRR, regularization_path


@dataclass(frozen=True)
class E1RunResult:
    stage: str
    path: Path | None
    configuration_hash: str | None
    details: Mapping[str, Any]


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _fallback_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(_plain(config), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def campaign_plan(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    roots = tuple(int(value) for value in config.get("structural_roots", [config["root"]]))
    data = config["data"]
    program = config["program"]
    n, R = int(program["num_qubits"]), int(program["num_reservoirs"])
    k = int(program["locality"])
    K = sum(math.comb(n, weight) * 3**weight for weight in range(1, k + 1))
    N = int(data["N"])
    return [
        {
            "root": root,
            "stage": "features",
            "backend": "nvidia_gpu",
            "estimator": "exact",
            "precision": "complex128",
            "feature_shape": [N, R, K],
            "feature_bytes": N * R * K * 8,
            "program": {"n": n, "R": R, "k": k, "w": int(data["window_length"])},
            "status": "planned-not-run",
        }
        for root in roots
    ]


def _write_campaign_plan(config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Path:
    root = Path(str(config["artifacts"]["root"])) / "aggregate" / "E1"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "campaign_plan.json"
    payload = {
        "protocol_version": config["protocol_version"],
        "experiment_contract": config["experiment_contract"],
        "implementation_revision": config["implementation_revision"],
        "jobs": list(rows),
    }
    text = json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n"
    if destination.exists() and destination.read_text(encoding="utf-8") != text:
        raise FileExistsError("A different E1 campaign plan already exists.")
    destination.write_text(text, encoding="utf-8")
    return destination


def _assert_exact_nvidia(backend: NvidiaBackend, estimator: ExactFeatureEstimator) -> None:
    capabilities = backend.capabilities
    if capabilities.backend_kind is not BackendKind.NVIDIA:
        raise RuntimeError("E1 requires BackendKind.NVIDIA.")
    if not capabilities.gpu_execution or not capabilities.exact_mixed_state_evolution:
        raise RuntimeError("E1 requires exact GPU mixed-state execution.")
    if estimator.kind is not EstimatorKind.EXACT:
        raise RuntimeError("E1 requires ExactFeatureEstimator.")
    if estimator.precision is not Precision.COMPLEX128 or estimator.return_states:
        raise RuntimeError("E1 requires complex128 direct expectations without states.")
    capabilities.require_estimator(EstimatorKind.EXACT)


def _select_tasks(dataset: Any, names: Sequence[str]) -> tuple[np.ndarray, tuple[str, ...]]:
    lookup = {name: index for index, name in enumerate(dataset.task_names)}
    requested = tuple(str(name) for name in names)
    missing = [name for name in requested if name not in lookup]
    if missing:
        raise KeyError(f"Dataset does not contain E1 tasks {missing}.")
    return dataset.targets[:, [lookup[name] for name in requested]], requested


def _feature_checkpoint(
    *,
    paths: E1ArtifactPaths,
    dataset: Any,
    program: Any,
    seeds: SeedBundle,
    spec: E1ProtocolSpec,
    resolved_config: Mapping[str, Any],
) -> tuple[FeatureBatch, dict[str, float]]:
    if paths.feature_checkpoint.exists():
        batch = reconstruct_feature_batch(paths.feature_checkpoint)
        if batch.execution_metadata.program_fingerprint != program.fingerprint():
            raise ValueError("Feature checkpoint belongs to a different program.")
        return batch, {"checkpoint_reused": 1.0}

    backend = NvidiaBackend(gpu_id=spec.gpu_id, chunk_size=spec.chunk_size, engine=spec.engine)
    estimator = ExactFeatureEstimator(precision=Precision.COMPLEX128, return_states=False)
    _assert_exact_nvidia(backend, estimator)
    execution = ExecutionSpec(
        seeds=seeds, chunk_size=spec.chunk_size, retain_device_array=True
    )
    compile_started = perf_counter()
    compiled = backend.compile(program, estimator)
    compile_seconds = perf_counter() - compile_started
    if compiled.compilation_metadata.get("history_truncation") is not False:
        raise RuntimeError("E1 exact execution forbids history truncation.")
    if compiled.compilation_metadata.get("branch_pruning") is not False:
        raise RuntimeError("E1 exact execution forbids branch pruning.")
    batch_device = backend.execute(compiled, dataset.windows, execution)
    if not batch_device.execution_metadata.exact:
        raise RuntimeError("NVIDIA backend returned non-exact E1 metadata.")

    import cupy as cp

    cp.cuda.Stream.null.synchronize()
    transfer_started = perf_counter()
    batch = batch_device.as_numpy()
    cp.cuda.Stream.null.synchronize()
    transfer_seconds = perf_counter() - transfer_started
    values = np.asarray(batch.values)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Exact NVIDIA features contain NaN or Inf.")
    if np.max(np.abs(values)) > 1.0 + 1e-10:
        raise FloatingPointError("Exact Pauli expectations lie outside [-1,1].")
    details = dict(batch.execution_metadata.details)
    timings = dict(details.get("timings", {}))
    timings.update(
        {
            "compile_seconds": float(compile_seconds),
            "device_to_host_seconds": float(transfer_seconds),
        }
    )
    details["timings"] = timings
    metadata = ExecutionMetadata(
        backend_kind=batch.execution_metadata.backend_kind,
        estimator_kind=batch.execution_metadata.estimator_kind,
        exact=True,
        program_fingerprint=batch.execution_metadata.program_fingerprint,
        started_at=batch.execution_metadata.started_at,
        details=details,
    )
    batch = FeatureBatch(
        values=values,
        estimator_kind=batch.estimator_kind,
        estimator_metadata=batch.estimator_metadata,
        execution_metadata=metadata,
        observable_labels=batch.observable_labels,
        reservoir_ids=batch.reservoir_ids,
        ordering_version=batch.ordering_version,
    )
    save_feature_artifact(
        paths.feature_checkpoint,
        program=program,
        execution=execution,
        batch=batch,
        resolved_configuration=resolved_config,
        dataset_manifest={
            **dataset.metadata,
            "fingerprint": dataset.fingerprint(),
            "task_names": dataset.task_names,
        },
        extra_json={
            "run_identity.json": {
                "configuration_hash": paths.configuration_hash,
                "dataset_fingerprint": dataset.fingerprint(),
                "program_fingerprint": program.fingerprint(),
                "stage": "features",
            }
        },
        artifact_kind="e1-feature-checkpoint",
    )
    return batch, timings


def _analyze(
    *,
    paths: E1ArtifactPaths,
    batch: FeatureBatch,
    dataset: Any,
    spec: E1ProtocolSpec,
) -> dict[str, Any]:
    if paths.analysis_checkpoint.exists():
        return load_analysis_checkpoint(paths.analysis_checkpoint)
    features = np.asarray(batch.flatten_for_readout(), dtype=np.float64)
    targets, task_names = _select_tasks(dataset, spec.task_names)
    readout = FiniteGridMaternKRR(
        nu_grid=spec.nu_grid,
        xi_grid=spec.xi_grid,
        lambda_grid=spec.lambda_grid,
        backend="cupy",
        tie_tolerance=spec.tie_tolerance,
    )
    main = readout.fit_select_refit(
        features=features, targets=targets, task_names=task_names, split=spec.split
    )
    hierarchy_names = tuple(
        name for name in ("future", "exp_memory", "volterra") if name in task_names
    )
    hierarchy_indices = [task_names.index(name) for name in hierarchy_names]
    path_rows = regularization_path(
        features=features,
        targets=targets[:, hierarchy_indices],
        task_names=hierarchy_names,
        split=spec.split,
        lambda_grid=spec.path_lambda_grid,
        nu=spec.path_nu,
        xi=spec.path_xi,
        backend="cupy",
    )
    learning_rows: list[dict[str, Any]] = []
    nested_seconds = 0.0
    for prefix in spec.prefixes:
        if prefix == len(spec.split.outer_train):
            for row in main.metric_rows:
                if row["task"] in hierarchy_names:
                    learning_rows.append({**row, "prefix": prefix, "reused_main": True})
            continue
        prefix_split = ChronologicalSplit.for_prefix(
            prefix=prefix,
            test_start=int(spec.split.test_start),
            test_stop=spec.split.test_stop,
        )
        nested = readout.fit_select_refit(
            features=features,
            targets=targets[:, hierarchy_indices],
            task_names=hierarchy_names,
            split=prefix_split,
        )
        nested_seconds += nested.elapsed_seconds
        for row in nested.metric_rows:
            learning_rows.append({**row, "prefix": prefix, "reused_main": False})

    document = {
        "schema_version": "quark.e1-analysis/v1",
        "task_names": task_names,
        "selected": main.selected,
        "metric_rows": main.metric_rows,
        "path_rows": path_rows,
        "learning_rows": learning_rows,
        "readout_diagnostics": main.diagnostics,
        "timings": {
            "main_readout_seconds": main.elapsed_seconds,
            "nested_readout_seconds": nested_seconds,
        },
    }
    arrays = {
        "predictions": main.predictions,
        "alphas": main.alphas,
        "scaler_mean": main.scaler.mean,
        "scaler_scale": main.scaler.scale,
        "scaler_constant_mask": main.scaler.constant_mask,
    }
    csv_documents = {
        "metrics.csv": records_to_csv(main.metric_rows),
        "readout_candidates.csv": records_to_csv(main.candidate_rows),
        "regularization_path.csv": records_to_csv(path_rows),
        "learning_curve.csv": records_to_csv(learning_rows),
    }
    save_analysis_checkpoint(
        paths.analysis_checkpoint,
        arrays=arrays,
        document=document,
        csv_documents=csv_documents,
    )
    return load_analysis_checkpoint(paths.analysis_checkpoint)


def _finalize_run(
    *,
    paths: E1ArtifactPaths,
    batch: FeatureBatch,
    dataset: Any,
    program: Any,
    seeds: SeedBundle,
    spec: E1ProtocolSpec,
    resolved_config: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> Path:
    if paths.run_path.exists():
        return paths.run_path
    document = analysis["document"]
    arrays = analysis["arrays"]
    feature_details = dict(batch.execution_metadata.details)
    feature_timings = dict(feature_details.get("timings", {}))
    analysis_timings = dict(document.get("timings", {}))
    timings = {**feature_timings, **analysis_timings}
    N, R, K = batch.shape
    resources = {
        **dict(feature_details.get("resources", {})),
        "N": N,
        "R": R,
        "K": K,
        "p": R * K,
        "reservoir_outputs": N * R,
        "temporal_channel_updates": N * R * spec.window_length,
    }
    execution = ExecutionSpec(
        seeds=seeds, chunk_size=spec.chunk_size, retain_device_array=True
    )
    save_feature_artifact(
        paths.run_path,
        program=program,
        execution=execution,
        batch=batch,
        resolved_configuration=resolved_config,
        dataset_manifest={
            **dataset.metadata,
            "fingerprint": dataset.fingerprint(),
            "task_names": dataset.task_names,
        },
        metrics={"rows": document["metric_rows"]},
        logs="E1 exact-NVIDIA run completed from checksum-validated stage artifacts.\n",
        extra_arrays={
            "targets.npz": {"values": dataset.targets},
            "context.npz": {
                "prediction_origins": dataset.context.prediction_origins,
                "future_indices": dataset.context.future_indices,
                "future_observations": dataset.context.future_observations,
            },
            "splits.npz": spec.split.to_arrays(),
            "predictions.npz": {"values": arrays["predictions"]},
            "readout.npz": {
                "alphas": arrays["alphas"],
                "scaler_mean": arrays["scaler_mean"],
                "scaler_scale": arrays["scaler_scale"],
                "scaler_constant_mask": arrays["scaler_constant_mask"],
            },
        },
        extra_json={
            "run_identity.json": {
                "configuration_hash": paths.configuration_hash,
                "dataset_fingerprint": dataset.fingerprint(),
                "program_fingerprint": program.fingerprint(),
                "stage": "complete",
            },
            "selected_readouts.json": {"selected": document["selected"]},
            "timings.json": timings,
            "resources.json": resources,
        },
        extra_text=analysis["csv"],
        artifact_kind="e1-exact-nvidia-run",
    )
    return paths.run_path


def run_e1(config: Mapping[str, Any]) -> E1RunResult:
    """Run one requested E1 stage without hidden backend or protocol changes."""

    config = _plain(config)
    stage = str(config.get("stage", "plan"))
    if stage not in {"plan", "features", "analyze", "all"}:
        raise ValueError("stage must be plan, features, analyze, or all.")
    if stage == "plan":
        rows = campaign_plan(config)
        path = _write_campaign_plan(config, rows)
        return E1RunResult(stage=stage, path=path, configuration_hash=None, details={"jobs": rows})

    spec = E1ProtocolSpec.from_mapping(config)
    if spec.mode == "full" and not bool(config.get("allow_full", False)):
        raise PermissionError("Full E1 execution requires allow_full=true.")
    seeds = SeedBundle.from_root(spec.root)
    artifact_root = Path(str(config["artifacts"]["root"]))
    failure_paths = E1ArtifactPaths(
        artifact_root=artifact_root,
        scenario=spec.scenario,
        method=spec.method,
        root=spec.root,
        configuration_hash=_fallback_hash(config),
    )
    try:
        dataset = generate_e1_dataset(
            seeds=seeds,
            num_windows=spec.num_windows,
            window_length=spec.window_length,
            stride=spec.stride,
            spec=StationaryVARMASpec.from_mapping(config["data"]),
        )
        dataset_hash = dataset.fingerprint()
        dataset_path = (
            artifact_root
            / "E1"
            / spec.scenario
            / "datasets"
            / f"root={spec.root}"
            / dataset_hash[:16]
        )
        save_e1_dataset_asset(dataset_path, dataset=dataset, seeds=seeds)
        program = make_protocol_program(
            input_dim=dataset.windows.shape[2],
            num_qubits=spec.num_qubits,
            num_reservoirs=spec.num_reservoirs,
            locality=spec.locality,
            reset_rate=spec.reset_rate,
            window_length=spec.window_length,
            seeds=seeds,
            angle_map="tanh",
        )
        configuration_hash = spec.configuration_hash(
            dataset_hash=dataset_hash,
            program_fingerprint=program.fingerprint(),
            resolved_configuration=config,
        )
        paths = E1ArtifactPaths(
            artifact_root=artifact_root,
            scenario=spec.scenario,
            method=spec.method,
            root=spec.root,
            configuration_hash=configuration_hash,
        )
        batch, feature_timings = _feature_checkpoint(
            paths=paths,
            dataset=dataset,
            program=program,
            seeds=seeds,
            spec=spec,
            resolved_config=config,
        )
        if stage == "features":
            return E1RunResult(
                stage=stage,
                path=paths.feature_checkpoint,
                configuration_hash=configuration_hash,
                details={"timings": feature_timings, "dataset_path": str(dataset_path)},
            )
        analysis = _analyze(paths=paths, batch=batch, dataset=dataset, spec=spec)
        run_path = _finalize_run(
            paths=paths,
            batch=batch,
            dataset=dataset,
            program=program,
            seeds=seeds,
            spec=spec,
            resolved_config=config,
            analysis=analysis,
        )
        return E1RunResult(
            stage=stage,
            path=run_path,
            configuration_hash=configuration_hash,
            details={"dataset_path": str(dataset_path)},
        )
    except Exception as exc:
        try:
            save_failure_artifact(
                paths=locals().get("paths", failure_paths),
                stage=stage,
                error=exc,
                config=config,
            )
        except Exception:
            pass
        raise


def aggregate_e1(
    *, artifact_root: str | Path, scenario: str = "reference-varma-functional"
) -> dict[str, Path]:
    """Regenerate E1 aggregate tables from immutable successful run artifacts."""

    import csv
    import io

    artifact_root = Path(artifact_root)
    method_root = artifact_root / "E1" / scenario / "quark-exact"
    run_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for status_path in sorted(method_root.glob("root=*/*/status.json")):
        run_path = status_path.parent
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "complete":
            continue
        manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
        execution = json.loads((run_path / "execution.json").read_text(encoding="utf-8"))
        root_value = int(run_path.parent.name.split("=", 1)[1])
        run_rows.append(
            {
                "root": root_value,
                "configuration_hash": run_path.name,
                "path": str(run_path),
                "status": "complete",
                "git_commit": manifest.get("git_commit"),
                "program_fingerprint": manifest.get("program_fingerprint"),
                "feature_shape": "x".join(map(str, execution["feature_shape"])),
            }
        )
        metrics = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
        for row in metrics.get("rows", []):
            metric_rows.append({"root": root_value, "run_path": str(run_path), **row})

    aggregate_root = artifact_root / "aggregate" / "E1"
    aggregate_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "run_index": aggregate_root / "run_index.csv",
        "metrics": aggregate_root / "metrics_long.csv",
    }
    for key, rows in (("run_index", run_rows), ("metrics", metric_rows)):
        destination = outputs[key]
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(records_to_csv(rows), encoding="utf-8")
        temporary.replace(destination)
    return outputs
