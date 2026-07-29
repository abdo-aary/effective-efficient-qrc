"""Portable artifact persistence without pickle."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from typing import Any, Mapping

import numpy as np

from src.core.program import QuaRKProgram
from src.core.requests import ExecutionSpec
from src.core.results import FeatureBatch
from src.estimators.csmom import ShadowSnapshots

ARTIFACT_SCHEMA_VERSION = "quark.run/v1"


class ArtifactIntegrityError(ValueError):
    """An artifact checksum or schema does not match its manifest."""


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _environment() -> dict[str, Any]:
    packages = {}
    for distribution in (
        "numpy",
        "scipy",
        "scikit-learn",
        "qiskit",
        "qiskit-aer",
        "qiskit-ibm-runtime",
        "cupy",
    ):
        try:
            packages[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            continue
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def save_feature_artifact(
    path: str | Path,
    *,
    program: QuaRKProgram,
    execution: ExecutionSpec,
    batch: FeatureBatch,
    resolved_configuration: Mapping[str, Any] | None = None,
    dataset_manifest: Mapping[str, Any] | None = None,
    predictions: np.ndarray | None = None,
    metrics: Mapping[str, Any] | None = None,
    logs: str | None = None,
) -> Path:
    """Atomically create an immutable feature-run directory."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(
            f"Artifact destination already exists: {destination}. "
            "Run artifacts are immutable; choose a new path."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    )
    try:
        np.savez_compressed(
            temporary / "program.npz",
            projection=program.projection.matrix,
            zz=program.reservoirs.zz,
            x_fields=program.reservoirs.x_fields,
            z_fields=program.reservoirs.z_fields,
            reset_rates=program.reservoirs.reset_rates,
        )
        values = batch.values.get() if hasattr(batch.values, "get") else batch.values
        arrays: dict[str, np.ndarray] = {"values": np.asarray(values)}
        if predictions is not None:
            arrays["predictions"] = np.asarray(predictions)
        np.savez_compressed(temporary / "results.npz", **arrays)
        if isinstance(batch.raw_snapshots, ShadowSnapshots):
            snapshot_arrays = {
                "bases": batch.raw_snapshots.bases,
                "outcomes": batch.raw_snapshots.outcomes,
            }
            if batch.raw_snapshots.trajectory_suffix_starts is not None:
                snapshot_arrays["trajectory_suffix_starts"] = (
                    batch.raw_snapshots.trajectory_suffix_starts
                )
                snapshot_arrays["trajectory_reset_counts"] = (
                    batch.raw_snapshots.trajectory_reset_counts
                )
            np.savez_compressed(
                temporary / "shadow_snapshots.npz", **snapshot_arrays
            )

        _write_json(temporary / "seeds.json", execution.seeds.to_dict())
        _write_json(
            temporary / "program.json",
            {
                "fingerprint": program.fingerprint(),
                "input_dim": program.input_dim,
                "num_qubits": program.num_qubits,
                "window_length": program.window_length,
                "angle_map": program.angle_map,
                "angle_scale": program.angle_scale,
                "projection_mode": program.projection.mode,
                "topology": {
                    "name": program.topology.name,
                    "edges": program.topology.edges,
                },
                "reset_state": program.reset_channel.state,
                "observable_labels": program.observables.labels,
                "reservoir_ids": program.reservoirs.reservoir_ids,
            },
        )
        _write_json(
            temporary / "execution.json",
            {
                "feature_shape": batch.shape,
                "feature_ordering": batch.ordering_version,
                "estimator_kind": batch.estimator_kind,
                "estimator_metadata": batch.estimator_metadata,
                "execution_metadata": batch.execution_metadata,
                "execution_request": {
                    "chunk_size": execution.chunk_size,
                    "retain_device_array": execution.retain_device_array,
                    "options": execution.options,
                },
            },
        )
        _write_json(temporary / "resolved_config.json", resolved_configuration or {})
        _write_json(temporary / "dataset.json", dataset_manifest or {})
        _write_json(temporary / "metrics.json", metrics or {})
        _write_json(temporary / "environment.json", _environment())
        (temporary / "run.log").write_text(logs or "", encoding="utf-8")
        _write_json(temporary / "status.json", {"status": "complete"})

        checksums = {
            item.name: _sha256(item)
            for item in sorted(temporary.iterdir())
            if item.is_file() and item.name != "manifest.json"
        }
        _write_json(
            temporary / "manifest.json",
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "git_commit": _git_commit(),
                "program_fingerprint": program.fingerprint(),
                "checksums": checksums,
            },
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_feature_artifact(path: str | Path) -> dict[str, Any]:
    """Validate and load every portable component of a feature artifact."""

    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactIntegrityError(
            f"Unsupported artifact schema {manifest.get('schema_version')!r}."
        )
    for name, expected in manifest.get("checksums", {}).items():
        actual = _sha256(root / name)
        if actual != expected:
            raise ArtifactIntegrityError(
                f"Checksum mismatch for {name}: expected {expected}, got {actual}."
            )
    with np.load(root / "results.npz", allow_pickle=False) as arrays:
        results = {name: arrays[name] for name in arrays.files}
    with np.load(root / "program.npz", allow_pickle=False) as arrays:
        program_arrays = {name: arrays[name] for name in arrays.files}
    shadow_snapshots = None
    if (root / "shadow_snapshots.npz").exists():
        with np.load(root / "shadow_snapshots.npz", allow_pickle=False) as arrays:
            shadow_snapshots = {name: arrays[name] for name in arrays.files}
    return {
        "manifest": manifest,
        "program": json.loads((root / "program.json").read_text(encoding="utf-8")),
        "program_arrays": program_arrays,
        "execution": json.loads(
            (root / "execution.json").read_text(encoding="utf-8")
        ),
        "resolved_configuration": json.loads(
            (root / "resolved_config.json").read_text(encoding="utf-8")
        ),
        "dataset": json.loads((root / "dataset.json").read_text(encoding="utf-8")),
        "seeds": json.loads((root / "seeds.json").read_text(encoding="utf-8")),
        "environment": json.loads(
            (root / "environment.json").read_text(encoding="utf-8")
        ),
        "metrics": json.loads((root / "metrics.json").read_text(encoding="utf-8")),
        "status": json.loads((root / "status.json").read_text(encoding="utf-8")),
        "logs": (root / "run.log").read_text(encoding="utf-8"),
        "results": results,
        "shadow_snapshots": shadow_snapshots,
    }
