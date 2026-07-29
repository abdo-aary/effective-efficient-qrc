"""Protocol-level E1 artifacts layered on portable ``quark.run/v1`` files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

from src.artifacts.run import (
    _sha256,
    _write_json,
    load_feature_artifact,
    save_feature_artifact,
)
from src.core.capabilities import BackendKind, EstimatorKind
from src.core.results import ExecutionMetadata, FeatureBatch
from src.core.seeds import SeedBundle
from src.data.generate.stationary_varma import E1Dataset
from src.data.label.context import TeacherContext


def records_to_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    rows = list(rows)
    if not rows:
        return ""
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))
    return buffer.getvalue()


def _atomic_directory(destination: Path, writer: Any) -> Path:
    if destination.exists():
        raise FileExistsError(f"Immutable artifact already exists: {destination}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    )
    try:
        writer(temporary)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def save_e1_dataset_asset(
    path: str | Path, *, dataset: E1Dataset, seeds: SeedBundle
) -> Path:
    destination = Path(path)
    fingerprint = dataset.fingerprint()
    if destination.exists():
        loaded = load_e1_dataset_asset(destination)
        if loaded.fingerprint() != fingerprint:
            raise ValueError("Existing dataset asset has a different fingerprint.")
        return destination

    def write(root: Path) -> None:
        np.savez_compressed(root / "dataset.npz", windows=dataset.windows, targets=dataset.targets)
        np.savez_compressed(
            root / "context.npz",
            prediction_origins=dataset.context.prediction_origins,
            future_indices=dataset.context.future_indices,
            future_observations=dataset.context.future_observations,
        )
        np.savez_compressed(
            root / "process.npz",
            phi=dataset.phi,
            theta=dataset.theta,
            transition=dataset.transition,
            noise_map=dataset.noise_map,
            stationary_covariance=dataset.stationary_covariance,
        )
        _write_json(
            root / "dataset.json",
            {
                **dataset.metadata,
                "fingerprint": fingerprint,
                "task_names": dataset.task_names,
                "shapes": {
                    "windows": dataset.windows.shape,
                    "targets": dataset.targets.shape,
                },
            },
        )
        _write_json(root / "seeds.json", seeds.to_dict())
        checksums = {
            item.name: _sha256(item) for item in sorted(root.iterdir()) if item.is_file()
        }
        _write_json(
            root / "manifest.json",
            {
                "schema_version": "quark.e1-dataset/v1",
                "fingerprint": fingerprint,
                "checksums": checksums,
            },
        )

    return _atomic_directory(destination, write)


def load_e1_dataset_asset(path: str | Path) -> E1Dataset:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["checksums"].items():
        if _sha256(root / filename) != expected:
            raise ValueError(f"Dataset checksum mismatch for {filename}.")
    with np.load(root / "dataset.npz", allow_pickle=False) as arrays:
        windows, targets = arrays["windows"], arrays["targets"]
    with np.load(root / "context.npz", allow_pickle=False) as arrays:
        context = TeacherContext(
            arrays["prediction_origins"],
            arrays["future_indices"],
            arrays["future_observations"],
        )
    with np.load(root / "process.npz", allow_pickle=False) as arrays:
        process = {name: arrays[name] for name in arrays.files}
    metadata = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    task_names = tuple(metadata.pop("task_names"))
    metadata.pop("shapes", None)
    expected_fingerprint = metadata.pop("fingerprint")
    dataset = E1Dataset(
        windows=windows,
        targets=targets,
        task_names=task_names,
        context=context,
        prediction_origins=context.prediction_origins,
        phi=process["phi"],
        theta=process["theta"],
        transition=process["transition"],
        noise_map=process["noise_map"],
        stationary_covariance=process["stationary_covariance"],
        metadata=metadata,
    )
    if dataset.fingerprint() != expected_fingerprint:
        raise ValueError("Reconstructed dataset fingerprint does not match its manifest.")
    return dataset


@dataclass(frozen=True)
class E1ArtifactPaths:
    artifact_root: Path
    scenario: str
    method: str
    root: int
    configuration_hash: str

    @property
    def scenario_root(self) -> Path:
        return self.artifact_root / "E1" / self.scenario

    @property
    def run_path(self) -> Path:
        return (
            self.scenario_root
            / self.method
            / f"root={self.root}"
            / self.configuration_hash[:16]
        )

    @property
    def workspace(self) -> Path:
        return (
            self.artifact_root
            / ".work"
            / "E1"
            / f"root={self.root}"
            / self.configuration_hash
        )

    @property
    def feature_checkpoint(self) -> Path:
        return self.workspace / "features"

    @property
    def analysis_checkpoint(self) -> Path:
        return self.workspace / "analysis"

    @property
    def dataset_root(self) -> Path:
        return self.scenario_root / "datasets" / f"root={self.root}"


def reconstruct_feature_batch(path: str | Path) -> FeatureBatch:
    payload = load_feature_artifact(path)
    execution = payload["execution"]
    metadata_raw = execution["execution_metadata"]
    metadata = ExecutionMetadata(
        backend_kind=BackendKind(metadata_raw["backend_kind"]),
        estimator_kind=EstimatorKind(metadata_raw["estimator_kind"]),
        exact=bool(metadata_raw["exact"]),
        program_fingerprint=str(metadata_raw["program_fingerprint"]),
        started_at=str(metadata_raw["started_at"]),
        details=metadata_raw.get("details", {}),
    )
    return FeatureBatch(
        values=payload["results"]["values"],
        estimator_kind=EstimatorKind(execution["estimator_kind"]),
        estimator_metadata=execution["estimator_metadata"],
        execution_metadata=metadata,
        observable_labels=tuple(payload["program"]["observable_labels"]),
        reservoir_ids=tuple(payload["program"]["reservoir_ids"]),
        ordering_version=str(execution["feature_ordering"]),
    )


def save_failure_artifact(
    *, paths: E1ArtifactPaths, stage: str, error: BaseException, config: Mapping[str, Any]
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = (
        paths.scenario_root
        / paths.method
        / "failures"
        / f"root={paths.root}"
        / paths.configuration_hash[:16]
        / f"attempt={timestamp}"
    )

    def write(root: Path) -> None:
        _write_json(
            root / "status.json",
            {
                "status": "failed",
                "stage": stage,
                "exception_type": type(error).__name__,
                "message": str(error),
                "timestamp": timestamp,
            },
        )
        _write_json(root / "resolved_config.json", config)
        (root / "traceback.txt").write_text(
            "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            encoding="utf-8",
        )
        checksums = {
            item.name: _sha256(item) for item in sorted(root.iterdir()) if item.is_file()
        }
        _write_json(
            root / "manifest.json",
            {"schema_version": "quark.e1-failure/v1", "checksums": checksums},
        )

    return _atomic_directory(destination, write)


def save_analysis_checkpoint(
    path: str | Path,
    *,
    arrays: Mapping[str, np.ndarray],
    document: Mapping[str, Any],
    csv_documents: Mapping[str, str],
) -> Path:
    destination = Path(path)
    if destination.exists():
        load_analysis_checkpoint(destination)
        return destination

    def write(root: Path) -> None:
        np.savez_compressed(
            root / "analysis.npz",
            **{name: np.asarray(value) for name, value in arrays.items()},
        )
        _write_json(root / "analysis.json", document)
        for filename, content in csv_documents.items():
            if Path(filename).name != filename or not filename.endswith(".csv"):
                raise ValueError(f"Invalid analysis CSV filename {filename!r}.")
            (root / filename).write_text(content, encoding="utf-8")
        checksums = {
            item.name: _sha256(item) for item in sorted(root.iterdir()) if item.is_file()
        }
        _write_json(
            root / "manifest.json",
            {"schema_version": "quark.e1-analysis/v1", "checksums": checksums},
        )

    return _atomic_directory(destination, write)


def load_analysis_checkpoint(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["checksums"].items():
        if _sha256(root / filename) != expected:
            raise ValueError(f"Analysis checksum mismatch for {filename}.")
    with np.load(root / "analysis.npz", allow_pickle=False) as values:
        arrays = {name: values[name] for name in values.files}
    return {
        "arrays": arrays,
        "document": json.loads((root / "analysis.json").read_text(encoding="utf-8")),
        "csv": {
            path.name: path.read_text(encoding="utf-8")
            for path in root.glob("*.csv")
        },
    }
