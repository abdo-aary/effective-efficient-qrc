"""Typed result returned by the high-level experiment lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np

from src.artifacts import save_feature_artifact

if TYPE_CHECKING:
    from src.experiment.experiment import Experiment


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ExperimentResult:
    experiment: "Experiment"
    metrics: Mapping[str, Any]
    elapsed_seconds: float

    def save(
        self,
        path: str | Path,
        *,
        resolved_configuration: Mapping[str, Any] | None = None,
        logs: str | None = None,
    ) -> Path:
        """Persist the run through the versioned portable artifact API."""

        featurizer = self.experiment.model.featurizer
        program = getattr(featurizer, "program", None)
        execution = getattr(featurizer, "execution", None)
        batch = getattr(featurizer, "last_feature_batch_", None)
        if program is None or execution is None or batch is None:
            raise RuntimeError(
                "Portable run persistence requires a backend-neutral featurizer "
                "that has completed feature extraction."
            )
        dataset_manifest: dict[str, Any] = {
            "shape_X": tuple(int(value) for value in self.experiment.dataset.X.shape),
            "shape_y": tuple(int(value) for value in self.experiment.dataset.y.shape),
            "meta": self.experiment.dataset.meta,
        }
        artifact = self.experiment.dataset_artifact
        if artifact is not None and artifact.data_path.exists():
            dataset_manifest.update(
                {
                    "data_path": str(artifact.data_path),
                    "sha256": _file_sha256(artifact.data_path),
                }
            )
        predictions = getattr(self.experiment.model, "y_pred_test_", None)
        return save_feature_artifact(
            path,
            program=program,
            execution=execution,
            batch=batch,
            resolved_configuration=resolved_configuration,
            dataset_manifest=dataset_manifest,
            predictions=None if predictions is None else np.asarray(predictions),
            metrics={
                "values": self.metrics,
                "elapsed_seconds": self.elapsed_seconds,
            },
            logs=logs,
        )
