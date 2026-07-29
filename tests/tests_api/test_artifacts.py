from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from src.artifacts import ArtifactIntegrityError, load_feature_artifact, save_feature_artifact
from src.core import (
    BackendKind,
    EstimatorKind,
    ExecutionMetadata,
    ExecutionSpec,
    FeatureBatch,
    ObservableSet,
    ProjectionSpec,
    QuaRKProgram,
    ResetChannelSpec,
    ReservoirParameters,
    ReservoirTopology,
    SeedBundle,
)
from src.estimators.csmom import ShadowSnapshots


def _objects():
    program = QuaRKProgram(
        ProjectionSpec(np.eye(1), "identity"),
        ReservoirTopology(1, (), "single"),
        ReservoirParameters(
            np.zeros((1, 0)),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.asarray([0.5]),
        ),
        ResetChannelSpec(),
        ObservableSet(("X", "Z")),
        1,
    )
    execution = ExecutionSpec(seeds=SeedBundle.from_root(3))
    batch = FeatureBatch(
        values=np.asarray([[[1.0, 0.0]]]),
        estimator_kind=EstimatorKind.EXACT,
        estimator_metadata={"semantics": "analytic_exact"},
        execution_metadata=ExecutionMetadata(
            backend_kind=BackendKind.AER,
            estimator_kind=EstimatorKind.EXACT,
            exact=True,
            program_fingerprint=program.fingerprint(),
        ),
        observable_labels=program.observables.labels,
        reservoir_ids=program.reservoirs.reservoir_ids,
    )
    return program, execution, batch


def test_portable_artifact_roundtrip_and_immutability(tmp_path):
    program, execution, batch = _objects()
    path = save_feature_artifact(
        tmp_path / "run",
        program=program,
        execution=execution,
        batch=batch,
        dataset_manifest={"sha256": "dataset-checksum"},
        metrics={"mse": 0.1},
    )
    loaded = load_feature_artifact(path)
    np.testing.assert_array_equal(loaded["results"]["values"], batch.values)
    assert loaded["manifest"]["schema_version"] == "quark.run/v1"
    assert loaded["metrics"] == {"mse": 0.1}
    with pytest.raises(FileExistsError):
        save_feature_artifact(
            path,
            program=program,
            execution=execution,
            batch=batch,
        )


def test_artifact_checksum_detects_tampering(tmp_path):
    program, execution, batch = _objects()
    path = save_feature_artifact(
        tmp_path / "run",
        program=program,
        execution=execution,
        batch=batch,
    )
    (path / "metrics.json").write_text(json.dumps({"changed": True}), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError):
        load_feature_artifact(path)


def test_artifact_preserves_ibm_trajectory_provenance(tmp_path):
    program, execution, batch = _objects()
    shape = (1, 1, 3, 1)
    snapshots = ShadowSnapshots(
        bases=np.zeros(shape, dtype=np.uint8),
        outcomes=np.ones(shape, dtype=np.int8),
        trajectory_suffix_starts=np.asarray([[[0, 2, 1]]]),
        trajectory_reset_counts=np.asarray([[[0, 1, 2]]]),
    )
    batch = replace(batch, raw_snapshots=snapshots)
    destination = save_feature_artifact(
        tmp_path / "ibm-run",
        program=program,
        execution=execution,
        batch=batch,
    )
    loaded = load_feature_artifact(destination)
    np.testing.assert_array_equal(
        loaded["shadow_snapshots"]["trajectory_suffix_starts"],
        snapshots.trajectory_suffix_starts,
    )
    np.testing.assert_array_equal(
        loaded["shadow_snapshots"]["trajectory_reset_counts"],
        snapshots.trajectory_reset_counts,
    )
