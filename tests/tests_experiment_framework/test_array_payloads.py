from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from src.experiment import ArrayAsset, LocalArtifactStore, NodePayload, Stage
from src.experiment.artifact_store import ArtifactIntegrityError
from src.experiment.providers import ProviderIdentity

from .helpers import minimal_plan


IDENTITY = ProviderIdentity("array-test", "v1", "cpu", "float64")


def test_owned_array_reload_is_read_only_mmap_and_prefix_view_is_referenced(tmp_path):
    plan = minimal_plan("array-assets")
    store = LocalArtifactStore(tmp_path, plan)
    values = np.arange(48, dtype=np.float64).reshape(4, 3, 4)

    owner = store.execute_node(
        stage=Stage.PREPARE,
        node_id="owner",
        spec={"kind": "owner"},
        upstream=(),
        producer_identity=IDENTITY,
        producer=lambda: NodePayload({"role": "owner"}, {"features": values}),
    )
    asset = owner.payload.require_asset("features")
    loaded = asset.load()
    assert isinstance(loaded.base, np.memmap)
    assert loaded.flags.writeable is False
    np.testing.assert_array_equal(loaded, values)

    view = asset.view((slice(None), slice(0, 2), slice(None)))
    derived = store.execute_node(
        stage=Stage.ACQUIRE,
        node_id="derived",
        spec={"kind": "prefix"},
        upstream=(owner,),
        producer_identity=IDENTITY,
        producer=lambda: NodePayload({"role": "derived"}, {"features": view}),
    )
    derived_asset = derived.payload.require_asset("features")
    assert derived_asset.path == asset.path
    assert derived_asset.shape == (4, 2, 4)
    assert not (derived.path / "arrays").exists()
    np.testing.assert_array_equal(derived_asset.load(), values[:, :2, :])


def test_array_references_survive_run_root_relocation(tmp_path):
    plan = minimal_plan("relocation")
    original = tmp_path / "original"
    store = LocalArtifactStore(original, plan)
    owner = store.execute_node(
        stage=Stage.PREPARE,
        node_id="owner",
        spec={},
        upstream=(),
        producer_identity=IDENTITY,
        producer=lambda: NodePayload({}, {"x": np.arange(8, dtype=np.int64)}),
    )
    store.execute_node(
        stage=Stage.ACQUIRE,
        node_id="view",
        spec={},
        upstream=(owner,),
        producer_identity=IDENTITY,
        producer=lambda: NodePayload({}, {"x": owner.payload.require_asset("x").view((slice(2, 6),))}),
    )

    moved = tmp_path / "moved"
    shutil.move(str(original), moved)
    moved_store = LocalArtifactStore(moved, plan)
    reloaded_owner = moved_store.execute_node(
        stage=Stage.PREPARE,
        node_id="owner",
        spec={},
        upstream=(),
        producer_identity=IDENTITY,
        producer=lambda: pytest.fail("owner should be cached"),
    )
    reloaded_view = moved_store.execute_node(
        stage=Stage.ACQUIRE,
        node_id="view",
        spec={},
        upstream=(reloaded_owner,),
        producer_identity=IDENTITY,
        producer=lambda: pytest.fail("view should be cached"),
    )
    np.testing.assert_array_equal(
        reloaded_view.payload.require_asset("x").load(),
        np.arange(2, 6, dtype=np.int64),
    )


def test_reference_requires_an_explicit_upstream_source_digest(tmp_path):
    plan = minimal_plan("source-guard")
    store = LocalArtifactStore(tmp_path, plan)
    owner = store.execute_node(
        stage=Stage.PREPARE,
        node_id="owner",
        spec={},
        upstream=(),
        producer_identity=IDENTITY,
        producer=lambda: NodePayload({}, {"x": np.arange(4)}),
    )
    asset = owner.payload.require_asset("x")
    forged = ArrayAsset(
        path=asset.path,
        checksum=asset.checksum,
        dtype=asset.dtype,
        base_shape=asset.base_shape,
        source_digest="0" * 64,
    )
    with pytest.raises(ArtifactIntegrityError, match="outside this node"):
        store.execute_node(
            stage=Stage.ACQUIRE,
            node_id="forged",
            spec={},
            upstream=(owner,),
            producer_identity=IDENTITY,
            producer=lambda: NodePayload({}, {"x": forged}),
        )


def test_array_corruption_and_provider_identity_separation(tmp_path):
    plan = minimal_plan("identity-separation")
    store = LocalArtifactStore(tmp_path, plan)
    first = store.execute_node(
        stage=Stage.PREPARE,
        node_id="node",
        spec={},
        upstream=(),
        producer_identity=IDENTITY,
        producer=lambda: NodePayload({}, {"x": np.arange(4, dtype=np.float64)}),
    )
    numerical = ProviderIdentity("numerical", "v1", "aer", "complex128")
    second = store.execute_node(
        stage=Stage.PREPARE,
        node_id="node",
        spec={},
        upstream=(),
        producer_identity=numerical,
        producer=lambda: NodePayload({}, {"x": np.arange(4, dtype=np.float64)}),
    )
    assert first.digest != second.digest
    assert first.path != second.path

    array_path = first.payload.require_asset("x").path
    array_path.write_bytes(array_path.read_bytes()[:-8] + b"corrupt!")
    fresh_store = LocalArtifactStore(tmp_path, plan)
    with pytest.raises(ArtifactIntegrityError, match="Checksum mismatch"):
        fresh_store.execute_node(
            stage=Stage.PREPARE,
            node_id="node",
            spec={},
            upstream=(),
            producer_identity=IDENTITY,
            producer=lambda: pytest.fail("corrupted artifacts are not cache hits"),
        )


def test_payload_json_contains_only_metadata_and_relative_descriptors(tmp_path):
    plan = minimal_plan("descriptors")
    store = LocalArtifactStore(tmp_path, plan)
    ref = store.execute_node(
        stage=Stage.PREPARE,
        node_id="node",
        spec={},
        upstream=(),
        producer_identity=IDENTITY,
        producer=lambda: NodePayload({"value": 3}, {"x": np.arange(3)}),
    )
    raw = json.loads((ref.path / "payload.json").read_text(encoding="utf-8"))
    descriptor = raw["assets"]["x"]
    assert raw["metadata"] == {"value": 3}
    assert not Path(descriptor["path"]).is_absolute()
    assert descriptor["path"].endswith(".npy")
    assert descriptor["dtype"] == np.dtype(np.int64).str
