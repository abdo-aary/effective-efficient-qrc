"""Immutable, checksum-validated local artifacts for experiment graph nodes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import traceback
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import uuid4

import numpy as np

from .domain import ExperimentPlan, Stage, content_digest, json_value
from .payload import ArrayAsset, ArraySlice, NodePayload, PAYLOAD_SCHEMA_VERSION


NODE_SCHEMA_VERSION = "quark.experiment-node/v2"


class ArtifactIntegrityError(ValueError):
    pass


@dataclass(frozen=True)
class ArtifactRef:
    stage: Stage
    node_id: str
    digest: str
    path: Path
    payload: NodePayload
    reused: bool


class ArtifactStore(Protocol):
    @property
    def run_root(self) -> Path:
        ...

    @property
    def experiment_identity(self) -> tuple[str, int, str, str]:
        ...

    def execute_node(
        self,
        *,
        stage: Stage,
        node_id: str,
        spec: object,
        upstream: Sequence[ArtifactRef],
        producer_identity: object,
        producer: Callable[[], NodePayload | Mapping[str, Any]],
    ) -> ArtifactRef:
        ...

    def record_failure(
        self, *, stage: Stage, node_id: str, error: BaseException
    ) -> Path:
        ...


class LocalArtifactStore:
    def __init__(self, root: str | Path, plan: ExperimentPlan) -> None:
        self.root = Path(root)
        self.plan = plan
        self._checksum_cache: dict[Path, tuple[int, int, str]] = {}

    @property
    def experiment_identity(self) -> tuple[str, int, str, str]:
        return (
            self.plan.campaign.value,
            self.plan.repetition.index,
            self.plan.manifest_digest,
            self.plan.digest,
        )

    @property
    def run_root(self) -> Path:
        return (
            self.root
            / "runs"
            / self.plan.campaign.value
            / f"repetition={self.plan.repetition.index:03d}"
            / self.plan.manifest_digest[:16]
        )

    def execute_node(
        self,
        *,
        stage: Stage,
        node_id: str,
        spec: object,
        upstream: Sequence[ArtifactRef],
        producer_identity: object,
        producer: Callable[[], NodePayload | Mapping[str, Any]],
    ) -> ArtifactRef:
        stage = Stage(stage)
        identity = {
            "schema_version": NODE_SCHEMA_VERSION,
            "campaign": self.plan.campaign.value,
            "repetition": self.plan.repetition.index,
            "manifest_digest": self.plan.manifest_digest,
            "plan_digest": self.plan.digest,
            "stage": stage.value,
            "node_id": node_id,
            "spec": json_value(spec),
            "producer_identity": json_value(producer_identity),
            "upstream": [
                {"node_id": item.node_id, "digest": item.digest} for item in upstream
            ],
        }
        digest = content_digest(identity)
        destination = (
            self.run_root / stage.value / _safe_node_name(node_id) / digest[:16]
        )
        if destination.exists():
            payload = self._load(destination, expected_digest=digest)
            return ArtifactRef(stage, node_id, digest, destination, payload, True)

        payload = NodePayload.coerce(producer())
        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{digest[:12]}.", dir=str(parent))
        )
        try:
            _write_json(temporary / "node.json", {**identity, "digest": digest})
            self._write_payload(
                temporary=temporary,
                destination=destination,
                digest=digest,
                payload=payload,
                upstream=upstream,
            )
            checksums = {
                str(path.relative_to(temporary)): _sha256(path)
                for path in sorted(temporary.rglob("*"))
                if path.is_file()
            }
            _write_json(
                temporary / "manifest.json",
                {
                    "schema_version": NODE_SCHEMA_VERSION,
                    "digest": digest,
                    "checksums": checksums,
                },
            )
            if destination.exists():
                raise FileExistsError(
                    f"Immutable artifact collision at {destination}."
                )
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        persisted = self._load(destination, expected_digest=digest)
        return ArtifactRef(stage, node_id, digest, destination, persisted, False)

    def _write_payload(
        self,
        *,
        temporary: Path,
        destination: Path,
        digest: str,
        payload: NodePayload,
        upstream: Sequence[ArtifactRef],
    ) -> None:
        descriptors: dict[str, dict[str, Any]] = {}
        allowed_sources = {item.digest for item in upstream}
        for item in upstream:
            allowed_sources.update(
                asset.source_digest
                for asset in item.payload.assets.values()
                if isinstance(asset, ArrayAsset)
            )

        for name, value in sorted(payload.assets.items()):
            if isinstance(value, np.ndarray):
                arrays = temporary / "arrays"
                arrays.mkdir(exist_ok=True)
                array_path = arrays / f"{name}.npy"
                np.save(array_path, value, allow_pickle=False)
                checksum = _sha256(array_path)
                descriptors[name] = {
                    "path": str(Path("arrays") / f"{name}.npy"),
                    "checksum": checksum,
                    "dtype": value.dtype.str,
                    "base_shape": list(value.shape),
                    "selector": [],
                    "source_digest": digest,
                }
                continue

            if value.source_digest not in allowed_sources:
                raise ArtifactIntegrityError(
                    f"Asset {name!r} references source digest "
                    f"{value.source_digest!r} outside this node's upstream graph."
                )
            self._validate_asset(value)
            descriptors[name] = value.descriptor(relative_to=destination)

        _write_json(
            temporary / "payload.json",
            {
                "schema_version": PAYLOAD_SCHEMA_VERSION,
                "metadata": dict(payload.metadata),
                "assets": descriptors,
            },
        )

    def record_failure(
        self, *, stage: Stage, node_id: str, error: BaseException
    ) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = (
            self.root
            / "failures"
            / self.plan.campaign.value
            / f"repetition={self.plan.repetition.index:03d}"
            / self.plan.manifest_digest[:16]
            / f"attempt={timestamp}-{uuid4().hex[:8]}"
        )
        destination.mkdir(parents=True, exist_ok=False)
        _write_json(
            destination / "status.json",
            {
                "status": "failed",
                "stage": Stage(stage).value,
                "node_id": node_id,
                "exception_type": type(error).__name__,
                "message": str(error),
                "timestamp": timestamp,
            },
        )
        (destination / "traceback.txt").write_text(
            "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
            encoding="utf-8",
        )
        return destination

    def _load(self, path: Path, *, expected_digest: str) -> NodePayload:
        manifest = _read_json(path / "manifest.json")
        if (
            manifest.get("schema_version") != NODE_SCHEMA_VERSION
            or manifest.get("digest") != expected_digest
        ):
            raise ArtifactIntegrityError(f"Artifact identity mismatch at {path}.")
        for filename, expected in manifest.get("checksums", {}).items():
            candidate = path / filename
            if self._checked_sha256(candidate) != expected:
                raise ArtifactIntegrityError(f"Checksum mismatch for {candidate}.")
        node = _read_json(path / "node.json")
        if node.get("digest") != expected_digest:
            raise ArtifactIntegrityError(f"Node digest mismatch at {path}.")

        raw = _read_json(path / "payload.json")
        if (
            not isinstance(raw, dict)
            or raw.get("schema_version") != PAYLOAD_SCHEMA_VERSION
            or not isinstance(raw.get("metadata"), dict)
            or not isinstance(raw.get("assets"), dict)
        ):
            raise ArtifactIntegrityError(f"Invalid node payload at {path}.")
        assets: dict[str, ArrayAsset] = {}
        for name, descriptor in raw["assets"].items():
            asset = self._asset_from_descriptor(path, descriptor)
            self._validate_asset(asset)
            assets[str(name)] = asset
        return NodePayload(metadata=raw["metadata"], assets=assets)

    def _asset_from_descriptor(
        self, node_path: Path, descriptor: Mapping[str, Any]
    ) -> ArrayAsset:
        relative = Path(str(descriptor.get("path", "")))
        if not str(relative) or relative.is_absolute():
            raise ArtifactIntegrityError("Array paths must be nonempty and relative.")
        path = (node_path / relative).resolve()
        run_root = self.run_root.resolve()
        if not path.is_relative_to(run_root):
            raise ArtifactIntegrityError(
                f"Array reference {path} escapes the experiment run root."
            )
        selector = tuple(
            ArraySlice.from_value(item) for item in descriptor.get("selector", ())
        )
        try:
            return ArrayAsset(
                path=path,
                checksum=str(descriptor["checksum"]),
                dtype=str(descriptor["dtype"]),
                base_shape=tuple(descriptor["base_shape"]),
                selector=selector,
                source_digest=str(descriptor["source_digest"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("Invalid array asset descriptor.") from exc

    def _validate_asset(self, asset: ArrayAsset) -> None:
        if self._checked_sha256(asset.path) != asset.checksum:
            raise ArtifactIntegrityError(
                f"Checksum mismatch for array asset {asset.path}."
            )
        try:
            base = np.load(asset.path, mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            raise ArtifactIntegrityError(
                f"Cannot load NumPy array asset {asset.path}."
            ) from exc
        if base.dtype.str != asset.dtype or tuple(base.shape) != asset.base_shape:
            raise ArtifactIntegrityError(
                f"Array dtype or shape mismatch for {asset.path}."
            )
        value = base[
            tuple(item.to_slice() for item in asset.selector)
            + (slice(None),) * (base.ndim - len(asset.selector))
        ]
        if tuple(value.shape) != asset.shape:
            raise ArtifactIntegrityError(
                f"Array selector mismatch for {asset.path}."
            )

        owner = asset.path.parent.parent
        node_file = owner / "node.json"
        if node_file.exists():
            source = _read_json(node_file).get("digest")
            if source != asset.source_digest:
                raise ArtifactIntegrityError(
                    f"Array source digest mismatch for {asset.path}."
                )

    def _checked_sha256(self, path: Path) -> str:
        try:
            stat = path.stat()
        except FileNotFoundError as exc:
            raise ArtifactIntegrityError(f"Missing artifact file {path}.") from exc
        cached = self._checksum_cache.get(path)
        key = (stat.st_size, stat.st_mtime_ns)
        if cached is not None and cached[:2] == key:
            return cached[2]
        checksum = _sha256(path)
        self._checksum_cache[path] = (key[0], key[1], checksum)
        return checksum

    @staticmethod
    def aggregate_comparisons(
        root: str | Path,
        *,
        campaign: str | None = None,
        study: str | None = None,
    ) -> list[dict[str, Any]]:
        root = Path(root)
        completed: set[tuple[str, int, str, str]] = set()
        finalize_pattern = "runs/*/repetition=*/*/finalize/*/*/node.json"
        for node_path in sorted(root.glob(finalize_pattern)):
            artifact_path = node_path.parent
            _validate_artifact_directory(artifact_path)
            node = _read_json(node_path)
            payload = _read_json(artifact_path / "payload.json")
            metadata = payload.get("metadata", {})
            if not isinstance(metadata, dict) or metadata.get("status") != "complete":
                continue
            for completed_study in metadata.get("studies", ()):
                completed.add(
                    (
                        str(node["campaign"]),
                        int(node["repetition"]),
                        str(node["manifest_digest"]),
                        str(completed_study),
                    )
                )

        rows: list[dict[str, Any]] = []
        pattern = "runs/*/repetition=*/*/analyze/*/*/node.json"
        for node_path in sorted(root.glob(pattern)):
            artifact_path = node_path.parent
            node = _read_json(node_path)
            spec = node.get("spec", {})
            if not isinstance(spec, dict) or "kind" not in spec or "evaluation_ids" not in spec:
                continue
            if campaign is not None and node.get("campaign") != campaign:
                continue
            if study is not None and spec.get("study_id") != study:
                continue
            completion_key = (
                str(node["campaign"]),
                int(node["repetition"]),
                str(node["manifest_digest"]),
                str(spec["study_id"]),
            )
            if completion_key not in completed:
                continue
            _validate_artifact_directory(artifact_path)
            payload = _read_json(artifact_path / "payload.json")
            rows.append(
                {
                    "campaign": node["campaign"],
                    "repetition": node["repetition"],
                    "study_id": spec["study_id"],
                    "comparison_id": node["node_id"],
                    "comparison_kind": spec["kind"],
                    "node_digest": node["digest"],
                    **dict(payload.get("metadata", {})),
                }
            )
        return rows

    @staticmethod
    def aggregate(root: str | Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        pattern = "runs/*/repetition=*/*/finalize/*/*/payload.json"
        for payload_path in sorted(Path(root).glob(pattern)):
            artifact_path = payload_path.parent
            _validate_artifact_directory(artifact_path)
            payload = _read_json(payload_path)
            rows.append(
                {
                    "path": str(artifact_path),
                    **dict(payload.get("metadata", {})),
                }
            )
        return rows


def _validate_artifact_directory(path: Path) -> None:
    manifest = _read_json(path / "manifest.json")
    for filename, expected in manifest.get("checksums", {}).items():
        candidate = path / filename
        if _sha256(candidate) != expected:
            raise ArtifactIntegrityError(f"Checksum mismatch for {candidate}.")


def _safe_node_name(node_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.=-]+", "__", node_id).strip("._") or "node"
    suffix = hashlib.sha256(node_id.encode("utf-8")).hexdigest()[:10]
    return f"{stem[:100]}--{suffix}"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(json_value(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(f"Invalid or missing JSON artifact {path}.") from exc
    if not isinstance(value, dict):
        raise ArtifactIntegrityError(f"Expected a JSON object at {path}.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
