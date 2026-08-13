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

from .domain import ExperimentPlan, Stage, content_digest, json_value


NODE_SCHEMA_VERSION = "quark.experiment-node/v1"


class ArtifactIntegrityError(ValueError):
    pass


@dataclass(frozen=True)
class ArtifactRef:
    stage: Stage
    node_id: str
    digest: str
    path: Path
    payload: Mapping[str, Any]
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
        producer: Callable[[], Mapping[str, Any]],
    ) -> ArtifactRef:
        ...

    def record_failure(self, *, stage: Stage, node_id: str, error: BaseException) -> Path:
        ...


class LocalArtifactStore:
    def __init__(self, root: str | Path, plan: ExperimentPlan) -> None:
        self.root = Path(root)
        self.plan = plan

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
        producer: Callable[[], Mapping[str, Any]],
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
            "upstream": [{"node_id": item.node_id, "digest": item.digest} for item in upstream],
        }
        digest = content_digest(identity)
        destination = self.run_root / stage.value / _safe_node_name(node_id) / digest[:16]
        if destination.exists():
            payload = self._load(destination, expected_digest=digest)
            return ArtifactRef(stage, node_id, digest, destination, payload, True)

        payload = dict(producer())
        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{digest[:12]}.", dir=str(parent)))
        try:
            _write_json(temporary / "node.json", {**identity, "digest": digest})
            _write_json(temporary / "payload.json", payload)
            checksums = {
                path.name: _sha256(path)
                for path in sorted(temporary.iterdir())
                if path.is_file()
            }
            _write_json(
                temporary / "manifest.json",
                {"schema_version": NODE_SCHEMA_VERSION, "digest": digest, "checksums": checksums},
            )
            if destination.exists():
                raise FileExistsError(f"Immutable artifact collision at {destination}.")
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return ArtifactRef(stage, node_id, digest, destination, payload, False)

    def record_failure(self, *, stage: Stage, node_id: str, error: BaseException) -> Path:
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
            "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            encoding="utf-8",
        )
        return destination

    def _load(self, path: Path, *, expected_digest: str) -> dict[str, Any]:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != NODE_SCHEMA_VERSION or manifest.get("digest") != expected_digest:
            raise ArtifactIntegrityError(f"Artifact identity mismatch at {path}.")
        for filename, expected in manifest.get("checksums", {}).items():
            if _sha256(path / filename) != expected:
                raise ArtifactIntegrityError(f"Checksum mismatch for {path / filename}.")
        node = json.loads((path / "node.json").read_text(encoding="utf-8"))
        if node.get("digest") != expected_digest:
            raise ArtifactIntegrityError(f"Node digest mismatch at {path}.")
        payload = json.loads((path / "payload.json").read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ArtifactIntegrityError("Node payload must be a JSON object.")
        return payload

    @staticmethod
    def aggregate(root: str | Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(Path(root).glob("runs/*/repetition=*/*/finalize/*/*/payload.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append({"path": str(path.parent), **payload})
        return rows


def _safe_node_name(node_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.=-]+", "__", node_id).strip("._") or "node"
    suffix = hashlib.sha256(node_id.encode("utf-8")).hexdigest()[:10]
    return f"{stem[:100]}--{suffix}"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(json_value(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

