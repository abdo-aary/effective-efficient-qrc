"""Order-independent hierarchical random streams for experiment repetitions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


SEED_TREE_SCHEMA_VERSION = "quark.path-seeds/v1"


def _normalize_path(path: str) -> str:
    parts = tuple(part.strip() for part in str(path).split("/"))
    if not parts or any(not part or part in {".", ".."} for part in parts):
        raise ValueError(f"Invalid seed path {path!r}.")
    return "/".join(parts)


@dataclass(frozen=True)
class PathSeedTree:
    """Derive a NumPy stream directly from ``(root_entropy, semantic_path)``.

    Unlike a spawn-order registry, adding or requesting another path cannot shift
    any existing stream.
    """

    root_entropy: tuple[int, ...]

    @classmethod
    def from_root(cls, entropy: int | tuple[int, ...]) -> "PathSeedTree":
        values = (int(entropy),) if isinstance(entropy, (int, np.integer)) else tuple(int(item) for item in entropy)
        if not values or any(item < 0 for item in values):
            raise ValueError("Seed entropy must contain nonnegative integers.")
        return cls(values)

    def sequence(self, path: str) -> np.random.SeedSequence:
        normalized = _normalize_path(path)
        payload = json.dumps(
            {"schema": SEED_TREE_SCHEMA_VERSION, "root": self.root_entropy, "path": normalized},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        words = np.frombuffer(hashlib.sha256(payload).digest(), dtype=np.uint32)
        entropy = tuple(self.root_entropy) + tuple(int(word) for word in words)
        return np.random.SeedSequence(entropy)

    def generator(self, path: str) -> np.random.Generator:
        return np.random.default_rng(self.sequence(path))

    def integer(self, path: str) -> int:
        return int(self.sequence(path).generate_state(1, dtype=np.uint32)[0])

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": SEED_TREE_SCHEMA_VERSION, "root_entropy": list(self.root_entropy)}

