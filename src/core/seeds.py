"""Named hierarchical random streams."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_STREAM_NAMES_V1 = (
    "dataset_generation",
    "dataset_split",
    "jl_projection",
    "reservoir_parameters",
    "observable_selection",
    "reset_trajectories",
    "shadow_bases",
    "measurement_outcomes",
    "classical_baseline",
    "model_selection",
    "replicate",
)
_STREAM_NAMES = _STREAM_NAMES_V1 + ("task_functionals",)
SEED_BUNDLE_SCHEMA_VERSION = "quark.seed-bundle/v2"


@dataclass(frozen=True)
class SeedBundle:
    root_entropy: tuple[int, ...]
    spawn_keys: tuple[tuple[str, tuple[int, ...]], ...]

    @classmethod
    def from_root(cls, entropy: int | tuple[int, ...]) -> "SeedBundle":
        entropy_tuple = (int(entropy),) if isinstance(entropy, (int, np.integer)) else tuple(entropy)
        root = np.random.SeedSequence(entropy_tuple)
        children = root.spawn(len(_STREAM_NAMES))
        return cls(
            root_entropy=entropy_tuple,
            spawn_keys=tuple(
                (name, tuple(int(value) for value in child.spawn_key))
                for name, child in zip(_STREAM_NAMES, children)
            ),
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.spawn_keys)

    def sequence(self, name: str) -> np.random.SeedSequence:
        lookup = dict(self.spawn_keys)
        if name not in lookup:
            raise KeyError(f"Unknown seed stream {name!r}; known streams: {sorted(lookup)}.")
        return np.random.SeedSequence(self.root_entropy, spawn_key=lookup[name])

    def generator(self, name: str) -> np.random.Generator:
        return np.random.default_rng(self.sequence(name))

    def integer(self, name: str) -> int:
        return int(self.sequence(name).generate_state(1, dtype=np.uint32)[0])

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SEED_BUNDLE_SCHEMA_VERSION,
            "root_entropy": list(self.root_entropy),
            "spawn_keys": {
                name: list(spawn_key) for name, spawn_key in self.spawn_keys
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SeedBundle":
        root = tuple(int(value) for value in payload["root_entropy"])  # type: ignore[index]
        raw = payload["spawn_keys"]  # type: ignore[index]
        if not isinstance(raw, dict):
            raise TypeError("spawn_keys must be an object.")
        unknown = set(raw) - set(_STREAM_NAMES)
        if unknown:
            raise ValueError(f"Unknown serialized seed streams: {sorted(unknown)}.")
        canonical_children = np.random.SeedSequence(root).spawn(len(_STREAM_NAMES))
        spawn_keys = tuple(
            (
                name,
                tuple(int(value) for value in raw[name])
                if name in raw
                else tuple(int(value) for value in canonical_children[index].spawn_key),
            )
            for index, name in enumerate(_STREAM_NAMES)
        )
        return cls(root_entropy=root, spawn_keys=spawn_keys)
