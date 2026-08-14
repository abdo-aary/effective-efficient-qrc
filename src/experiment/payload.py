"""Typed metadata and immutable NumPy assets for experiment nodes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .domain import content_digest, json_value


PAYLOAD_SCHEMA_VERSION = "quark.node-payload/v1"


@dataclass(frozen=True)
class ArraySlice:
    start: int | None = None
    stop: int | None = None
    step: int | None = None

    def __post_init__(self) -> None:
        if self.step is not None and int(self.step) <= 0:
            raise ValueError("Array selectors support only positive slice steps.")
        for name in ("start", "stop", "step"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, int(value))

    def to_slice(self) -> slice:
        return slice(self.start, self.stop, self.step)

    def to_dict(self) -> dict[str, int | None]:
        return {"start": self.start, "stop": self.stop, "step": self.step}

    @classmethod
    def from_value(cls, value: slice | Mapping[str, Any] | "ArraySlice") -> "ArraySlice":
        if isinstance(value, cls):
            return value
        if isinstance(value, slice):
            return cls(value.start, value.stop, value.step)
        return cls(value.get("start"), value.get("stop"), value.get("step"))


@dataclass(frozen=True)
class ArrayAsset:
    """A checksum-bound view of one immutable uncompressed NumPy array."""

    path: Path
    checksum: str
    dtype: str
    base_shape: tuple[int, ...]
    selector: tuple[ArraySlice, ...] = ()
    source_digest: str = ""

    def __post_init__(self) -> None:
        path = Path(self.path)
        shape = tuple(int(item) for item in self.base_shape)
        if not shape or any(item < 0 for item in shape):
            raise ValueError("Array assets require a nonempty nonnegative base shape.")
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as exc:
            raise ValueError(f"Invalid array dtype {self.dtype!r}.") from exc
        if dtype.hasobject:
            raise ValueError("Object arrays are forbidden.")
        selector = tuple(ArraySlice.from_value(item) for item in self.selector)
        if len(selector) > len(shape):
            raise ValueError("Array selector rank exceeds the base array rank.")
        _selector_shape(shape, selector)
        if len(str(self.checksum)) != 64:
            raise ValueError("Array assets require a SHA-256 checksum.")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "dtype", dtype.str)
        object.__setattr__(self, "base_shape", shape)
        object.__setattr__(self, "selector", selector)
        object.__setattr__(self, "checksum", str(self.checksum))
        object.__setattr__(self, "source_digest", str(self.source_digest))

    @property
    def shape(self) -> tuple[int, ...]:
        return _selector_shape(self.base_shape, self.selector)

    def load(self, *, validate_checksum: bool = True) -> np.ndarray:
        if validate_checksum and _sha256(self.path) != self.checksum:
            raise ValueError(f"Checksum mismatch for array asset {self.path}.")
        base = np.load(self.path, mmap_mode="r", allow_pickle=False)
        if base.dtype.str != self.dtype or tuple(base.shape) != self.base_shape:
            raise ValueError(f"Array header mismatch for {self.path}.")
        value = base[_selector_tuple(self.base_shape, self.selector)]
        value.setflags(write=False)
        return value

    def view(self, selector: Sequence[slice | ArraySlice]) -> "ArrayAsset":
        composed = _compose_selectors(self.base_shape, self.selector, selector)
        return ArrayAsset(
            path=self.path,
            checksum=self.checksum,
            dtype=self.dtype,
            base_shape=self.base_shape,
            selector=composed,
            source_digest=self.source_digest,
        )

    def descriptor(self, *, relative_to: Path) -> dict[str, Any]:
        return {
            "path": str(self.path.resolve().relative_to(relative_to.resolve()))
            if self.path.resolve().is_relative_to(relative_to.resolve())
            else _relative_path(self.path.resolve(), relative_to.resolve()),
            "checksum": self.checksum,
            "dtype": self.dtype,
            "base_shape": list(self.base_shape),
            "selector": [item.to_dict() for item in self.selector],
            "source_digest": self.source_digest,
        }


@dataclass(frozen=True)
class NodePayload:
    """JSON metadata plus named owned or referenced array assets."""

    metadata: Mapping[str, Any]
    assets: Mapping[str, np.ndarray | ArrayAsset] = MappingProxyType({})

    def __post_init__(self) -> None:
        metadata = json_value(dict(self.metadata))
        if not isinstance(metadata, dict):
            raise TypeError("Node payload metadata must be a mapping.")
        assets = dict(self.assets)
        if len(assets) != len(set(assets)):
            raise ValueError("Node payload asset names must be unique.")
        for name, value in assets.items():
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name.startswith("."):
                raise ValueError(f"Invalid array asset name {name!r}.")
            if not isinstance(value, (np.ndarray, ArrayAsset)):
                raise TypeError(f"Asset {name!r} must be a NumPy array or ArrayAsset.")
            if isinstance(value, np.ndarray) and value.dtype.hasobject:
                raise ValueError("Object arrays are forbidden.")
        object.__setattr__(self, "metadata", MappingProxyType(metadata))
        object.__setattr__(self, "assets", MappingProxyType(assets))

    @classmethod
    def coerce(cls, value: "NodePayload | Mapping[str, Any]") -> "NodePayload":
        return value if isinstance(value, cls) else cls(metadata=value)

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.metadata[key]

    def require_asset(self, name: str) -> ArrayAsset:
        try:
            asset = self.assets[name]
        except KeyError as exc:
            raise KeyError(f"Payload has no array asset {name!r}.") from exc
        if not isinstance(asset, ArrayAsset):
            raise TypeError(f"Asset {name!r} has not been persisted yet.")
        return asset

    def semantic_value(self) -> dict[str, Any]:
        assets: dict[str, Any] = {}
        for name, asset in self.assets.items():
            if isinstance(asset, ArrayAsset):
                assets[name] = {
                    "checksum": asset.checksum,
                    "dtype": asset.dtype,
                    "base_shape": asset.base_shape,
                    "selector": tuple(
                        (item.start, item.stop, item.step) for item in asset.selector
                    ),
                    "source_digest": asset.source_digest,
                }
            else:
                array = np.asarray(asset)
                assets[name] = {
                    "checksum": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                    "dtype": array.dtype.str,
                    "shape": array.shape,
                }
        return {"metadata": dict(self.metadata), "assets": assets}

    @property
    def digest(self) -> str:
        return content_digest(self.semantic_value())


def _selector_tuple(
    shape: tuple[int, ...],
    selector: Sequence[ArraySlice],
) -> tuple[slice, ...]:
    return tuple(item.to_slice() for item in selector) + (slice(None),) * (
        len(shape) - len(selector)
    )


def _selector_shape(
    shape: tuple[int, ...],
    selector: Sequence[ArraySlice],
) -> tuple[int, ...]:
    result = []
    for length, item in zip(shape, _selector_tuple(shape, selector)):
        start, stop, step = item.indices(length)
        result.append(len(range(start, stop, step)))
    return tuple(result)


def _compose_selectors(
    shape: tuple[int, ...],
    existing: Sequence[ArraySlice],
    requested: Sequence[slice | ArraySlice],
) -> tuple[ArraySlice, ...]:
    if len(requested) > len(shape):
        raise ValueError("Array selector rank exceeds the array rank.")
    current = _selector_tuple(shape, existing)
    requested_full = tuple(ArraySlice.from_value(item).to_slice() for item in requested)
    requested_full += (slice(None),) * (len(shape) - len(requested_full))
    composed = []
    for length, outer, inner in zip(shape, current, requested_full):
        outer_range = range(*outer.indices(length))
        selected = outer_range[inner]
        if not isinstance(selected, range):
            raise ValueError("Integer selectors are not supported.")
        composed.append(ArraySlice(selected.start, selected.stop, selected.step))
    return tuple(composed)


def _relative_path(path: Path, start: Path) -> str:
    import os

    return os.path.relpath(path, start=start)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
