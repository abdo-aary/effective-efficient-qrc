"""Small NumPy/CuPy backend selection helpers.

The project stores artifacts as NumPy arrays, but some rebuttal experiments use
CuPy internally for Gram matrices, readouts, and QRC state evolution.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any

import numpy as np


def import_cupy() -> Any:
    """Import CuPy with a clear project-level error message."""

    try:
        import cupy as cp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "CuPy is required for backend='cupy'. Install it with "
            "`conda run -n shaqr python -m pip install cupy-cuda12x`."
        ) from exc
    return cp


def cupy_available() -> bool:
    """Return True when CuPy is importable and at least one CUDA device is visible."""

    try:
        cp = import_cupy()
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def resolve_backend(backend: str = "auto") -> str:
    """Resolve ``auto`` to ``cupy`` when available, otherwise ``numpy``."""

    backend = str(backend)
    if backend == "auto":
        return "cupy" if cupy_available() else "numpy"
    if backend not in {"numpy", "cupy"}:
        raise ValueError(f"backend must be one of {{'auto','numpy','cupy'}}, got {backend!r}.")
    if backend == "cupy":
        import_cupy()
    return backend


def get_array_module(x: Any = None, *, backend: str = "auto") -> Any:
    """Return ``numpy`` or ``cupy`` for ``x`` and requested backend."""

    resolved = resolve_backend(backend)
    if resolved == "cupy":
        cp = import_cupy()
        if x is not None and isinstance(x, np.ndarray) and backend == "auto":
            return np
        return cp
    return np


def asarray(x: Any, *, backend: str = "auto", dtype: Any | None = None, device: int | None = None) -> Any:
    """Convert ``x`` to the requested backend array type."""

    resolved = resolve_backend(backend)
    if resolved == "numpy":
        return np.asarray(asnumpy(x), dtype=dtype)
    cp = import_cupy()
    with device_context(device):
        return cp.asarray(x, dtype=dtype)


def asnumpy(x: Any) -> np.ndarray:
    """Convert NumPy/CuPy arrays to NumPy."""

    try:
        cp = import_cupy()
    except ImportError:
        cp = None
    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


@contextmanager
def device_context(device: int | None):
    """Context manager selecting a CUDA device when CuPy is in use."""

    if device is None:
        with nullcontext():
            yield
        return
    cp = import_cupy()
    with cp.cuda.Device(int(device)):
        yield


def is_cupy_array(x: Any) -> bool:
    """Return True if ``x`` is a CuPy ndarray."""

    try:
        cp = import_cupy()
    except ImportError:
        return False
    return isinstance(x, cp.ndarray)
