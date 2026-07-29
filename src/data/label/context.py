"""Immutable teacher context for protocol temporal functionals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _readonly(value: object, *, dtype: object, ndim: int, name: str) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {array.shape}.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class TeacherContext:
    """Per-window teacher information that is never exposed to the learner."""

    prediction_origins: np.ndarray
    future_indices: np.ndarray
    future_observations: np.ndarray
    protocol_version: str = "quark-e1-functionals/v1"

    def __post_init__(self) -> None:
        origins = _readonly(
            self.prediction_origins, dtype=np.int64, ndim=1, name="prediction_origins"
        )
        future_indices = _readonly(
            self.future_indices, dtype=np.int64, ndim=1, name="future_indices"
        )
        future = _readonly(
            self.future_observations,
            dtype=np.float64,
            ndim=2,
            name="future_observations",
        )
        if origins.shape != future_indices.shape or origins.shape[0] != future.shape[0]:
            raise ValueError("Teacher-context arrays must share the example axis.")
        if not np.array_equal(future_indices, origins + 1):
            raise ValueError("Future indices must be exactly one step after origins.")
        object.__setattr__(self, "prediction_origins", origins)
        object.__setattr__(self, "future_indices", future_indices)
        object.__setattr__(self, "future_observations", future)
