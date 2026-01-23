from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ..label.base import BaseLabelFunctional


@dataclass
class WindowsDataset:
    """Returned dataset container."""

    X: np.ndarray  # (N,w,d)
    y: np.ndarray  # (L,N) with L being the different functionals used to label.

    label_functionals: List[BaseLabelFunctional]
    meta: Dict[str, object]


class BaseGenerator(ABC):
    """Abstract dataset generator that returns windows and labels.

    The goal is to generate X of shape (N,w,d) and y of shape (N,).
    """

    def __init__(self, *, seed: int = 0):
        self.seed = int(seed)

    @abstractmethod
    def make_windows_dataset(
        self,
        *,
        N: int,
        w: int,
        d: int,
        label_functionals: List[BaseLabelFunctional],
    ) -> WindowsDataset:
        """Generate a dataset of N windows of length w, dimension d, plus labels of shape (L, N,) where L is
        the number of chosen functionals. This is produced to test on using multiple labels on the same input process
        for fast experiments."""
        raise NotImplementedError


def ensure_bounded(X: np.ndarray, *, mode: str = "tanh", clip: float = 0.999) -> np.ndarray:
    """Ensure values are in (-1,1) to match typical bounded-input assumptions.

    mode:
      - "tanh": apply tanh
      - "clip": clip to [-clip, clip]
      - "none": no-op
    """
    if mode == "none":
        return X
    if mode == "tanh":
        return np.tanh(X)
    if mode == "clip":
        return np.clip(X, -float(clip), float(clip))
    raise ValueError(f"Unknown bounded mode={mode!r}")
