from __future__ import annotations

from abc import abstractmethod
from typing import Dict, List

import numpy as np

from .base import BaseGenerator, WindowsDataset, ensure_bounded
from ..label.base import BaseLabelFunctional


class BetaMixingGenerator(BaseGenerator):
    """Generate windows from a single beta-mixing process realization.

    Theoretical motivation: windows are sampled with a gap s between them,
    so that dependence between windows decays with beta(s) (as in the paper).

    Concretely we:
      1) simulate a path (x_t) of length T
      2) take N windows of length w with stride (w+s)

    Window i uses indices:
        [t_i - w + 1, ..., t_i],  with t_i = burn_in + w-1 + i*(w+s)

    Parameters
    ----------
    s : int
        between-window spacing (gap) in *time steps*
    burn_in : int
        number of initial samples to discard to reduce sensitivity to x_0
    bounded : str
        ensure values lie in (-1,1): 'tanh' | 'clip' | 'none'
    """

    def __init__(
            self,
            *,
            s: int,
            burn_in: int = 200,
            bounded: str = "tanh",
            clip: float = 0.999,
            seed: int = 0,
    ):
        super().__init__(seed=seed)
        self.s = int(s)
        self.burn_in = int(burn_in)
        self.bounded = str(bounded)
        self.clip = float(clip)

    @abstractmethod
    def simulate_path(self, *, T: int, d: int, rng: np.random.Generator) -> np.ndarray:
        """Return a process realization of shape (T,d)."""
        raise NotImplementedError

    def make_windows_dataset(
            self,
            *,
            N: int,
            w: int,
            d: int,
            label_functionals: List[BaseLabelFunctional],
    ) -> WindowsDataset:
        if N <= 0 or w <= 0 or d <= 0:
            raise ValueError("N,w,d must all be positive.")
        if self.s < 0:
            raise ValueError("s must be >= 0.")
        if self.burn_in < 0:
            raise ValueError("burn_in must be >= 0.")

        rng = np.random.default_rng(self.seed)

        stride = w + self.s
        # last index used is burn_in + w-1 + (N-1)*stride, so total length is that + 1
        T = self.burn_in + w + (N - 1) * stride

        path = self.simulate_path(T=T, d=d, rng=rng)
        if path.shape != (T, d):
            raise ValueError(f"simulate_path must return (T,d)={(T, d)}, got {path.shape}.")

        path = ensure_bounded(path, mode=self.bounded, clip=self.clip)

        X = np.empty((N, w, d), dtype=float)

        L = len(label_functionals)
        y = np.empty((L, N,), dtype=float)

        for i in range(N):
            end = self.burn_in + (w - 1) + i * stride
            start = end - w + 1
            X_win = path[start: end + 1]
            X[i] = X_win
            for l, label_functional in enumerate(label_functionals):
                y[l, i] = float(label_functional(X_win, rng=rng))

        meta: Dict[str, object] = {
            "seed": self.seed,
            "burn_in": self.burn_in,
            "s": self.s,
            "stride": stride,
            "bounded": self.bounded,
        }
        return WindowsDataset(X=X, y=y, meta=meta, label_functionals=label_functionals)
