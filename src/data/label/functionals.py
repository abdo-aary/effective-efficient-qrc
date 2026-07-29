from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np

from .base import BaseLabelFunctional, LabelNoise
from .context import TeacherContext


def _apply_nonlinearity(z: float, kind: Literal["none", "tanh", "sigmoid"]) -> float:
    if kind == "none":
        return float(z)
    if kind == "tanh":
        return float(np.tanh(z))
    if kind == "sigmoid":
        # stable sigmoid
        z = float(z)
        if z >= 0:
            return float(1.0 / (1.0 + np.exp(-z)))
        ez = float(np.exp(z))
        return float(ez / (1.0 + ez))
    raise ValueError(f"Unknown nonlinearity={kind!r}")


def _check_X_win(X_win: np.ndarray) -> Tuple[int, int]:
    if not isinstance(X_win, np.ndarray):
        raise TypeError("X_win must be a numpy array")
    if X_win.ndim != 2:
        raise ValueError(f"X_win must have shape (w,d); got {X_win.shape}")
    w, d = X_win.shape
    if w <= 0 or d <= 0:
        raise ValueError(f"X_win must have positive dimensions; got {X_win.shape}")
    return w, d


def _unit_norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 0:
        raise ValueError("Vector must have positive finite norm")
    return v / n


def _sample_unit_vector(rng: np.random.Generator, d: int) -> np.ndarray:
    for _ in range(100):
        g = rng.normal(size=d)
        n = float(np.linalg.norm(g))
        if n > 0:
            return (g / n).astype(float)
    raise RuntimeError("Could not sample a non-zero Gaussian vector")


def _sample_orthonormal_pair(rng: np.random.Generator, d: int) -> Tuple[np.ndarray, np.ndarray]:
    u = _sample_unit_vector(rng, d)
    for _ in range(200):
        v = _sample_unit_vector(rng, d)
        v = v - float(np.dot(u, v)) * u
        n = float(np.linalg.norm(v))
        if n > 1e-12:
            return u, (v / n).astype(float)
    raise RuntimeError("Could not sample an orthonormal pair")


def _sample_unit_vector_orthogonal_to(rng: np.random.Generator, u: np.ndarray) -> np.ndarray:
    """Sample a unit vector v such that <u,v> = 0 (up to numerical tolerance)."""
    u = _unit_norm(u)
    d = int(u.shape[0])
    for _ in range(300):
        v = rng.normal(size=d)
        # Gram–Schmidt
        v = v - float(np.dot(u, v)) * u
        n = float(np.linalg.norm(v))
        if np.isfinite(n) and n > 1e-12:
            return (v / n).astype(float)
    raise RuntimeError("Could not sample a unit vector orthogonal to u")


def _fading_projection(X_win: np.ndarray, vec: np.ndarray, alpha: float) -> float:
    r"""Window-truncated fading sum.

    Convention:
      - X_win[-1] is the most recent point (time t)
      - X_win[-1-k] corresponds to time t-k

    Computes  \sum_{k=0}^{w-1} alpha^k <vec, x_{t-k}>.
    """
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0,1)")
    w, d = _check_X_win(X_win)
    vec = np.asarray(vec, dtype=float).reshape(-1)
    if vec.shape[0] != d:
        raise ValueError(f"vec must have shape (d,), got {vec.shape} with d={d}")

    X_rev = X_win[::-1, :]  # index 0 is most recent
    weights = alpha ** np.arange(w, dtype=float)
    return float((X_rev @ vec) @ weights)


@dataclass(frozen=True)
class OneStepForecastFunctional(BaseLabelFunctional):
    """Forecast functional (Appendix E.2), in a windowed dataset convention.

    Using the common window convention (last element is the “next/current” point):
        y = <u, X_win[-1]>,
    with u on the unit sphere (sampled if not provided).

    IMPORTANT: if u is not provided, we sample it ONCE (at first call) and reuse it
    for all windows, to avoid pathological "different task per sample" labels.
    """

    u: Optional[np.ndarray] = None
    _u_cached: Optional[np.ndarray] = field(default=None, init=False, repr=False, compare=False)

    nonlinearity: Literal["none", "tanh", "sigmoid"] = "none"
    noise: LabelNoise = LabelNoise(0.0)

    def _resolve_u(self, d: int, rng: np.random.Generator) -> np.ndarray:
        if self.u is not None:
            u = np.asarray(self.u, dtype=float).reshape(-1)
            if u.shape[0] != d:
                raise ValueError(f"u must have shape (d,), got {u.shape} with d={d}")
            return _unit_norm(u)

        if self._u_cached is None:
            object.__setattr__(self, "_u_cached", _sample_unit_vector(rng, d))
        return np.asarray(self._u_cached, dtype=float).reshape(-1)

    def __call__(self, X_win: np.ndarray, rng: np.random.Generator) -> float:
        _, d = _check_X_win(X_win)

        u = self._resolve_u(d, rng)
        z = float(X_win[-1, :] @ u)
        y = _apply_nonlinearity(z, self.nonlinearity)
        return float(self.noise.add(y, rng))


@dataclass(frozen=True)
class ExpFadingLinearFunctional(BaseLabelFunctional):
    r"""Exponential fading linear functional (Appendix E.2).

    y = \sum_{k=0}^{w-1} alpha^k <u, x_{t-k}>, with x_t = X_win[-1].

    IMPORTANT: if u is not provided, we sample it ONCE (at first call) and reuse it
    for all windows, to avoid pathological "different task per sample" labels.
    """

    alpha: float = 0.8
    u: Optional[np.ndarray] = None
    _u_cached: Optional[np.ndarray] = field(default=None, init=False, repr=False, compare=False)

    nonlinearity: Literal["none", "tanh", "sigmoid"] = "none"
    noise: LabelNoise = LabelNoise(0.0)

    def _resolve_u(self, d: int, rng: np.random.Generator) -> np.ndarray:
        if self.u is not None:
            u = np.asarray(self.u, dtype=float).reshape(-1)
            if u.shape[0] != d:
                raise ValueError(f"u must have shape (d,), got {u.shape} with d={d}")
            return _unit_norm(u)

        if self._u_cached is None:
            object.__setattr__(self, "_u_cached", _sample_unit_vector(rng, d))
        return np.asarray(self._u_cached, dtype=float).reshape(-1)

    def __call__(self, X_win: np.ndarray, rng: np.random.Generator) -> float:
        _, d = _check_X_win(X_win)

        u = self._resolve_u(d, rng)
        z = _fading_projection(X_win, u, self.alpha)
        y = _apply_nonlinearity(z, self.nonlinearity)
        return float(self.noise.add(y, rng))


@dataclass(frozen=True)
class VolteraFunctional(BaseLabelFunctional):
    r"""Second-order (Volterra-style) functional (Appendix E.2).

    Let
      L_u = \sum_{k=0}^{w-1} alpha^k <u, x_{t-k}>,   L_v likewise.

    Then
      y = L_u + 0.5 * (L_v)^2

    If v is provided and orthogonalize_v=True, we Gram–Schmidt v against u
    and then normalize.

    IMPORTANT:
      - If u and/or v are not provided, we sample them ONCE (at first call) and reuse
        them for all windows.
      - If v is not provided and orthogonalize_v=True, we sample v orthogonal to the
        *actual* u used (not an unrelated random u).
    """

    alpha: float = 0.8
    u: Optional[np.ndarray] = None
    v: Optional[np.ndarray] = None
    _u_cached: Optional[np.ndarray] = field(default=None, init=False, repr=False, compare=False)
    _v_cached: Optional[np.ndarray] = field(default=None, init=False, repr=False, compare=False)

    orthogonalize_v: bool = True
    nonlinearity: Literal["none", "tanh", "sigmoid"] = "none"
    noise: LabelNoise = LabelNoise(0.0)

    def _resolve_u(self, d: int, rng: np.random.Generator) -> np.ndarray:
        if self.u is not None:
            u = np.asarray(self.u, dtype=float).reshape(-1)
            if u.shape[0] != d:
                raise ValueError(f"u must have shape (d,), got {u.shape} with d={d}")
            return _unit_norm(u)

        if self._u_cached is None:
            object.__setattr__(self, "_u_cached", _sample_unit_vector(rng, d))
        return np.asarray(self._u_cached, dtype=float).reshape(-1)

    def _resolve_v(self, u: np.ndarray, d: int, rng: np.random.Generator) -> np.ndarray:
        if self.v is not None:
            v = np.asarray(self.v, dtype=float).reshape(-1)
            if v.shape[0] != d:
                raise ValueError(f"v must have shape (d,), got {v.shape} with d={d}")
            if self.orthogonalize_v:
                v = v - float(np.dot(u, v)) * u
            return _unit_norm(v)

        if self._v_cached is None:
            if self.orthogonalize_v:
                v = _sample_unit_vector_orthogonal_to(rng, u)
            else:
                v = _sample_unit_vector(rng, d)
            object.__setattr__(self, "_v_cached", v)
        return np.asarray(self._v_cached, dtype=float).reshape(-1)

    def __call__(self, X_win: np.ndarray, rng: np.random.Generator) -> float:
        _, d = _check_X_win(X_win)

        u = self._resolve_u(d, rng)
        v = self._resolve_v(u, d, rng)

        Lu = _fading_projection(X_win, u, self.alpha)
        Lv = _fading_projection(X_win, v, self.alpha)

        z = float(Lu + 0.5 * (Lv**2))
        y = _apply_nonlinearity(z, self.nonlinearity)
        return float(self.noise.add(y, rng))


# Theory-aligned, RNG-free protocol functionals used by E1. The legacy
# functionals above remain solely for stored-configuration compatibility.
U3 = np.asarray((1.0, -1.0, 1.0), dtype=float) / 3.0
V3 = np.asarray((1.0, 1.0, -1.0), dtype=float) / 3.0


def _protocol_direction(value: np.ndarray, d: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.shape != (d,):
        raise ValueError(f"{name} must have shape {(d,)}, got {vector.shape}.")
    if np.linalg.norm(vector, ord=1) > 1.0 + 1e-15:
        raise ValueError(f"{name} must have l1 norm at most one.")
    return vector


@dataclass(frozen=True)
class OneStepFutureFunctional:
    """Genuine one-step target using data strictly after the input window."""

    u: np.ndarray = field(default_factory=lambda: U3.copy())
    name: str = "future"

    def evaluate(
        self, X_win: np.ndarray, *, index: int, context: TeacherContext
    ) -> float:
        _, d = _check_X_win(X_win)
        u = _protocol_direction(self.u, d, "u")
        if not 0 <= int(index) < context.future_observations.shape[0]:
            raise IndexError("Teacher-context index is outside the dataset.")
        return float(context.future_observations[int(index)] @ u)


@dataclass(frozen=True)
class NormalizedExpMemoryFunctional:
    gamma: float = 0.8
    u: np.ndarray = field(default_factory=lambda: U3.copy())
    name: str = "exp_memory"

    def evaluate(
        self, X_win: np.ndarray, *, index: int, context: TeacherContext
    ) -> float:
        del index, context
        w, d = _check_X_win(X_win)
        if not 0.0 < float(self.gamma) < 1.0:
            raise ValueError("gamma must lie in (0,1).")
        u = _protocol_direction(self.u, d, "u")
        weights = float(self.gamma) ** np.arange(w, dtype=float)
        return float((X_win[::-1] @ u) @ weights / np.sum(weights))


@dataclass(frozen=True)
class VolterraFunctional:
    gamma: float = 0.8
    u: np.ndarray = field(default_factory=lambda: U3.copy())
    v: np.ndarray = field(default_factory=lambda: V3.copy())
    name: str = "volterra"

    def evaluate(
        self, X_win: np.ndarray, *, index: int, context: TeacherContext
    ) -> float:
        del index, context
        w, d = _check_X_win(X_win)
        if not 0.0 < float(self.gamma) < 1.0:
            raise ValueError("gamma must lie in (0,1).")
        u = _protocol_direction(self.u, d, "u")
        v = _protocol_direction(self.v, d, "v")
        weights = float(self.gamma) ** np.arange(w, dtype=float)
        A_w = float(np.sum(weights))
        reverse = X_win[::-1]
        L_u = float((reverse @ u) @ weights)
        L_v = float((reverse @ v) @ weights)
        return float((L_u + 0.5 * L_v * L_v) / (A_w + 0.5 * A_w * A_w))


@dataclass(frozen=True)
class DelayedRecallFunctional:
    delay: int
    u: np.ndarray = field(default_factory=lambda: U3.copy())
    name: str | None = None

    def __post_init__(self) -> None:
        delay = int(self.delay)
        if delay < 0:
            raise ValueError("delay must be nonnegative.")
        object.__setattr__(self, "delay", delay)
        object.__setattr__(self, "name", self.name or f"delay_{delay}")

    def evaluate(
        self, X_win: np.ndarray, *, index: int, context: TeacherContext
    ) -> float:
        del index, context
        w, d = _check_X_win(X_win)
        if self.delay >= w:
            raise ValueError(f"delay={self.delay} requires window length > delay.")
        u = _protocol_direction(self.u, d, "u")
        return float(X_win[-1 - self.delay] @ u)


@dataclass(frozen=True)
class SparseCrossLagFunctional:
    delay_u: int = 0
    delay_v: int = 15
    u: np.ndarray = field(default_factory=lambda: U3.copy())
    v: np.ndarray = field(default_factory=lambda: V3.copy())
    name: str = "cross_0_15"

    def evaluate(
        self, X_win: np.ndarray, *, index: int, context: TeacherContext
    ) -> float:
        del index, context
        w, d = _check_X_win(X_win)
        tau_u, tau_v = int(self.delay_u), int(self.delay_v)
        if min(tau_u, tau_v) < 0 or max(tau_u, tau_v) >= w:
            raise ValueError("Cross-lag delays must lie inside the input window.")
        u = _protocol_direction(self.u, d, "u")
        v = _protocol_direction(self.v, d, "v")
        return float((X_win[-1 - tau_u] @ u) * (X_win[-1 - tau_v] @ v))


def e1_functionals() -> tuple[object, ...]:
    """Return the complete E1 task suite in canonical artifact order."""

    return (
        OneStepFutureFunctional(),
        NormalizedExpMemoryFunctional(),
        VolterraFunctional(),
        DelayedRecallFunctional(1),
        DelayedRecallFunctional(5),
        DelayedRecallFunctional(10),
        DelayedRecallFunctional(20),
        SparseCrossLagFunctional(),
    )
