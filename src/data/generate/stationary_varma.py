"""Exactly stationary Gaussian VARMA data for the E1 protocol."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import brentq

from src.core.seeds import SeedBundle
from src.data.label.context import TeacherContext
from src.data.label.functionals import e1_functionals


def _readonly(value: object, *, dtype: object, ndim: int, name: str) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {array.shape}.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class StationaryVARMASpec:
    d: int = 3
    p: int = 3
    q: int = 3
    target_companion_radius: float = 0.7
    ma_amplitude: float = 0.5
    ma_decay: float = 0.5
    observation_scale: float = 1.0

    def __post_init__(self) -> None:
        if min(int(self.d), int(self.p), int(self.q)) < 1:
            raise ValueError("d, p, and q must be positive.")
        if not 0.0 < float(self.target_companion_radius) < 1.0:
            raise ValueError("target_companion_radius must lie in (0,1).")
        if float(self.ma_amplitude) < 0.0:
            raise ValueError("ma_amplitude must be nonnegative.")
        if not 0.0 < float(self.ma_decay) < 1.0:
            raise ValueError("ma_decay must lie in (0,1).")
        if float(self.observation_scale) <= 0.0:
            raise ValueError("observation_scale must be positive.")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "StationaryVARMASpec":
        fixed = {
            "process": "stationary_varma",
            "innovation_covariance": "identity",
            "ar_draw": "gaussian_unit_spectral_weighted",
            "stationary_initialization": "lyapunov",
            "burn_in": 0,
        }
        for name, expected in fixed.items():
            if value.get(name) != expected:
                raise ValueError(
                    f"E1 requires data.{name}={expected!r}, got {value.get(name)!r}."
                )
        return cls(
            d=int(value["d"]),
            p=int(value["p"]),
            q=int(value["q"]),
            target_companion_radius=float(value["target_companion_radius"]),
            ma_amplitude=float(value["ma_amplitude"]),
            ma_decay=float(value["ma_decay"]),
            observation_scale=float(value["observation_scale"]),
        )


@dataclass(frozen=True)
class E1Dataset:
    windows: np.ndarray
    targets: np.ndarray
    task_names: tuple[str, ...]
    context: TeacherContext
    prediction_origins: np.ndarray
    phi: np.ndarray
    theta: np.ndarray
    transition: np.ndarray
    noise_map: np.ndarray
    stationary_covariance: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        windows = _readonly(self.windows, dtype=np.float64, ndim=3, name="windows")
        targets = _readonly(self.targets, dtype=np.float64, ndim=2, name="targets")
        origins = _readonly(
            self.prediction_origins, dtype=np.int64, ndim=1, name="prediction_origins"
        )
        if windows.shape[0] != targets.shape[0] or windows.shape[0] != origins.shape[0]:
            raise ValueError("Dataset arrays must share the example axis.")
        if targets.shape[1] != len(self.task_names):
            raise ValueError("Target columns must match task_names.")
        if not np.all(np.isfinite(windows)) or not np.all(np.isfinite(targets)):
            raise ValueError("E1 dataset contains nonfinite values.")
        if np.max(np.abs(windows)) > 1.0 + 1e-12:
            raise ValueError("Bounded E1 inputs must lie in [-1,1].")
        if np.max(np.abs(targets)) > 1.0 + 1e-12:
            raise ValueError("E1 targets must lie in [-1,1].")
        object.__setattr__(self, "windows", windows)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "prediction_origins", origins)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for array in (
            self.windows,
            self.targets,
            self.prediction_origins,
            self.phi,
            self.theta,
            self.transition,
            self.noise_map,
            self.stationary_covariance,
        ):
            value = np.asarray(array)
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(value.shape).encode("ascii"))
            digest.update(value.tobytes(order="C"))
        digest.update(
            json.dumps(self.metadata, sort_keys=True, separators=(",", ":")).encode()
        )
        return digest.hexdigest()


def companion_matrix(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 3 or phi.shape[1] != phi.shape[2]:
        raise ValueError("phi must have shape (p,d,d).")
    p, d, _ = phi.shape
    result = np.zeros((p * d, p * d), dtype=float)
    result[:d, : p * d] = np.concatenate(list(phi), axis=1)
    if p > 1:
        result[d:, :-d] = np.eye((p - 1) * d)
    return result


def spectral_radius(matrix: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(matrix, dtype=float)))))


def _unit_spectral_direction(*, d: int, rng: np.random.Generator) -> np.ndarray:
    value = rng.normal(size=(d, d))
    norm = float(np.linalg.svd(value, compute_uv=False)[0])
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("Could not sample a finite nonzero matrix direction.")
    return value / norm


def _draw_coefficients(
    spec: StationaryVARMASpec, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, float, float]:
    directions = np.stack(
        [_unit_spectral_direction(d=spec.d, rng=rng) for _ in range(spec.p)]
    )
    weights = rng.uniform(0.0, 1.0, size=spec.p)
    if float(np.sum(weights)) <= 0.0:
        raise RuntimeError("Degenerate AR coefficient weights.")
    base_phi = directions * (weights / np.sum(weights))[:, None, None]
    target = float(spec.target_companion_radius)

    def objective(scale: float) -> float:
        return spectral_radius(companion_matrix(scale * base_phi)) - target

    high = 1.0
    while objective(high) < 0.0 and high < 2.0**20:
        high *= 2.0
    if objective(high) < 0.0:
        raise RuntimeError("Could not bracket the requested companion radius.")
    scale = float(brentq(objective, 0.0, high, xtol=1e-14, rtol=1e-14))
    phi = scale * base_phi
    achieved = spectral_radius(companion_matrix(phi))
    if abs(achieved - target) > 1e-12:
        raise RuntimeError(
            f"Companion-radius rescaling failed: target={target}, got {achieved}."
        )
    theta = np.stack(
        [
            spec.ma_amplitude
            * (spec.ma_decay**index)
            * _unit_spectral_direction(d=spec.d, rng=rng)
            for index in range(spec.q)
        ]
    )
    return phi, theta, scale, achieved


def augmented_transition(
    phi: np.ndarray, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return F,G for [U_t..U_(t-p+1), eps_t..eps_(t-q+1)]."""

    phi = np.asarray(phi, dtype=float)
    theta = np.asarray(theta, dtype=float)
    p, d, _ = phi.shape
    q = theta.shape[0]
    blocks = p + q
    F = np.zeros((blocks * d, blocks * d), dtype=float)
    G = np.zeros((blocks * d, d), dtype=float)
    F[:d, : p * d] = np.concatenate(list(phi), axis=1)
    F[:d, p * d :] = np.concatenate(list(theta), axis=1)
    G[:d] = np.eye(d)
    if p > 1:
        F[d : p * d, : (p - 1) * d] = np.eye((p - 1) * d)
    G[p * d : (p + 1) * d] = np.eye(d)
    if q > 1:
        F[(p + 1) * d :, p * d : (p + q - 1) * d] = np.eye((q - 1) * d)
    return F, G


def _sample_stationary_state(
    covariance: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    covariance = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    scale = max(1.0, float(np.max(np.abs(values))))
    if float(np.min(values)) < -1e-11 * scale:
        raise RuntimeError("Stationary covariance is not positive semidefinite.")
    values = np.maximum(values, 0.0)
    return vectors @ (np.sqrt(values) * rng.normal(size=values.shape[0]))


def generate_e1_dataset(
    *,
    seeds: SeedBundle,
    num_windows: int,
    window_length: int,
    stride: int,
    spec: StationaryVARMASpec | None = None,
) -> E1Dataset:
    """Generate one exactly stationary, chronologically windowed E1 dataset."""

    spec = spec or StationaryVARMASpec()
    N, w, step = int(num_windows), int(window_length), int(stride)
    if min(N, w, step) < 1:
        raise ValueError("num_windows, window_length, and stride must be positive.")
    children = seeds.sequence("dataset_generation").spawn(3)
    parameter_rng, initial_rng, innovation_rng = map(np.random.default_rng, children)
    phi, theta, coefficient_scale, achieved_radius = _draw_coefficients(
        spec, parameter_rng
    )
    F, G = augmented_transition(phi, theta)
    innovation_covariance = np.eye(spec.d)
    Q = G @ innovation_covariance @ G.T
    stationary_covariance = solve_discrete_lyapunov(F, Q)
    residual = stationary_covariance - F @ stationary_covariance @ F.T - Q
    residual_relative = float(
        np.linalg.norm(residual, ord="fro")
        / max(1.0, np.linalg.norm(stationary_covariance, ord="fro"))
    )
    if residual_relative > 1e-10:
        raise RuntimeError(f"Lyapunov residual is too large: {residual_relative}.")

    raw_length = w + (N - 1) * step + 1
    state = _sample_stationary_state(stationary_covariance, initial_rng)
    U = np.empty((raw_length, spec.d), dtype=float)
    U[0] = state[: spec.d]
    for time_index in range(1, raw_length):
        innovation = innovation_rng.multivariate_normal(
            np.zeros(spec.d), innovation_covariance
        )
        state = F @ state + G @ innovation
        U[time_index] = state[: spec.d]
    bounded = np.tanh(U / float(spec.observation_scale))

    origins = (w - 1) + np.arange(N, dtype=np.int64) * step
    windows = np.stack(
        [bounded[origin - w + 1 : origin + 1] for origin in origins]
    )
    context = TeacherContext(
        prediction_origins=origins,
        future_indices=origins + 1,
        future_observations=bounded[origins + 1],
    )
    functionals = e1_functionals()
    targets = np.empty((N, len(functionals)), dtype=float)
    for example_index, window in enumerate(windows):
        for task_index, functional in enumerate(functionals):
            targets[example_index, task_index] = functional.evaluate(
                window, index=example_index, context=context
            )
    task_names = tuple(str(functional.name) for functional in functionals)
    metadata: dict[str, Any] = {
        "protocol_version": "quark-e1-data/v1",
        "root_entropy": list(seeds.root_entropy),
        "num_windows": N,
        "window_length": w,
        "stride": step,
        "raw_length": raw_length,
        "dimension": spec.d,
        "orders": {"p": spec.p, "q": spec.q},
        "innovation_covariance": "identity",
        "observation_scale": spec.observation_scale,
        "target_companion_radius": spec.target_companion_radius,
        "achieved_companion_radius": achieved_radius,
        "coefficient_scale": coefficient_scale,
        "ma_amplitude": spec.ma_amplitude,
        "ma_decay": spec.ma_decay,
        "stationary_initialization": "augmented-gaussian-lyapunov",
        "lyapunov_relative_residual": residual_relative,
        "dataset_child_spawn_keys": {
            name: list(sequence.spawn_key)
            for name, sequence in zip(
                ("coefficient_parameters", "stationary_initial_state", "innovations"),
                children,
            )
        },
        "task_functionals_stream": "unused-fixed-protocol-directions",
    }
    return E1Dataset(
        windows=windows,
        targets=targets,
        task_names=task_names,
        context=context,
        prediction_origins=origins,
        phi=phi,
        theta=theta,
        transition=F,
        noise_map=G,
        stationary_covariance=stationary_covariance,
        metadata=metadata,
    )
