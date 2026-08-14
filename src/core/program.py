"""The mathematical, backend-neutral QuaRK program."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .observables import ObservableSet


def _readonly_array(value: object, *, dtype: object, ndim: int, name: str) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {array.shape}.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ProjectionSpec:
    matrix: np.ndarray
    mode: str = "gaussian_jl"

    def __post_init__(self) -> None:
        matrix = _readonly_array(self.matrix, dtype=np.float64, ndim=2, name="projection matrix")
        if matrix.shape[0] < 1 or matrix.shape[1] < 1:
            raise ValueError("Projection dimensions must be positive.")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "mode", str(self.mode))

    @property
    def input_dim(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def encoded_width(self) -> int:
        return int(self.matrix.shape[1])


@dataclass(frozen=True)
class ReservoirTopology:
    num_qubits: int
    edges: tuple[tuple[int, int], ...]
    name: str = "custom"

    def __post_init__(self) -> None:
        n = int(self.num_qubits)
        if n < 1:
            raise ValueError("num_qubits must be positive.")
        edges = tuple((int(a), int(b)) for a, b in self.edges)
        for a, b in edges:
            if a == b or not (0 <= a < n and 0 <= b < n):
                raise ValueError(f"Invalid topology edge {(a, b)} for n={n}.")
        object.__setattr__(self, "num_qubits", n)
        object.__setattr__(self, "edges", edges)


@dataclass(frozen=True)
class ReservoirParameters:
    """Ordered parameters for all spatially multiplexed reservoirs."""

    zz: np.ndarray
    x_fields: np.ndarray
    z_fields: np.ndarray
    reset_rates: np.ndarray
    reservoir_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        zz = _readonly_array(self.zz, dtype=np.float64, ndim=2, name="zz")
        x_fields = _readonly_array(self.x_fields, dtype=np.float64, ndim=2, name="x_fields")
        z_fields = _readonly_array(self.z_fields, dtype=np.float64, ndim=2, name="z_fields")
        reset_rates = _readonly_array(
            self.reset_rates, dtype=np.float64, ndim=1, name="reset_rates"
        )
        R = int(reset_rates.shape[0])
        if R < 1:
            raise ValueError("At least one reservoir realization is required.")
        if zz.shape[0] != R or x_fields.shape[0] != R or z_fields.shape[0] != R:
            raise ValueError("All reservoir parameter arrays must have the same first dimension.")
        if x_fields.shape != z_fields.shape:
            raise ValueError("x_fields and z_fields must have identical shapes.")
        if np.any((reset_rates < 0.0) | (reset_rates > 1.0)):
            raise ValueError("reset_rates (lambda values) must lie in [0, 1].")
        ids = self.reservoir_ids or tuple(f"reservoir-{index}" for index in range(R))
        if len(ids) != R or len(set(ids)) != R:
            raise ValueError("reservoir_ids must be unique and match the number of reservoirs.")
        object.__setattr__(self, "zz", zz)
        object.__setattr__(self, "x_fields", x_fields)
        object.__setattr__(self, "z_fields", z_fields)
        object.__setattr__(self, "reset_rates", reset_rates)
        object.__setattr__(self, "reservoir_ids", tuple(str(item) for item in ids))

    @property
    def count(self) -> int:
        return int(self.reset_rates.shape[0])

    @property
    def num_qubits(self) -> int:
        return int(self.x_fields.shape[1])

    @classmethod
    def sample_legacy(
        cls,
        *,
        num_reservoirs: int,
        num_qubits: int,
        num_edges: int,
        lambda_first: float,
        seed: int,
        eps: float = 1e-8,
        dynamics_mode: str = "random",
    ) -> "ReservoirParameters":
        """Reproduce the legacy parameter draw order exactly."""

        R = int(num_reservoirs)
        if R < 1:
            raise ValueError("num_reservoirs must be positive.")
        if not (0.0 <= float(lambda_first) <= 1.0):
            raise ValueError("lambda_first must lie in [0, 1].")
        rng = np.random.default_rng(int(seed))
        zz = rng.uniform(-np.pi, np.pi, size=(R, int(num_edges)))
        x_fields = rng.uniform(-np.pi, np.pi, size=(R, int(num_qubits)))
        z_fields = rng.uniform(-np.pi, np.pi, size=(R, int(num_qubits)))
        reset_rates = np.empty(R, dtype=float)
        reset_rates[0] = float(lambda_first)
        if R > 1:
            reset_rates[1:] = rng.uniform(float(eps), 1.0 - float(eps), size=R - 1)
        if dynamics_mode == "zero":
            zz.fill(0.0)
            x_fields.fill(0.0)
            z_fields.fill(0.0)
        elif dynamics_mode != "random":
            raise ValueError("dynamics_mode must be 'random' or 'zero'.")
        return cls(zz=zz, x_fields=x_fields, z_fields=z_fields, reset_rates=reset_rates)


@dataclass(frozen=True)
class BalancedReservoirParameters:
    """Frozen branch parameters for the paper's balanced random mixer."""

    local_axes: np.ndarray
    local_angles: np.ndarray
    edge_axes_left: np.ndarray
    edge_axes_right: np.ndarray
    edge_angles: np.ndarray
    matching_orders: np.ndarray
    reset_rates: np.ndarray
    reservoir_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        local_axes = _readonly_array(
            self.local_axes, dtype=np.float64, ndim=3, name="local_axes"
        )
        local_angles = _readonly_array(
            self.local_angles, dtype=np.float64, ndim=2, name="local_angles"
        )
        edge_axes_left = _readonly_array(
            self.edge_axes_left, dtype=np.float64, ndim=3, name="edge_axes_left"
        )
        edge_axes_right = _readonly_array(
            self.edge_axes_right, dtype=np.float64, ndim=3, name="edge_axes_right"
        )
        edge_angles = _readonly_array(
            self.edge_angles, dtype=np.float64, ndim=2, name="edge_angles"
        )
        matching_orders = _readonly_array(
            self.matching_orders, dtype=np.int64, ndim=2, name="matching_orders"
        )
        reset_rates = _readonly_array(
            self.reset_rates, dtype=np.float64, ndim=1, name="reset_rates"
        )

        R = int(reset_rates.shape[0])
        if R < 1:
            raise ValueError("At least one balanced reservoir realization is required.")
        if local_axes.shape != (*local_angles.shape, 3):
            raise ValueError("local_axes must have shape (R,n,3) matching local_angles.")
        if local_angles.shape[0] != R:
            raise ValueError("Balanced local parameters must have first dimension R.")
        if edge_axes_left.shape != edge_axes_right.shape:
            raise ValueError("Left and right edge-axis arrays must have identical shapes.")
        if edge_axes_left.shape != (*edge_angles.shape, 3):
            raise ValueError("Edge axes must have shape (R,E,3) matching edge_angles.")
        if edge_angles.shape[0] != R or matching_orders.shape[0] != R:
            raise ValueError("Balanced edge and matching arrays must have first dimension R.")
        for name, axes in (
            ("local_axes", local_axes),
            ("edge_axes_left", edge_axes_left),
            ("edge_axes_right", edge_axes_right),
        ):
            norms = np.linalg.norm(axes, axis=-1)
            if not np.allclose(norms, 1.0, atol=1e-12, rtol=0.0):
                raise ValueError(f"{name} rows must be unit vectors.")
        J = int(matching_orders.shape[1])
        expected_order = np.arange(J, dtype=np.int64)
        if any(not np.array_equal(np.sort(row), expected_order) for row in matching_orders):
            raise ValueError("Every matching order must be a permutation of range(J).")
        if np.any((reset_rates <= 0.0) | (reset_rates >= 1.0)):
            raise ValueError("Balanced reset_rates must lie strictly in (0,1).")
        ids = self.reservoir_ids or tuple(
            f"balanced-reservoir-{index}" for index in range(R)
        )
        if len(ids) != R or len(set(ids)) != R:
            raise ValueError("reservoir_ids must be unique and match the number of reservoirs.")

        object.__setattr__(self, "local_axes", local_axes)
        object.__setattr__(self, "local_angles", local_angles)
        object.__setattr__(self, "edge_axes_left", edge_axes_left)
        object.__setattr__(self, "edge_axes_right", edge_axes_right)
        object.__setattr__(self, "edge_angles", edge_angles)
        object.__setattr__(self, "matching_orders", matching_orders)
        object.__setattr__(self, "reset_rates", reset_rates)
        object.__setattr__(self, "reservoir_ids", tuple(str(item) for item in ids))

    @property
    def count(self) -> int:
        return int(self.reset_rates.shape[0])

    @property
    def num_qubits(self) -> int:
        return int(self.local_angles.shape[1])


@dataclass(frozen=True)
class ResetChannelSpec:
    state: str = "plus"

    def __post_init__(self) -> None:
        if self.state != "plus":
            raise ValueError("The initial refactor supports only the |+> reset state.")


@dataclass(frozen=True)
class QuaRKProgram:
    projection: ProjectionSpec
    topology: ReservoirTopology
    reservoirs: ReservoirParameters | BalancedReservoirParameters
    reset_channel: ResetChannelSpec
    observables: ObservableSet
    window_length: int
    angle_map: str = "linear"
    angle_scale: float = float(np.pi * (1.0 - 1e-6))

    def __post_init__(self) -> None:
        w = int(self.window_length)
        if w < 1:
            raise ValueError("window_length must be positive.")
        n = self.topology.num_qubits
        if self.projection.encoded_width != n:
            raise ValueError("Projection width must equal the reservoir qubit count.")
        if self.reservoirs.num_qubits != n:
            raise ValueError("Reservoir fields do not match the topology width.")
        if isinstance(self.reservoirs, ReservoirParameters):
            if self.reservoirs.zz.shape[1] != len(self.topology.edges):
                raise ValueError("ZZ parameter width must equal the number of topology edges.")
        elif self.reservoirs.edge_angles.shape[1] != len(self.topology.edges):
            raise ValueError("Balanced edge parameter width must equal the topology edges.")
        if self.observables.num_qubits != n:
            raise ValueError("Observable width must equal the reservoir qubit count.")
        if self.angle_map not in {"linear", "tanh"}:
            raise ValueError("angle_map must be 'linear' or 'tanh'.")
        object.__setattr__(self, "window_length", w)
        object.__setattr__(self, "angle_scale", float(self.angle_scale))

    @property
    def input_dim(self) -> int:
        return self.projection.input_dim

    @property
    def num_qubits(self) -> int:
        return self.topology.num_qubits

    @property
    def num_reservoirs(self) -> int:
        return self.reservoirs.count

    def validate_windows(self, windows: object) -> np.ndarray:
        array = np.asarray(windows, dtype=float)
        expected_tail = (self.window_length, self.input_dim)
        if array.ndim != 3 or tuple(array.shape[1:]) != expected_tail:
            raise ValueError(
                f"windows must have shape (N,{expected_tail[0]},{expected_tail[1]}), "
                f"got {array.shape}."
            )
        return array

    def fingerprint(self) -> str:
        payload = {
            "window_length": self.window_length,
            "angle_map": self.angle_map,
            "angle_scale": self.angle_scale,
            "topology": {
                "num_qubits": self.num_qubits,
                "edges": self.topology.edges,
                "name": self.topology.name,
            },
            "projection_mode": self.projection.mode,
            "observable_labels": self.observables.labels,
            "reset_state": self.reset_channel.state,
            "reservoir_ids": self.reservoirs.reservoir_ids,
            "reservoir_parameterization": (
                "balanced_random_axis"
                if isinstance(self.reservoirs, BalancedReservoirParameters)
                else "legacy_ising"
            ),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        if isinstance(self.reservoirs, BalancedReservoirParameters):
            reservoir_arrays = (
                self.reservoirs.local_axes,
                self.reservoirs.local_angles,
                self.reservoirs.edge_axes_left,
                self.reservoirs.edge_axes_right,
                self.reservoirs.edge_angles,
                self.reservoirs.matching_orders,
                self.reservoirs.reset_rates,
            )
        else:
            reservoir_arrays = (
                self.reservoirs.zz,
                self.reservoirs.x_fields,
                self.reservoirs.z_fields,
                self.reservoirs.reset_rates,
            )
        for array in (self.projection.matrix, *reservoir_arrays):
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()
