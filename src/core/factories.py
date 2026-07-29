"""Deterministic backend-neutral factories for protocol QuaRK programs."""

from __future__ import annotations

import numpy as np

from .observables import ObservableSet
from .program import (
    ProjectionSpec,
    QuaRKProgram,
    ResetChannelSpec,
    ReservoirParameters,
    ReservoirTopology,
)
from .seeds import SeedBundle


def gaussian_jl_projection(
    *, input_dim: int, encoded_width: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample a Gaussian projection with entries distributed as N(0, 1/n)."""

    d = int(input_dim)
    n = int(encoded_width)
    if d < 1 or n < 1:
        raise ValueError("Projection dimensions must be positive.")
    return rng.normal(0.0, 1.0 / np.sqrt(n), size=(d, n))


def ring_topology(num_qubits: int) -> ReservoirTopology:
    n = int(num_qubits)
    if n < 1:
        raise ValueError("num_qubits must be positive.")
    edges = [(index, index + 1) for index in range(n - 1)]
    if n > 2:
        edges.append((n - 1, 0))
    return ReservoirTopology(num_qubits=n, edges=tuple(edges), name="ring")


def sample_protocol_reservoirs(
    *,
    num_reservoirs: int,
    topology: ReservoirTopology,
    reset_rate: float,
    rng: np.random.Generator,
) -> ReservoirParameters:
    """Sample independent reservoir parameters with one fixed reset rate."""

    R = int(num_reservoirs)
    lam = float(reset_rate)
    if R < 1:
        raise ValueError("num_reservoirs must be positive.")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("reset_rate must lie in [0,1].")
    return ReservoirParameters(
        zz=rng.uniform(-np.pi, np.pi, size=(R, len(topology.edges))),
        x_fields=rng.uniform(-np.pi, np.pi, size=(R, topology.num_qubits)),
        z_fields=rng.uniform(-np.pi, np.pi, size=(R, topology.num_qubits)),
        reset_rates=np.full(R, lam, dtype=float),
        reservoir_ids=tuple(f"reservoir-{index}" for index in range(R)),
    )


def make_protocol_program(
    *,
    input_dim: int,
    num_qubits: int,
    num_reservoirs: int,
    locality: int,
    reset_rate: float,
    window_length: int,
    seeds: SeedBundle,
    angle_map: str = "tanh",
) -> QuaRKProgram:
    """Build a protocol program from independent named seed streams."""

    topology = ring_topology(num_qubits)
    projection = gaussian_jl_projection(
        input_dim=input_dim,
        encoded_width=num_qubits,
        rng=seeds.generator("jl_projection"),
    )
    reservoirs = sample_protocol_reservoirs(
        num_reservoirs=num_reservoirs,
        topology=topology,
        reset_rate=reset_rate,
        rng=seeds.generator("reservoir_parameters"),
    )
    return QuaRKProgram(
        projection=ProjectionSpec(projection, mode="gaussian_jl"),
        topology=topology,
        reservoirs=reservoirs,
        reset_channel=ResetChannelSpec(),
        observables=ObservableSet.local_paulis(
            num_qubits=num_qubits, locality=locality
        ),
        window_length=window_length,
        angle_map=angle_map,
    )
