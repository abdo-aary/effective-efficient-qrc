"""Deterministic backend-neutral factories for protocol QuaRK programs."""

from __future__ import annotations

import numpy as np

from .observables import ObservableSet
from .program import (
    BalancedReservoirParameters,
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


def cycle_matchings(topology: ReservoirTopology) -> tuple[tuple[int, ...], ...]:
    """Return the canonical two/three-layer edge colouring of a cycle."""

    n = topology.num_qubits
    expected = tuple((q, q + 1) for q in range(n - 1)) + ((n - 1, 0),)
    if n < 3 or topology.edges != expected:
        raise ValueError("Balanced cycle matchings require the canonical ring topology.")
    if n % 2 == 0:
        return (tuple(range(0, n, 2)), tuple(range(1, n, 2)))
    return (
        tuple(range(0, n - 1, 2)),
        tuple(range(1, n - 1, 2)),
        (n - 1,),
    )


def _sphere_axes(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    axes = rng.normal(size=(*shape, 3))
    norms = np.linalg.norm(axes, axis=-1, keepdims=True)
    while np.any(norms == 0.0):
        mask = norms[..., 0] == 0.0
        axes[mask] = rng.normal(size=(int(mask.sum()), 3))
        norms = np.linalg.norm(axes, axis=-1, keepdims=True)
    return axes / norms


def sample_balanced_reservoirs(
    *,
    num_reservoirs: int,
    topology: ReservoirTopology,
    lambda_plus: float,
    gamma: float,
    mixer_rng: np.random.Generator,
    memory_rng: np.random.Generator,
    angle_distribution: str = "uniform_symmetric",
    lambda_0: float = float(np.exp(-1.0)),
) -> BalancedReservoirParameters:
    """Sample the paper's balanced mixer and coupled log-uniform rates."""

    R = int(num_reservoirs)
    upper = float(lambda_plus)
    lower = float(lambda_0)
    gamma = float(gamma)
    matchings = cycle_matchings(topology)
    if R < 1:
        raise ValueError("num_reservoirs must be positive.")
    if not 0.0 < lower < upper < 1.0:
        raise ValueError("Expected 0 < lambda_0 < lambda_plus < 1.")
    if not 0.0 < gamma <= np.pi:
        raise ValueError("gamma must lie in (0,pi].")
    if angle_distribution != "uniform_symmetric":
        raise ValueError("Only the locked uniform_symmetric angle law is supported.")

    n = topology.num_qubits
    E = len(topology.edges)
    # Draw complete branches in order so every requested R is a true prefix.
    branch_draws = []
    for _ in range(R):
        branch_draws.append(
            (
                _sphere_axes(mixer_rng, (n,)),
                mixer_rng.uniform(-gamma, gamma, size=n),
                _sphere_axes(mixer_rng, (E,)),
                _sphere_axes(mixer_rng, (E,)),
                mixer_rng.uniform(-gamma, gamma, size=E),
                mixer_rng.permutation(len(matchings)),
            )
        )
    local_axes = np.stack([draw[0] for draw in branch_draws])
    local_angles = np.stack([draw[1] for draw in branch_draws])
    edge_axes_left = np.stack([draw[2] for draw in branch_draws])
    edge_axes_right = np.stack([draw[3] for draw in branch_draws])
    edge_angles = np.stack([draw[4] for draw in branch_draws])
    matching_orders = np.stack([draw[5] for draw in branch_draws])
    rate_uniforms = memory_rng.uniform(0.0, 1.0, size=R)
    reset_rates = lower * np.power(upper / lower, rate_uniforms)

    import hashlib

    ids: list[str] = []
    for index in range(R):
        digest = hashlib.sha256()
        for array in (
            local_axes[index],
            local_angles[index],
            edge_axes_left[index],
            edge_axes_right[index],
            edge_angles[index],
            matching_orders[index],
        ):
            digest.update(np.asarray(array).tobytes(order="C"))
        ids.append(f"balanced-{digest.hexdigest()[:20]}")
    return BalancedReservoirParameters(
        local_axes=local_axes,
        local_angles=local_angles,
        edge_axes_left=edge_axes_left,
        edge_axes_right=edge_axes_right,
        edge_angles=edge_angles,
        matching_orders=matching_orders,
        reset_rates=reset_rates,
        reservoir_ids=tuple(ids),
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
