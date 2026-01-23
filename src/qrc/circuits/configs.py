"""src.qrc.circuits.configs
========================

Configuration utilities for quantum reservoir circuits.

This module defines:

- a small topology description (:class:`QRTopologyConfig`)
- a base configuration interface (:class:`BaseQRConfig`)
- a concrete ring-topology configuration (:class:`RingQRConfig`)
- utilities to generate Johnson–Lindenstrauss (JL) random projections

The JL projection is used to map an input vector ``x in R^d`` to an injected
coordinate vector ``z = x @ Pi`` of dimension ``zeta`` (typically ``zeta = n``,
the number of reservoir qubits).

All returned configurations are *pure NumPy* objects and are therefore easy to
serialize and to use in tests.

"""

from abc import ABC
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


def sample_jl_projections_gaussian(input_dim: int, zeta: int, seed: int) -> np.ndarray:
    """Sample a Gaussian Johnson–Lindenstrauss projection matrix.

    The returned matrix :math:`\Pi \in \mathbb{R}^{d \times \zeta}` has entries
    distributed as :math:`\mathcal{N}(0, 1/\zeta)`. Applying

    ``z = x @ Pi``

    maps a vector ``x in R^d`` to ``z in R^zeta`` while approximately preserving
    pairwise distances for suitably large ``zeta``.

    Parameters
    ----------
    input_dim : int
        Original dimension ``d``.
    zeta : int
        Projected dimension ``ζ`` (often chosen equal to the number of qubits).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Projection matrix of shape ``(input_dim, zeta)``.

    Examples
    --------
    >>> import numpy as np
    >>> Pi = sample_jl_projections_gaussian(input_dim=100, zeta=20, seed=123)
    >>> x = np.random.default_rng(0).normal(size=(100,))
    >>> z = x @ Pi
    >>> z.shape
    (20,)
    """
    rng = np.random.default_rng(seed)
    projection = rng.normal(loc=0.0, scale=1.0 / np.sqrt(zeta), size=(input_dim, zeta))
    return projection


@dataclass(frozen=True)
class QRTopologyConfig:
    """Topology description for a reservoir register.

    Parameters
    ----------
    num_qubits : int
        Number of reservoir qubits.
    edges : Sequence[Tuple[int, int]]
        Undirected coupling edges. Each tuple ``(i, j)`` denotes a two-qubit
        interaction between qubits ``i`` and ``j``.

    Notes
    -----
    The ordering of edges is used when creating parameter vectors for ZZ
    couplings and therefore affects the parameter column ordering expected by
    downstream runners.
    """
    num_qubits: int
    edges: Sequence[Tuple[int, int]]


class BaseQRConfig(ABC):
    """Abstract interface for reservoir configurations.

    Implementations should provide at least the following attributes:

    Attributes
    ----------
    topology : QRTopologyConfig
        Connectivity graph of the reservoir qubits.
    input_dim : int
        Dimension ``d`` of the input vector at each time step.
    num_qubits : int
        Number of reservoir qubits ``n`` (also used as injection dimension).
    projection : numpy.ndarray
        JL projection matrix with shape ``(input_dim, num_qubits)``.
    seed : int
        Seed used for deterministic construction.
    """
    topology: QRTopologyConfig
    input_dim: int
    num_qubits: int
    projection: np.ndarray
    seed: int


class RingQRConfig(BaseQRConfig):
    """Ring-topology reservoir configuration with a JL projection.

    Parameters
    ----------
    input_dim : int
        Input dimension ``d``.
    num_qubits : int
        Number of reservoir qubits ``n``.
    seed : int, default=12345
        Seed used to sample the JL projection.

    Attributes
    ----------
    topology : QRTopologyConfig
        Ring connectivity.
    projection : numpy.ndarray
        JL projection matrix of shape ``(input_dim, num_qubits)``.
    """

    def __init__(self, input_dim: int, num_qubits: int, seed: int = 12345):
        self.topology = ring_topology(num_qubits=num_qubits)
        self.input_dim = int(input_dim)
        self.num_qubits = int(num_qubits)
        self.seed = int(seed)
        self.projection = sample_jl_projections_gaussian(
            input_dim=self.input_dim,
            zeta=self.num_qubits,
            seed=self.seed,
        )


def ring_topology(num_qubits: int) -> QRTopologyConfig:
    """Construct a ring coupling topology.

    The ring uses nearest-neighbour couplings ``(i, i+1)`` and, for ``n>2``,
    closes the ring with the edge ``(n-1, 0)``.

    Parameters
    ----------
    num_qubits : int
        Number of qubits ``n``.

    Returns
    -------
    QRTopologyConfig
        Topology object containing the edge list.
    """
    qubits = list(range(num_qubits))
    edges = []
    n = len(qubits)
    for i in range(n - 1):
        edges.append((qubits[i], qubits[i + 1]))
    if n > 2:
        edges.append((qubits[n - 1], qubits[0]))

    return QRTopologyConfig(num_qubits=num_qubits, edges=tuple(edges))
