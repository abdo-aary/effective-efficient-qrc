from abc import ABC
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


def sample_jl_projections_gaussian(input_dim: int, zeta: int, seed: int) -> np.ndarray:
    """
    Returns Π with shape (d, zeta) such that entries ~ N(0, 1/zeta).
    Using z = x @ Π gives z in R^zeta:
    Usage example:
    ``
        num_reservoirs = R = 5
        input_dim = d = 100
        num_qubits = zeta = 20  # typically ζ = injection dimension

        Pi = sample_jl_projections_gaussian(d, zeta, seed=123)

        x = np.random.default_rng(0).normal(size=(d,))
        z0 = x @ Pi   # shape (zeta,)
        z0.shape
    ``
    """
    rng = np.random.default_rng(seed)
    projection = rng.normal(loc=0.0, scale=1.0 / np.sqrt(zeta), size=(input_dim, zeta))
    return projection


@dataclass(frozen=True)
class QRTopologyConfig:
    # This stores the indices of the qubits
    num_qubits: int
    # This stores the topology of the hardware: qubit-qubit interactions.
    edges: Sequence[Tuple[int, int]]


class BaseQRConfig(ABC):
    topology: QRTopologyConfig
    input_dim: int
    num_qubits: int
    projection: np.ndarray
    seed: int


class RingQRConfig(BaseQRConfig):
    def __init__(self, input_dim, num_qubits: int, seed: int = 12345):
        """
        :param input_dim:
        :param num_qubits:
        # :param lambda_0: contains the lambda_0 value that insures non-collision with high probability (lemma 4.3).
        :param seed:
        """
        # rng = np.random.default_rng(seed)
        # rand_lambdas = rng.random(num_reservoirs - 1)
        # # move any exact 0 to the smallest representable float > 0 (almost never happens anyway)
        # rand_lambdas[rand_lambdas == 0.0] = np.nextafter(0.0, 1.0)
        #
        # self.lambdas = [lambda_0] + rand_lambdas.tolist()
        self.topology = ring_topology(num_qubits=num_qubits)
        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.projection: np.ndarray = sample_jl_projections_gaussian(input_dim=input_dim, zeta=num_qubits, seed=seed)


def ring_topology(num_qubits) -> QRTopologyConfig:
    # Order the sequence of qubits by indices
    # Run a (i, i+1) edges and finish by (n-1, 0) edge
    qubits = list(range(num_qubits))
    edges = []
    n = len(qubits)
    for i in range(n - 1):
        edges.append((qubits[i], qubits[i + 1]))
    if n > 2:
        edges.append((qubits[n - 1], qubits[0]))

    return QRTopologyConfig(num_qubits=num_qubits, edges=tuple(edges))
