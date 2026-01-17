from __future__ import annotations

import math

import numpy as np
from qiskit.circuit import ParameterVector
from typing import List, Sequence, Tuple
from itertools import combinations, product
from qiskit.quantum_info import SparsePauliOp


def angle_positioning_linear(
        z: ParameterVector,
        *,
        scale: float = np.pi * (1.0 - 1e-6),  # keep strict (-pi, pi) if |z_j|<1
):
    """
    Elementwise θ(z_j) = scale * z_j.

    Properties (for each coordinate):
      - continuous
      - strictly monotone increasing if scale>0
      - injective
      - maps (-1,1) -> (-scale, scale) ⊂ (-pi, pi) if scale < pi
    """
    return [scale * z[j] for j in range(len(z))]


def generate_k_local_paulis(locality: int, num_qubits: int) -> List[SparsePauliOp]:
    """
    Generate all Pauli strings on `num_qubits` whose locality is <= `locality`,
    returned as `SparsePauliOp` objects.

    - Locality = number of non-identity positions (X, Y, or Z). Qubit 0 is the **rightmost** character of the string.

    The all-identity operator (0-local) is *not* included.
    """

    if locality < 1:
        return []

    locality = min(locality, num_qubits)

    paulis: List[SparsePauliOp] = []
    letters = ("X", "Y", "Z")

    # choose locality ℓ = 1..k
    for l in range(1, locality + 1):
        # choose which qubits are non-identity
        for qubit_indices in combinations(range(num_qubits), l):
            # for those qubits, choose which Pauli (X/Y/Z) acts
            for pauli_choice in product(letters, repeat=l):
                chars = ["I"] * num_qubits

                for q, p in zip(qubit_indices, pauli_choice):
                    # map qubit index -> string index (qubit 0 is rightmost)
                    str_idx = num_qubits - 1 - q
                    chars[str_idx] = p

                pauli_str = "".join(chars)
                paulis.append(SparsePauliOp(pauli_str))

    return paulis


def get_theoretical_shots(
        eps: float,
        delta: float,
        locality: int,
        num_data_pts: int,
        num_obs: int,
        num_draws: int,
) -> int:
    """
    Theoretical number of classical-shadow snapshots ensuring
    sup-norm error <= eps for all (draw, data point, observable)
    with probability at least 1 - delta.

    Parameters
    ----------
    eps : float
        Target uniform accuracy ε > 0.
    delta : float
        Failure probability δ in (0, 1).
    locality : int
        k in 'k-local' for the observables.
    num_data_pts : int
        M = number of data points.
    num_obs : int
        Number of observables per state.
    num_draws : int
        R = number of circuit draws / feature-map draws.

    Returns
    -------
    int
        Recommended total number of snapshots N_shots.
    """
    if eps <= 0:
        raise ValueError("eps must be > 0.")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1).")

    # 34 / eps^2 factor from Theorem 1
    prefactor = 34.0 / (eps ** 2)

    # Upper bound on shadow norm^2 for k-local observables
    shadow_norm_sq = (3.0 / 2.0) ** locality

    # Total number of “features’’ we care about in the union bound:
    num_features = num_draws * num_data_pts * num_obs

    # N ≈ (34/eps^2) * (3/2)^k * log(num_features / delta)
    n_shots = prefactor * shadow_norm_sq * math.log(num_features / delta)

    return math.ceil(n_shots)
