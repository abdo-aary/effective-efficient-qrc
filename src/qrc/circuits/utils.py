"""src.qrc.circuits.utils
======================

Small utilities used by the circuit construction and feature-map pipeline.

This module currently provides:

- **Angle positioning** maps, which turn injected coordinates ``z`` into
  gate angles (typically used for ``Ry`` data injection).
- Utilities to generate k-local Pauli observables for measurement/feature maps.
- A helper to compute the *theoretical* number of classical-shadow snapshots.

All functions are designed to work with Qiskit's :class:`~qiskit.circuit.ParameterExpression`
constraints (notably, only a limited set of elementwise operations is available).
"""

from __future__ import annotations

import math
from itertools import combinations, product
from typing import List

import numpy as np
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def angle_positioning_linear(
    z: ParameterVector,
    *,
    scale: float = np.pi * (1.0 - 1e-6),
):
    """Linear angle positioning ``theta(z_j) = scale * z_j`` (elementwise).

    Parameters
    ----------
    z : qiskit.circuit.ParameterVector
        Injected coordinate vector (typically length ``n``).
    scale : float, default=pi*(1-1e-6)
        Multiplicative scale. The default keeps angles strictly inside ``(-pi, pi)``
        for inputs in ``(-1, 1)``.

    Returns
    -------
    list
        List of Qiskit parameter expressions, one per element of `z`.

    Notes
    -----
    For each coordinate:

    - continuous
    - strictly monotone increasing if ``scale > 0``
    - injective (as a map ``z_j -> theta(z_j)``)
    - maps ``(-1, 1) -> (-scale, scale)``.
    """
    return [scale * z[j] for j in range(len(z))]


def _tanh_parameter_expression(u):
    """Compute ``tanh(u)`` as a Qiskit `ParameterExpression`.

    Qiskit's :class:`~qiskit.circuit.ParameterExpression` supports ``exp()`` but
    does not provide a native ``tanh``. We use the numerically stable identity:

    ``tanh(u) = 1 - 2 / (exp(2u) + 1)``

    Parameters
    ----------
    u : qiskit.circuit.ParameterExpression
        Parameter expression.

    Returns
    -------
    qiskit.circuit.ParameterExpression
        Expression representing ``tanh(u)``.
    """
    e2u = (2 * u).exp()
    return 1 - (2 / (e2u + 1))


def angle_positioning_tanh(
    z: ParameterVector,
    *,
    scale: float = np.pi * (1.0 - 1e-6),
):
    """Saturating angle positioning ``theta(z_j) = scale * tanh(z_j)`` (elementwise).

    Parameters
    ----------
    z : qiskit.circuit.ParameterVector
        Injected coordinate vector (typically length ``n``).
    scale : float, default=pi*(1-1e-6)
        Output scale. The default ensures angles remain strictly inside ``(-pi, pi)``
        even for large `z` values.

    Returns
    -------
    list
        List of Qiskit parameter expressions, one per element of `z`.

    Notes
    -----
    This map sends ``R -> (-scale, scale)``. Using a saturating map is often more
    robust than the linear map when the projected coordinates have heavy tails.
    """
    return [scale * _tanh_parameter_expression(z[j]) for j in range(len(z))]


def generate_k_local_paulis(locality: int, num_qubits: int) -> List[SparsePauliOp]:
    """Generate all Pauli strings with locality up to ``k``.

    Locality is the number of non-identity positions (X, Y, or Z). The all-identity
    operator (0-local) is not included.

    Parameters
    ----------
    locality : int
        Maximum locality ``k``. Values <= 0 return an empty list.
    num_qubits : int
        Number of qubits in each Pauli string.

    Returns
    -------
    list of qiskit.quantum_info.SparsePauliOp
        All Pauli operators with locality in ``{1, ..., min(k, num_qubits)}``.

    Notes
    -----
    Qubit indexing convention follows Qiskit's Pauli string convention:
    **qubit 0 corresponds to the rightmost character**.

    Examples
    --------
    For ``num_qubits=2`` and ``locality=1``, the function returns
    ``[XI, YI, ZI, IX, IY, IZ]`` as `SparsePauliOp` objects.
    """
    if locality < 1:
        return []

    locality = min(locality, num_qubits)

    paulis: List[SparsePauliOp] = []
    letters = ("X", "Y", "Z")

    for l in range(1, locality + 1):
        for qubit_indices in combinations(range(num_qubits), l):
            for pauli_choice in product(letters, repeat=l):
                chars = ["I"] * num_qubits
                for q, p in zip(qubit_indices, pauli_choice):
                    str_idx = num_qubits - 1 - q  # qubit 0 is rightmost
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
    """Compute the theoretical number of classical-shadow snapshots.

    The returned value corresponds to a union-bound guarantee that the sup-norm
    estimation error is at most ``eps`` simultaneously over all

    - circuit draws (``num_draws``)
    - data points (``num_data_pts``)
    - observables (``num_obs``)

    with probability at least ``1 - delta``.

    Parameters
    ----------
    eps : float
        Target uniform accuracy ``ε > 0``.
    delta : float
        Failure probability ``δ`` in ``(0, 1)``.
    locality : int
        Locality ``k`` of the observables.
    num_data_pts : int
        Number of data points ``M``.
    num_obs : int
        Number of observables per state.
    num_draws : int
        Number of circuit/reservoir draws ``R``.

    Returns
    -------
    int
        Recommended number of snapshots ``N_shots``.

    Raises
    ------
    ValueError
        If `eps` <= 0 or `delta` is not in (0, 1).

    Notes
    -----
    The formula implemented is:

    ``N ≈ (34/eps^2) * (3/2)^k * log((R*M*num_obs)/delta)``.
    """
    if eps <= 0:
        raise ValueError("eps must be > 0.")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1).")

    prefactor = 34.0 / (eps ** 2)
    shadow_norm_sq = (3.0 / 2.0) ** locality
    num_features = num_draws * num_data_pts * num_obs
    n_shots = prefactor * shadow_norm_sq * math.log(num_features / delta)

    return math.ceil(n_shots)
