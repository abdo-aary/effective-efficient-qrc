"""Framework-independent Pauli expectation calculations."""

from __future__ import annotations

import numpy as np

_SINGLE_PAULI = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.asarray([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.asarray([[1, 0], [0, -1]], dtype=np.complex128),
}


def pauli_matrix(label: str) -> np.ndarray:
    matrix = np.asarray([[1.0 + 0.0j]])
    for character in label:
        try:
            matrix = np.kron(matrix, _SINGLE_PAULI[character])
        except KeyError as exc:
            raise ValueError(f"Invalid Pauli label {label!r}.") from exc
    return matrix


def exact_pauli_expectations(
    states: np.ndarray,
    observable_labels: tuple[str, ...],
) -> np.ndarray:
    """Return ``Tr(rho O)`` with shape ``(N,R,K)``."""

    states = np.asarray(states)
    if states.ndim != 4 or states.shape[-1] != states.shape[-2]:
        raise ValueError(f"states must have shape (N,R,D,D), got {states.shape}.")
    dim = int(states.shape[-1])
    n = dim.bit_length() - 1
    if (1 << n) != dim:
        raise ValueError("State dimension must be a power of two.")
    matrices = []
    for label in observable_labels:
        if len(label) != n:
            raise ValueError(f"Observable {label!r} does not match n={n}.")
        matrices.append(pauli_matrix(label))
    operators = np.stack(matrices, axis=0)
    return np.einsum("nrij,kji->nrk", states, operators, optimize=True).real
