"""Sampling helpers for genuine local-Pauli classical shadows."""

from __future__ import annotations

import numpy as np

from src.estimators.csmom import ShadowSnapshots

_H = np.asarray([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2.0)
_SDG = np.asarray([[1, 0], [0, -1j]], dtype=np.complex128)
_MEASUREMENT_ROTATIONS = (
    _H,  # X: H
    _H @ _SDG,  # Y: Sdg then H
    np.eye(2, dtype=np.complex128),  # Z
)


def _basis_rotation(basis_qorder: np.ndarray) -> np.ndarray:
    # Hilbert-space tensor order is q_{n-1} ... q_0.
    result = np.asarray([[1.0 + 0.0j]])
    for q in range(int(basis_qorder.shape[0]) - 1, -1, -1):
        result = np.kron(result, _MEASUREMENT_ROTATIONS[int(basis_qorder[q])])
    return result


def sample_local_pauli_shadows_density(
    states: np.ndarray,
    bases: np.ndarray,
    *,
    outcome_rng: np.random.Generator,
) -> ShadowSnapshots:
    """Sample local-Pauli outcomes from reduced density matrices.

    ``states`` has shape ``(N,R,D,D)`` and ``bases`` has shape ``(N,R,S,n)``.
    The implementation caches product-basis probabilities per state and basis,
    which is appropriate for the deliberately small Aer oracle.
    """

    states = np.asarray(states, dtype=np.complex128)
    bases = np.asarray(bases, dtype=np.uint8)
    if states.ndim != 4 or states.shape[-1] != states.shape[-2]:
        raise ValueError(f"states must have shape (N,R,D,D), got {states.shape}.")
    N, R, dim, _ = states.shape
    if bases.ndim != 4 or bases.shape[:2] != (N, R):
        raise ValueError("bases must have shape (N,R,S,n) matching states.")
    S, n = int(bases.shape[2]), int(bases.shape[3])
    if dim != (1 << n):
        raise ValueError(f"State dimension {dim} does not match n={n}.")

    outcomes = np.empty((N, R, S, n), dtype=np.int8)
    for i in range(N):
        for r in range(R):
            unique, inverse = np.unique(bases[i, r], axis=0, return_inverse=True)
            probabilities: list[np.ndarray] = []
            for basis in unique:
                rotation = _basis_rotation(basis)
                rotated = rotation @ states[i, r] @ rotation.conj().T
                probs = np.clip(np.real(np.diag(rotated)), 0.0, None)
                total = float(probs.sum())
                if not np.isfinite(total) or total <= 0.0:
                    raise ValueError("Invalid measurement probabilities derived from density matrix.")
                probabilities.append(probs / total)
            uniforms = outcome_rng.random(S)
            for s in range(S):
                cdf = np.cumsum(probabilities[int(inverse[s])])
                basis_index = min(int(np.searchsorted(cdf, uniforms[s], side="right")), dim - 1)
                for q in range(n):
                    outcomes[i, r, s, q] = 1 if ((basis_index >> q) & 1) == 0 else -1
    return ShadowSnapshots(bases=bases, outcomes=outcomes)
