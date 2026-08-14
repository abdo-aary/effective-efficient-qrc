"""Dense independent CPU realization of the balanced reservoir channel."""

from __future__ import annotations

import numpy as np

from src.core.factories import cycle_matchings
from src.core.program import BalancedReservoirParameters, QuaRKProgram

_PAULI = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.asarray([[0.0, -1j], [1j, 0.0]], dtype=np.complex128),
    "Z": np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}


def _tensor_operator(num_qubits: int, factors: dict[int, np.ndarray]) -> np.ndarray:
    result = np.asarray([[1.0 + 0.0j]])
    for qubit in reversed(range(int(num_qubits))):
        result = np.kron(result, factors.get(qubit, _PAULI["I"]))
    return result


def _axis_operator(axis: np.ndarray) -> np.ndarray:
    return axis[0] * _PAULI["X"] + axis[1] * _PAULI["Y"] + axis[2] * _PAULI["Z"]


def _rotation(generator: np.ndarray, angle: float) -> np.ndarray:
    dim = generator.shape[0]
    return (
        np.cos(float(angle) / 2.0) * np.eye(dim, dtype=np.complex128)
        - 1j * np.sin(float(angle) / 2.0) * generator
    )


def balanced_mixer_matrix(program: QuaRKProgram, reservoir_index: int) -> np.ndarray:
    """Materialize one frozen balanced mixer for the small CPU oracle."""

    parameters = program.reservoirs
    if not isinstance(parameters, BalancedReservoirParameters):
        raise TypeError("balanced_mixer_matrix requires balanced reservoir parameters.")
    r = int(reservoir_index)
    n = program.num_qubits
    unitary = np.eye(1 << n, dtype=np.complex128)
    for qubit in range(n):
        generator = _tensor_operator(
            n, {qubit: _axis_operator(parameters.local_axes[r, qubit])}
        )
        unitary = _rotation(generator, parameters.local_angles[r, qubit]) @ unitary

    matchings = cycle_matchings(program.topology)
    for matching_index in parameters.matching_orders[r]:
        for edge_index in matchings[int(matching_index)]:
            left, right = program.topology.edges[edge_index]
            generator = _tensor_operator(
                n,
                {
                    left: _axis_operator(parameters.edge_axes_left[r, edge_index]),
                    right: _axis_operator(parameters.edge_axes_right[r, edge_index]),
                },
            )
            unitary = _rotation(generator, parameters.edge_angles[r, edge_index]) @ unitary
    return unitary


def _input_unitary(program: QuaRKProgram, projected_input: np.ndarray) -> np.ndarray:
    values = np.asarray(projected_input, dtype=np.float64)
    if program.angle_map == "linear":
        angles = program.angle_scale * values
    elif program.angle_map == "tanh":
        angles = program.angle_scale * np.tanh(values)
    else:
        raise ValueError(f"Unsupported angle_map {program.angle_map!r}.")
    unitary = np.eye(1 << program.num_qubits, dtype=np.complex128)
    for qubit, angle in enumerate(angles):
        generator = _tensor_operator(program.num_qubits, {qubit: _PAULI["Y"]})
        unitary = _rotation(generator, float(angle)) @ unitary
    return unitary


def balanced_density_matrices(program: QuaRKProgram, windows: np.ndarray) -> np.ndarray:
    """Evolve the exact mixed state with no history truncation or branch pruning."""

    parameters = program.reservoirs
    if not isinstance(parameters, BalancedReservoirParameters):
        raise TypeError("balanced_density_matrices requires balanced reservoir parameters.")
    windows = program.validate_windows(windows)
    projected = windows @ program.projection.matrix
    N = int(windows.shape[0])
    R = program.num_reservoirs
    dim = 1 << program.num_qubits
    plus = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)
    rho_plus = np.outer(plus, plus.conj())
    mixers = tuple(balanced_mixer_matrix(program, r) for r in range(R))
    states = np.empty((N, R, dim, dim), dtype=np.complex128)

    for sample in range(N):
        input_unitaries = tuple(
            _input_unitary(program, projected[sample, step])
            for step in range(program.window_length)
        )
        for branch in range(R):
            rho = rho_plus.copy()
            mixer = mixers[branch]
            lam = float(parameters.reset_rates[branch])
            for input_unitary in input_unitaries:
                driven = mixer @ input_unitary
                rho = driven @ rho @ driven.conj().T
                rho = lam * rho + (1.0 - lam) * rho_plus
            states[sample, branch] = rho
    return states
