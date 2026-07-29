"""Temporary boundary helpers for legacy Qiskit circuit construction.

This module is intentionally outside :mod:`src.core`.  It converts an immutable
``QuaRKProgram`` into the PUB representation consumed by the independently
implemented Aer oracle while the experiment configurations are migrated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.program import QuaRKProgram


@dataclass
class LegacyTopology:
    num_qubits: int
    edges: tuple[tuple[int, int], ...]


@dataclass
class LegacyQRCConfig:
    input_dim: int
    num_qubits: int
    projection: np.ndarray
    topology: LegacyTopology
    seed: int = 0


def legacy_config(program: QuaRKProgram) -> LegacyQRCConfig:
    return LegacyQRCConfig(
        input_dim=program.input_dim,
        num_qubits=program.num_qubits,
        projection=program.projection.matrix,
        topology=LegacyTopology(
            num_qubits=program.num_qubits,
            edges=program.topology.edges,
        ),
    )


def _angle_positioning(program: QuaRKProgram):
    if program.angle_map == "linear":
        return lambda z: [program.angle_scale * z[index] for index in range(len(z))]

    def tanh(z):
        values = []
        for item in z:
            exp_twice = (2 * item).exp()
            values.append(program.angle_scale * (1 - 2 / (exp_twice + 1)))
        return values

    return tanh


def template_pubs(program: QuaRKProgram, windows: np.ndarray):
    """Build one legacy template PUB without resampling program parameters."""

    from src.backends.aer.circuits import CircuitFactory

    cfg = legacy_config(program)
    circuit, z_steps = CircuitFactory.instantiateFullIsingRingEvolutionTemplate(
        qrc_cfg=cfg,
        angle_positioning=_angle_positioning(program),
        w=program.window_length,
    )
    projected = np.asarray(windows, dtype=float) @ program.projection.matrix
    injection = projected.reshape(projected.shape[0], -1)
    reservoir_values = np.concatenate(
        [
            program.reservoirs.zz,
            program.reservoirs.x_fields,
            program.reservoirs.z_fields,
            program.reservoirs.reset_rates[:, None],
        ],
        axis=1,
    )
    N = int(injection.shape[0])
    R = program.num_reservoirs
    values = np.concatenate(
        [
            np.broadcast_to(injection[:, None, :], (N, R, injection.shape[1])),
            np.broadcast_to(
                reservoir_values[None, :, :],
                (N, R, reservoir_values.shape[1]),
            ),
        ],
        axis=2,
    )
    injection_parameters = [parameter for step in z_steps for parameter in step]
    circuit.metadata["param_order"] = (
        injection_parameters
        + list(circuit.metadata["J"])
        + list(circuit.metadata["h_x"])
        + list(circuit.metadata["h_z"])
        + [circuit.metadata["lam"]]
    )
    return cfg, [(circuit, values)]
