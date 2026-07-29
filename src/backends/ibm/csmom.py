"""Hardware-compatible circuit generation and result decoding for CSMoM."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.backends.ibm.trajectories import HardwareShadowPlan
from src.core.program import QuaRKProgram
from src.estimators.csmom import ShadowSnapshots


@dataclass(frozen=True)
class CircuitGroup:
    key: tuple[int, int, int, tuple[int, ...]]
    snapshot_indices: tuple[int, ...]
    circuit: Any


def _angle(program: QuaRKProgram, value: float) -> float:
    if program.angle_map == "linear":
        return program.angle_scale * float(value)
    return program.angle_scale * float(np.tanh(value))


def build_grouped_shadow_circuits(
    program: QuaRKProgram,
    windows: np.ndarray,
    plan: HardwareShadowPlan,
) -> list[CircuitGroup]:
    """Build one ordinary circuit per unique (state, suffix, basis) group."""

    try:
        from qiskit import ClassicalRegister, QuantumCircuit
    except ImportError as exc:
        from src.core.exceptions import BackendDependencyError

        raise BackendDependencyError("IBM CSMoM circuit construction requires Qiskit.") from exc

    projected = np.asarray(windows, dtype=float) @ program.projection.matrix
    N, R, S = plan.suffix_starts.shape
    grouped: dict[tuple[int, int, int, tuple[int, ...]], list[int]] = defaultdict(list)
    for i in range(N):
        for r in range(R):
            for s in range(S):
                key = (
                    i,
                    r,
                    int(plan.suffix_starts[i, r, s]),
                    tuple(int(value) for value in plan.bases[i, r, s]),
                )
                grouped[key].append(s)

    circuits: list[CircuitGroup] = []
    n = program.num_qubits
    for key, snapshot_indices in grouped.items():
        i, r, start, basis = key
        circuit = QuantumCircuit(n)
        circuit.h(range(n))
        for t in range(start, program.window_length):
            for q in range(n):
                circuit.ry(_angle(program, projected[i, t, q]), q)
            for edge, (q1, q2) in enumerate(program.topology.edges):
                circuit.rzz(float(program.reservoirs.zz[r, edge]), q1, q2)
            for q in range(n):
                circuit.rz(float(program.reservoirs.z_fields[r, q]), q)
            for q in range(n):
                circuit.rx(float(program.reservoirs.x_fields[r, q]), q)
        for q, basis_code in enumerate(basis):
            if basis_code == 0:
                circuit.h(q)
            elif basis_code == 1:
                circuit.sdg(q)
                circuit.h(q)
        measurement = ClassicalRegister(n, "meas")
        circuit.add_register(measurement)
        circuit.measure(range(n), measurement)
        circuit.metadata = {
            "window": i,
            "reservoir": r,
            "suffix_start": start,
            "basis_qorder": list(basis),
            "snapshot_indices": list(snapshot_indices),
        }
        circuits.append(
            CircuitGroup(
                key=key,
                snapshot_indices=tuple(snapshot_indices),
                circuit=circuit,
            )
        )
    return circuits


def counts_to_snapshots(
    *,
    program: QuaRKProgram,
    plan: HardwareShadowPlan,
    groups: list[CircuitGroup],
    counts_by_group: list[dict[str, int]],
) -> ShadowSnapshots:
    if len(groups) != len(counts_by_group):
        raise ValueError("Group/count result length mismatch.")
    N, R, S, n = plan.bases.shape
    outcomes = np.empty((N, R, S, n), dtype=np.int8)
    for group, counts in zip(groups, counts_by_group):
        i, r, _, _ = group.key
        bitstrings: list[str] = []
        for raw, count in sorted(counts.items()):
            compact = str(raw).replace(" ", "")
            bitstrings.extend([compact] * int(count))
        if len(bitstrings) != len(group.snapshot_indices):
            raise ValueError(
                f"Expected {len(group.snapshot_indices)} outcomes for group {group.key}, "
                f"received {len(bitstrings)}."
            )
        for snapshot, bitstring in zip(group.snapshot_indices, bitstrings):
            if len(bitstring) < n:
                bitstring = bitstring.zfill(n)
            for q in range(n):
                bit = int(bitstring[-1 - q])
                outcomes[i, r, snapshot, q] = 1 if bit == 0 else -1
    return ShadowSnapshots(
        bases=plan.bases,
        outcomes=outcomes,
        trajectory_suffix_starts=plan.suffix_starts,
        trajectory_reset_counts=plan.reset_counts,
    )
