"""Pre-sampled reset-channel trajectories for hardware execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.program import QuaRKProgram
from src.core.requests import ExecutionSpec
from src.estimators.csmom import CSMoMFeatureEstimator


@dataclass(frozen=True)
class HardwareShadowPlan:
    """Structural randomness for one IBM CSMoM execution."""

    bases: np.ndarray  # (N,R,S,n)
    suffix_starts: np.ndarray  # (N,R,S), first unitary step to execute
    reset_counts: np.ndarray  # (N,R,S), sampled resets in the full window

    def __post_init__(self) -> None:
        bases = np.asarray(self.bases, dtype=np.uint8)
        suffix = np.asarray(self.suffix_starts, dtype=np.int32)
        reset_counts = np.asarray(self.reset_counts, dtype=np.int32)
        if (
            bases.ndim != 4
            or suffix.shape != bases.shape[:3]
            or reset_counts.shape != bases.shape[:3]
        ):
            raise ValueError("Invalid hardware shadow plan shapes.")
        if np.any(bases > 2):
            raise ValueError("Basis codes must lie in {0,1,2}.")
        object.__setattr__(self, "bases", bases)
        object.__setattr__(self, "suffix_starts", suffix)
        object.__setattr__(self, "reset_counts", reset_counts)


def sample_hardware_shadow_plan(
    program: QuaRKProgram,
    windows: np.ndarray,
    estimator: CSMoMFeatureEstimator,
    execution: ExecutionSpec,
) -> HardwareShadowPlan:
    N = int(windows.shape[0])
    R = program.num_reservoirs
    S = estimator.snapshots
    n = program.num_qubits
    w = program.window_length
    bases = execution.seeds.generator("shadow_bases").integers(
        0, 3, size=(N, R, S, n), dtype=np.uint8
    )
    uniforms = execution.seeds.generator("reset_trajectories").random((N, R, S, w))
    suffix_starts = np.zeros((N, R, S), dtype=np.int32)
    reset_counts = np.zeros((N, R, S), dtype=np.int32)
    for r, lam in enumerate(program.reservoirs.reset_rates):
        reset = uniforms[:, r] >= float(lam)
        reset_counts[:, r] = reset.sum(axis=2)
        for i in range(N):
            for s in range(S):
                reset_steps = np.flatnonzero(reset[i, s])
                suffix_starts[i, r, s] = (
                    int(reset_steps[-1]) + 1 if reset_steps.size else 0
                )
    return HardwareShadowPlan(
        bases=bases,
        suffix_starts=suffix_starts,
        reset_counts=reset_counts,
    )
