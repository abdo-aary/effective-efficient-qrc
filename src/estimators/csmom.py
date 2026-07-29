"""Local-Pauli classical shadows with median-of-means aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.capabilities import EstimatorKind
from src.core.program import QuaRKProgram
from src.core.requests import ExecutionSpec
from src.core.results import FeatureBatch

X_BASIS = np.uint8(0)
Y_BASIS = np.uint8(1)
Z_BASIS = np.uint8(2)
_BASIS_BY_PAULI = {"X": X_BASIS, "Y": Y_BASIS, "Z": Z_BASIS}


@dataclass(frozen=True)
class ShadowSnapshots:
    """Local measurement bases and signs.

    Both arrays use shape ``(N, R, S, n)``.  Bases are encoded as X=0, Y=1,
    Z=2 and outcomes are signs in ``{-1,+1}``, in qubit order ``0..n-1``.
    """

    bases: np.ndarray
    outcomes: np.ndarray
    trajectory_suffix_starts: np.ndarray | None = None
    trajectory_reset_counts: np.ndarray | None = None

    def __post_init__(self) -> None:
        bases = np.asarray(self.bases, dtype=np.uint8)
        outcomes = np.asarray(self.outcomes, dtype=np.int8)
        if bases.ndim != 4 or bases.shape != outcomes.shape:
            raise ValueError(
                "Shadow bases and outcomes must have the same shape (N,R,S,n)."
            )
        if np.any(bases > 2):
            raise ValueError("Shadow basis codes must lie in {0,1,2}.")
        if np.any((outcomes != -1) & (outcomes != 1)):
            raise ValueError("Shadow outcomes must lie in {-1,+1}.")
        suffix_starts = self.trajectory_suffix_starts
        reset_counts = self.trajectory_reset_counts
        if (suffix_starts is None) != (reset_counts is None):
            raise ValueError(
                "Trajectory suffix starts and reset counts must be provided together."
            )
        if suffix_starts is not None:
            suffix_starts = np.asarray(suffix_starts, dtype=np.int32)
            reset_counts = np.asarray(reset_counts, dtype=np.int32)
            if suffix_starts.shape != bases.shape[:3]:
                raise ValueError(
                    "Trajectory suffix starts must have shape (N,R,S)."
                )
            if reset_counts.shape != bases.shape[:3]:
                raise ValueError(
                    "Trajectory reset counts must have shape (N,R,S)."
                )
            if np.any(suffix_starts < 0) or np.any(reset_counts < 0):
                raise ValueError("Trajectory provenance values must be non-negative.")
        object.__setattr__(self, "bases", bases)
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "trajectory_suffix_starts", suffix_starts)
        object.__setattr__(self, "trajectory_reset_counts", reset_counts)

    @property
    def snapshots(self) -> int:
        return int(self.bases.shape[2])


def deterministic_block_ids(snapshots: int, median_blocks: int) -> np.ndarray:
    """Assign consecutive snapshots to deterministic, near-equal MoM blocks."""

    S = int(snapshots)
    G = int(median_blocks)
    if S < 1:
        raise ValueError("snapshots must be positive.")
    if not (1 <= G <= S):
        raise ValueError(f"median_blocks must satisfy 1 <= G <= S, got G={G}, S={S}.")
    block_ids = np.empty(S, dtype=np.int64)
    for block, indices in enumerate(np.array_split(np.arange(S), G)):
        block_ids[indices] = block
    return block_ids


def reconstruct_csmom(
    snapshots: ShadowSnapshots,
    observable_labels: tuple[str, ...],
    *,
    median_blocks: int,
) -> np.ndarray:
    """Reconstruct ordered Pauli expectations from local-Pauli snapshots."""

    bases = snapshots.bases
    outcomes = snapshots.outcomes
    N, R, S, n = bases.shape
    block_ids = deterministic_block_ids(S, median_blocks)
    K = len(observable_labels)
    estimates = np.empty((N, R, K), dtype=np.float64)

    for k, label in enumerate(observable_labels):
        if len(label) != n or set(label) - set("IXYZ"):
            raise ValueError(f"Unsupported Pauli label {label!r} for n={n}.")
        support = [
            (q, _BASIS_BY_PAULI[label[n - 1 - q]])
            for q in range(n)
            if label[n - 1 - q] != "I"
        ]
        if not support:
            estimates[:, :, k] = 1.0
            continue

        matched = np.ones((N, R, S), dtype=bool)
        signs = np.ones((N, R, S), dtype=np.int8)
        for q, basis in support:
            matched &= bases[:, :, :, q] == basis
            signs *= outcomes[:, :, :, q]
        single_snapshot = (
            matched.astype(np.float64) * signs.astype(np.float64) * (3.0 ** len(support))
        )
        group_means = np.empty((N, R, int(median_blocks)), dtype=np.float64)
        for block in range(int(median_blocks)):
            group_means[:, :, block] = single_snapshot[:, :, block_ids == block].mean(axis=2)
        estimates[:, :, k] = np.median(group_means, axis=2)
    return estimates


@dataclass(frozen=True)
class CSMoMFeatureEstimator:
    """Estimate local Pauli features with classical shadows and MoM."""

    snapshots: int
    median_blocks: int
    measurement_ensemble: str = "local_pauli"
    retain_raw_snapshots: bool = False

    def __post_init__(self) -> None:
        S = int(self.snapshots)
        G = int(self.median_blocks)
        deterministic_block_ids(S, G)
        if self.measurement_ensemble != "local_pauli":
            raise ValueError("Only measurement_ensemble='local_pauli' is supported.")
        object.__setattr__(self, "snapshots", S)
        object.__setattr__(self, "median_blocks", G)

    @property
    def kind(self) -> EstimatorKind:
        return EstimatorKind.CSMOM

    def estimate(
        self,
        program: QuaRKProgram,
        windows: np.ndarray,
        backend: object,
        execution: ExecutionSpec | None = None,
    ) -> FeatureBatch:
        execution = execution or ExecutionSpec()
        backend.capabilities.require_estimator(self.kind)  # type: ignore[attr-defined]
        windows = program.validate_windows(windows)
        compiled = backend.compile(program, self)  # type: ignore[attr-defined]
        return backend.execute(compiled, windows, execution)  # type: ignore[attr-defined]
