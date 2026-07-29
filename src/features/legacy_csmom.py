"""Compatibility retriever for genuine local-Pauli CSMoM features.

New code should use :class:`src.estimators.csmom.CSMoMFeatureEstimator` with a
QuaRK backend. This wrapper only supports stored Hydra configurations that still
run an exact density-matrix simulator before feature retrieval.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from qiskit.quantum_info import Operator, SparsePauliOp

from src.compute.backend import asnumpy, import_cupy, is_cupy_array
from src.core.observables import ObservableSet
from src.core.seeds import SeedBundle
from src.estimators.csmom import ShadowSnapshots, reconstruct_csmom
from src.features.shadows import sample_local_pauli_shadows_density
from src.backends.aer.legacy_runner import ExactExpectationResults, ExactResults
from src.features.legacy_retrievers import BaseFeatureMapsRetriever


class CSFeatureMapsRetriever(BaseFeatureMapsRetriever):
    """Legacy-shaped wrapper around true local-Pauli classical shadows.

    ``shots`` means shared shadow snapshots per state, not independent shots
    per observable. The estimator uses the standard ``3^k`` inverse-channel
    factor and may lie outside ``[-1,1]`` at finite sample sizes.
    """

    def __init__(
        self,
        qrc_cfg,
        observables: Sequence[Operator | SparsePauliOp],
        *,
        default_shots: Optional[int] = None,
        default_n_groups: Optional[int] = None,
        backend: str = "auto",
    ):
        self.qrc_cfg = qrc_cfg
        self.observables = list(observables)
        if not self.observables:
            raise ValueError("observables must be non-empty.")
        self.observable_set = ObservableSet.from_qiskit(self.observables)
        if self.observable_set.num_qubits != int(qrc_cfg.num_qubits):
            raise ValueError("Observable width does not match qrc_cfg.num_qubits.")
        self.default_shots = default_shots
        self.default_n_groups = default_n_groups
        self.backend = str(backend)
        self.fmps = None

    @staticmethod
    def _pick_n_groups(shots: int) -> int:
        return max(1, min(16, int(np.sqrt(shots))))

    def get_feature_maps(
        self,
        results: ExactResults,
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        n_groups: Optional[int] = None,
    ):
        if isinstance(results, ExactExpectationResults):
            raise ValueError(
                "Genuine classical shadows require state or measurement samples; "
                "exact expectation values alone are insufficient."
            )
        if not isinstance(results, ExactResults):
            raise TypeError(f"Expected ExactResults, got {type(results)!r}.")
        shots = self.default_shots if shots is None else shots
        if shots is None:
            raise ValueError("shots must be provided (or set default_shots).")
        shots = int(shots)
        if shots < 1:
            raise ValueError(f"shots must be positive, got {shots}.")
        n_groups = self.default_n_groups if n_groups is None else n_groups
        if n_groups is None:
            n_groups = self._pick_n_groups(shots)
        n_groups = int(n_groups)
        if not (1 <= n_groups <= shots):
            raise ValueError(
                f"n_groups must satisfy 1 <= n_groups <= shots, got {n_groups}, {shots}."
            )

        source_was_cupy = is_cupy_array(results.states)
        states = asnumpy(results.states)
        if states.ndim != 4:
            raise ValueError(f"Expected states shape (N,R,D,D), got {states.shape}.")
        N, R, dim1, dim2 = states.shape
        expected_dim = 1 << int(self.qrc_cfg.num_qubits)
        if dim1 != dim2 or dim1 != expected_dim:
            raise ValueError(
                f"Expected state tail ({expected_dim},{expected_dim}), got ({dim1},{dim2})."
            )

        seeds = SeedBundle.from_root(0 if seed is None else int(seed))
        bases = seeds.generator("shadow_bases").integers(
            0,
            3,
            size=(N, R, shots, int(self.qrc_cfg.num_qubits)),
            dtype=np.uint8,
        )
        snapshots: ShadowSnapshots = sample_local_pauli_shadows_density(
            states,
            bases,
            outcome_rng=seeds.generator("measurement_outcomes"),
        )
        values = reconstruct_csmom(
            snapshots,
            self.observable_set.labels,
            median_blocks=n_groups,
        )
        fmps = values.reshape(N, R * len(self.observables))
        use_cupy = self.backend == "cupy" or (
            self.backend == "auto" and source_was_cupy
        )
        if use_cupy:
            fmps = import_cupy().asarray(fmps)
        self.fmps = fmps
        return fmps
