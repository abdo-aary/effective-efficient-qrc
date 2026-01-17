import numpy as np
from typing import Optional, Sequence

from qiskit.quantum_info import Operator, SparsePauliOp

from .fmp_retriever import BaseFeatureMapsRetriever, ExactFeatureMapsRetriever
from .circuit_run import ExactResults


class CSFeatureMapsRetriever(BaseFeatureMapsRetriever):
    """
    Noisy (classical-shadow-like) feature maps from ExactResults density matrices.

    Input:
      - results.states: (N, R, 2**n, 2**n)

    Output:
      - fmps: (N, R*K) where K=len(observables), same ordering as ExactFeatureMapsRetriever.

    Noise model:
      - Assumes observables are Pauli strings (eigenvalues ±1) or otherwise bounded in [-1,1].
      - For each (i,r,k), we simulate 'shots' ±1 measurements with mean mu = Tr(rho O).
      - Median-of-Means (MoM): split shots into K groups, take median of group means.
    """

    def __init__(
        self,
        cfg,
        observables: Sequence[Operator | SparsePauliOp],
        *,
        default_shots: Optional[int] = None,
        default_n_groups: Optional[int] = None,
    ):
        self.cfg = cfg
        self.observables = list(observables)
        if len(self.observables) == 0:
            raise ValueError("observables must be non-empty.")

        self.default_shots = default_shots
        self.default_n_groups = default_n_groups

        # Reuse the exact retriever to compute μ = Tr(ρ O) with the same ordering/cache.
        self._exact = ExactFeatureMapsRetriever(cfg, self.observables)

    @staticmethod
    def _pick_n_groups(shots: int) -> int:
        # heuristic: ~sqrt(shots) groups, cap at 16
        return max(1, min(16, int(np.sqrt(shots))))

    def get_feature_maps(
        self,
        results: "ExactResults",
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
        n_groups: Optional[int] = None,
    ) -> np.ndarray:
        # ------------------------
        # validate / defaults
        # ------------------------
        if shots is None:
            shots = self.default_shots
        if shots is None:
            raise ValueError("shots must be provided (or set default_shots in constructor).")
        shots = int(shots)
        if shots <= 0:
            raise ValueError(f"shots must be positive, got {shots}.")

        # Basic cfg compatibility check (avoid relying on object equality)
        n = int(self.cfg.num_qubits)
        dim = 1 << n
        states = np.asarray(results.states)
        if states.ndim != 4:
            raise ValueError(f"Expected results.states shape (N,R,dim,dim), got {states.shape}")
        if states.shape[2:] != (dim, dim):
            raise ValueError(f"State dim mismatch: expected {(dim, dim)}, got {states.shape[2:]}")

        N, R = states.shape[0], states.shape[1]
        K = len(self.observables)

        # ------------------------
        # exact expectations μ
        # ------------------------
        mu_flat = self._exact.get_feature_maps(results)   # (N, R*K)
        mu = mu_flat.reshape(N, R, K)

        # For Pauli observables, μ ∈ [-1,1]. Clip for numerical drift / bounded assumptions.
        mu = np.clip(mu, -1.0, 1.0)

        # ------------------------
        # MoM setup
        # ------------------------
        if n_groups is None:
            n_groups = self.default_n_groups
        if n_groups is None:
            n_groups = self._pick_n_groups(shots)

        n_groups = int(n_groups)
        n_groups = max(1, min(n_groups, shots))
        batch_size = max(shots // n_groups, 1)

        # ------------------------
        # simulate group means using Binomial model (exact for ±1 outcomes)
        #
        # If X ∈ {±1} with E[X]=μ, then P(X=+1) = (1+μ)/2.
        # For a batch of size b, sum of +1 outcomes ~ Binomial(b, p),
        # and batch mean = ( (#(+1) - #(−1)) / b ) = (2*count - b)/b.
        # ------------------------
        rng = np.random.default_rng(seed)
        p = (1.0 + mu) / 2.0
        p = np.clip(p, 0.0, 1.0)

        counts = rng.binomial(
            n=batch_size,
            p=p[..., None],
            size=mu.shape + (n_groups,),
        )
        group_means = (2.0 * counts - batch_size) / float(batch_size)  # (N,R,K,G)

        # Median-of-Means aggregation
        cs_est = np.median(group_means, axis=-1)  # (N,R,K)

        fmps = cs_est.reshape(N, R * K).astype(float)
        self.fmps = fmps
        return fmps
