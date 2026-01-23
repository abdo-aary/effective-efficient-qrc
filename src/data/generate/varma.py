from __future__ import annotations

from typing import Literal, Optional, Sequence, List

import numpy as np

from .beta_mixing import BetaMixingGenerator


InnovationKind = Literal["normal", "student_t"]


class VARMAGenerator(BetaMixingGenerator):
    """Stable VARMA(p,q) generator as in Appendix D.4.1.

    Model:
        Z_t = sum_{i=1}^p Phi_i Z_{t-i} + eps_t + sum_{j=1}^q Theta_j eps_{t-j}
        X_t = tanh(Z_t)  (applied by ensure_bounded(mode="tanh") in BetaMixingGenerator)

    Stability construction (sufficient condition):
        Draw random directions U_i with ||U_i||_2 = 1 (spectral norm),
        draw weights a_i >= 0 with sum a_i = gamma < 1,
        set Phi_i = a_i U_i  =>  sum ||Phi_i||_2 = gamma < 1,
        which implies det(I - sum Phi_i z^i) != 0 for all |z| <= 1.

    MA construction:
        Theta_0 = I, and Theta_j = b_j V_j with ||V_j||_2 = 1 and
        b_j = eta * rho^(j-1), rho in (0,1).

    Notes
    -----
    - This class returns the *unbounded* path Z of shape (T,d). Bounding into (-1,1)^d
      is handled by BetaMixingGenerator.ensure_bounded (typically mode="tanh"), matching
      the paper's X_t = tanh(Z_t).
    - The BetaMixingGenerator's `burn_in` already plays the role of the VARMA burn-in:
      windows start at index `burn_in`, so the initial transient is not used for training.
    """

    def __init__(
        self,
        *,
        # VARMA orders
        p: int,
        q: int,
        # stability + MA amplitude controls
        gamma: float = 0.7,
        eta: float = 0.5,
        rho: float = 0.5,
        # innovation distribution
        innovation: InnovationKind = "normal",
        innovation_scale: float = 1.0,
        student_df: float = 5.0,
        # optional: provide explicit matrices instead of random generation
        Phi: Optional[Sequence[np.ndarray]] = None,
        Theta: Optional[Sequence[np.ndarray]] = None,
        # caching behavior
        regenerate_params_each_call: bool = True,
        # BetaMixingGenerator args
        s: int,
        burn_in: int = 200,
        bounded: str = "tanh",
        clip: float = 0.999,
        seed: int = 0,
    ):
        super().__init__(s=s, burn_in=burn_in, bounded=bounded, clip=clip, seed=seed)

        self.p = int(p)
        self.q = int(q)
        if self.p <= 0:
            raise ValueError("p must be >= 1 for VARMA(p,q).")
        if self.q < 0:
            raise ValueError("q must be >= 0 for VARMA(p,q).")

        self.gamma = float(gamma)
        if not (0.0 < self.gamma < 1.0):
            raise ValueError("gamma must be in (0,1) to guarantee stability via norm budget.")

        self.eta = float(eta)
        if self.eta < 0:
            raise ValueError("eta must be >= 0.")

        self.rho = float(rho)
        if not (0.0 < self.rho < 1.0):
            raise ValueError("rho must be in (0,1).")

        self.innovation = innovation
        self.innovation_scale = float(innovation_scale)
        if self.innovation_scale <= 0:
            raise ValueError("innovation_scale must be > 0.")
        self.student_df = float(student_df)
        if self.innovation == "student_t" and self.student_df <= 2:
            raise ValueError("student_df must be > 2 (finite variance) for the setting in the paper.")

        self.Phi_user = Phi
        self.Theta_user = Theta

        # Early validation (doesn't require knowing d)
        if self.Phi_user is not None and len(self.Phi_user) != self.p:
            raise ValueError(f"Provided Phi must have length p={self.p}, got {len(self.Phi_user)}.")
        if self.Theta_user is not None and len(self.Theta_user) != self.q:
            raise ValueError(f"Provided Theta must have length q={self.q}, got {len(self.Theta_user)}.")

        # Optional: also ensure each provided matrix is 2D square (still doesn't require d)
        for i, M in enumerate(self.Phi_user or []):
            M = np.asarray(M)
            if M.ndim != 2 or M.shape[0] != M.shape[1]:
                raise ValueError(f"Phi[{i}] must be a square 2D matrix, got shape {M.shape}.")
        for j, M in enumerate(self.Theta_user or []):
            M = np.asarray(M)
            if M.ndim != 2 or M.shape[0] != M.shape[1]:
                raise ValueError(f"Theta[{j}] must be a square 2D matrix, got shape {M.shape}.")

        # populated after simulate_path
        self.Phi_: Optional[List[np.ndarray]] = None
        self.Theta_: Optional[List[np.ndarray]] = None

        self.regenerate_params_each_call = bool(regenerate_params_each_call)

    @staticmethod
    def _spectral_norm(A: np.ndarray) -> float:
        # Largest singular value
        return float(np.linalg.svd(A, compute_uv=False)[0])

    def _draw_unit_spectral_direction(self, *, d: int, rng: np.random.Generator) -> np.ndarray:
        A = rng.normal(size=(d, d))
        nrm = self._spectral_norm(A)
        if nrm == 0.0:
            # extremely unlikely; fallback
            A = np.eye(d)
            nrm = 1.0
        return A / nrm  # ||.||_2 = 1

    def _draw_stable_Phi(self, *, d: int, rng: np.random.Generator) -> List[np.ndarray]:
        # directions
        U = [self._draw_unit_spectral_direction(d=d, rng=rng) for _ in range(self.p)]
        # weights a_i summing to gamma
        w = rng.uniform(0.0, 1.0, size=(self.p,))
        w_sum = float(np.sum(w))
        if w_sum == 0.0:
            w = np.ones((self.p,), dtype=float)
            w_sum = float(self.p)
        a = (self.gamma * w) / w_sum  # sum a = gamma
        return [float(a[i]) * U[i] for i in range(self.p)]

    def _draw_Theta(self, *, d: int, rng: np.random.Generator) -> List[np.ndarray]:
        # Theta_0 is implicitly I in the recursion; we store only Theta_1..Theta_q
        Theta = []
        for j in range(1, self.q + 1):
            Vj = self._draw_unit_spectral_direction(d=d, rng=rng)
            bj = self.eta * (self.rho ** (j - 1))
            Theta.append(float(bj) * Vj)
        return Theta

    def _sample_innovations(self, *, T: int, d: int, rng: np.random.Generator) -> np.ndarray:
        if self.innovation == "normal":
            eps = rng.normal(loc=0.0, scale=self.innovation_scale, size=(T, d))
            return eps
        if self.innovation == "student_t":
            # standard_t has Lebesgue density; scale controls dispersion
            eps = rng.standard_t(df=self.student_df, size=(T, d)) * self.innovation_scale
            return eps
        raise ValueError(f"Unknown innovation={self.innovation!r}")

    def simulate_path(self, *, T: int, d: int, rng: np.random.Generator) -> np.ndarray:
        if T <= 0 or d <= 0:
            raise ValueError("T and d must be positive.")

        # Choose (Phi, Theta) either from user or random construction
        if (self.Phi_ is None) or self.regenerate_params_each_call:
            if self.Phi_user is not None:
                Phi = [np.asarray(M, dtype=float) for M in self.Phi_user]
                if len(Phi) != self.p:
                    raise ValueError(f"Provided Phi must have length p={self.p}, got {len(Phi)}.")
                for i, M in enumerate(Phi):
                    if M.shape != (d, d):
                        raise ValueError(f"Phi[{i}] must be shape {(d,d)}, got {M.shape}.")
            else:
                Phi = self._draw_stable_Phi(d=d, rng=rng)

            if self.Theta_user is not None:
                Theta = [np.asarray(M, dtype=float) for M in self.Theta_user]
                if len(Theta) != self.q:
                    raise ValueError(f"Provided Theta must have length q={self.q}, got {len(Theta)}.")
                for j, M in enumerate(Theta):
                    if M.shape != (d, d):
                        raise ValueError(f"Theta[{j}] must be shape {(d,d)}, got {M.shape}.")
            else:
                Theta = self._draw_Theta(d=d, rng=rng)

            self.Phi_ = Phi
            self.Theta_ = Theta

        Phi = self.Phi_
        Theta = self.Theta_
        assert Phi is not None and Theta is not None

        # innovations
        eps = self._sample_innovations(T=T, d=d, rng=rng)

        # recursion
        Z = np.zeros((T, d), dtype=float)
        # For t < max(p,q), the missing past terms are treated as 0 (init at 0),
        # which is exactly the “initialized at 0” burn-in simulation in the paper.
        for t in range(T):
            zt = np.zeros((d,), dtype=float)

            # AR part
            for i in range(1, self.p + 1):
                ti = t - i
                if ti >= 0:
                    zt += Phi[i - 1] @ Z[ti]

            # Theta_0 eps_t with Theta_0 = I
            zt += eps[t]

            # MA part
            for j in range(1, self.q + 1):
                tj = t - j
                if tj >= 0:
                    zt += Theta[j - 1] @ eps[tj]

            Z[t] = zt

        # IMPORTANT: return unbounded Z; bounding (tanh/clip/none) is applied upstream.
        return Z
