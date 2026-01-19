from __future__ import annotations

import pickle
from abc import ABC, abstractmethod

from typing import Sequence

import numpy as np

from src.qrc.circuits.configs import BaseQRConfig
from .circuit_run import Results

from qiskit.quantum_info import Operator, SparsePauliOp


class BaseFeatureMapsRetriever(ABC):
    """
    Base class for retrieving feature maps
    """
    cfg: BaseQRConfig
    fmps: np.ndarray = None
    observables: Sequence[Operator]

    @abstractmethod
    def get_feature_maps(self, results: Results, **kwargs):
        ...

    # ------------------------------------------------------------------
    # Persistence API
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """
        Save the MKLearner (including any precomputed approximations) to disk.

        Parameters
        ----------
        path : str
            Destination path for the pickle file.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "BaseFeatureMapsRetriever":
        """
        Load a previously saved BaseFeatureMapsRetriever from disk.

        Parameters
        ----------
        path : str
            Path to the pickle file created by :meth:`save`.

        Returns
        -------
        MKLearner
            The reconstructed learner instance.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, BaseFeatureMapsRetriever):
            raise TypeError(f"Expected a BaseFeatureMapsRetriever instance in {path!r}, got {type(obj)!r}.")
        return obj


class ExactFeatureMapsRetriever(BaseFeatureMapsRetriever):
    """
    Compute exact feature maps from Results (density matrices).

    Input:
      - results.states: shape (N, R, 2**n, 2**n)

    Output:
      - feature maps: shape (N, R * K) where K = len(observables)
        ordering is: [obs_0..obs_{K-1} for r=0] then [obs_0..obs_{K-1} for r=1] ...
    """

    def __init__(self, cfg: BaseQRConfig, observables: Sequence[Operator | SparsePauliOp]):
        self.cfg = cfg
        self.observables = observables
        self._pauli_cache = None  # list of (xmask:int, zmask:int, ny:int) or None
        self._dense_cache = None  # obs_mat (K,dim,dim) if needed

    @staticmethod
    def _pauli_label(op: Operator | SparsePauliOp) -> str | None:
        """Return a single-Pauli label like 'IXYZ' if op is a 1-term SparsePauliOp; else None."""
        if isinstance(op, SparsePauliOp):
            # generate_k_local_paulis returns single-term ops; accept that case
            if len(op.paulis) == 1:
                return op.paulis[0].to_label()
        return None

    @staticmethod
    def _masks_from_label(label: str) -> tuple[int, int, int]:
        """
        Convert a Pauli label (leftmost = qubit n-1, rightmost = qubit 0) into bitmasks.
        Uses Qiskit little-endian: qubit q corresponds to bit (1<<q).

        Returns: (xmask, zmask, ny) where:
          - xmask has 1s on qubits with X or Y
          - zmask has 1s on qubits with Z or Y
          - ny = number of Y's (global phase i^ny)
        """
        n = len(label)
        xmask = 0
        zmask = 0
        ny = 0
        for q in range(n):
            ch = label[n - 1 - q]  # qubit 0 is rightmost character
            if ch == "X":
                xmask |= (1 << q)
            elif ch == "Z":
                zmask |= (1 << q)
            elif ch == "Y":
                xmask |= (1 << q)
                zmask |= (1 << q)
                ny += 1
        return xmask, zmask, ny

    @staticmethod
    def _bitcount_parity(arr: np.ndarray) -> np.ndarray:
        """Return parity (0/1) of bitcounts for each entry of arr (uint)."""
        if hasattr(np, "bit_count"):
            return (np.bit_count(arr) & 1).astype(np.int8)
        # fallback
        v = np.vectorize(lambda x: int(x).bit_count() & 1, otypes=[np.int8])
        return v(arr)

    def _ensure_pauli_cache(self, n: int):
        labels = []
        for op in self.observables:
            lab = self._pauli_label(op)
            if lab is None:
                self._pauli_cache = None
                return
            if len(lab) != n:
                raise ValueError(f"Observable label length {len(lab)} != num_qubits {n}: {lab!r}")
            labels.append(lab)
        self._pauli_cache = [self._masks_from_label(lab) for lab in labels]

    def _ensure_dense_cache(self):
        # Only build dense matrices if we cannot use Pauli-optimized path.
        if self._dense_cache is None:
            self._dense_cache = np.stack([op.to_matrix() for op in self.observables], axis=0)

    def get_feature_maps(self, results: Results, **kwargs) -> np.ndarray:
        states = np.asarray(results.states)
        if states.ndim != 4:
            raise ValueError(f"Expected states shape (N,R,dim,dim), got {states.shape}")

        N, R, dim1, dim2 = states.shape
        if dim1 != dim2:
            raise ValueError(f"Density matrices must be square; got {dim1}x{dim2}")
        n = int(self.cfg.num_qubits)
        dim = 1 << n
        if dim1 != dim:
            raise ValueError(f"Expected dim=2**n={dim}, got {dim1}. Check cfg.num_qubits vs results.states.")

        K = len(self.observables)
        if K == 0:
            out = np.zeros((N, 0), dtype=np.float64)
            self.fmps = out
            return out

        # Flatten batch: (B,dim,dim) with B=N*R
        rho = states.reshape(N * R, dim, dim)

        # --- Fast path for Pauli strings (SparsePauliOp) ---
        if self._pauli_cache is None and all(isinstance(op, SparsePauliOp) for op in self.observables):
            self._ensure_pauli_cache(n)

        if self._pauli_cache is not None:
            rows = np.arange(dim, dtype=np.uint32)
            feats = np.empty((N * R, K), dtype=np.float64)

            for k, (xmask, zmask, ny) in enumerate(self._pauli_cache):
                cols = rows ^ np.uint32(xmask)

                # vals[b, x] = rho_b[x, x^xmask]
                vals = rho[:, rows, cols]  # (B,dim)

                # phase(x) = i^{ny} * (-1)^{popcount(zmask & x)}
                parity = self._bitcount_parity(np.bitwise_and(rows, np.uint32(zmask)))
                sign = 1.0 - 2.0 * parity.astype(np.float64)  # +1 for even, -1 for odd
                phase = (1j ** ny) * sign  # (dim,)

                exp = np.sum(vals * phase[None, :], axis=1)  # (B,)
                feats[:, k] = exp.real

            fmps = feats.reshape(N, R, K).reshape(N, R * K)
            self.fmps = fmps
            return fmps

        # --- Generic fallback: Tr(rho O) with dense matrices (OK for small n) ---
        self._ensure_dense_cache()
        obs_mat = self._dense_cache  # (K,dim,dim)

        # exp[b,k] = Tr(rho_b @ O_k) = sum_{i,j} rho_{i,j} O_{j,i}
        exp = np.einsum("bij,kji->bk", rho, obs_mat).real  # (B,K)
        fmps = exp.reshape(N, R, K).reshape(N, R * K)

        self.fmps = fmps
        return fmps
