"""src.qrc.run.circuit_run

Runners and result containers for executing QRC circuits.

This module defines:

- :class:`Results` and concrete subclasses such as :class:`ExactResults` that store
  simulator outputs in a consistent shape.
- :class:`BaseCircuitsRunner`, an interface for executing a list of PUBs
  (parameterized circuit + parameter value matrix).
- :class:`ExactAerCircuitsRunner`, an Aer-based runner that executes PUBs using
  ``AerSimulator(method="density_matrix")`` and returns reduced density matrices
  on the reservoir subsystem.

A **PUB** is represented as a ``(qc, param_values)`` tuple where:

- ``qc`` is a Qiskit :class:`~qiskit.QuantumCircuit` that is parameterized by the
  reservoir parameters (and typically also injection parameters already bound by
  the circuit factory).
- ``param_values`` is a NumPy array of shape ``(R, P)`` that provides *R* distinct
  reservoir parameterizations for the same circuit (spatial multiplexing). The
  column order is **not** ``qc.parameters`` order: it is rebuilt from circuit
  metadata to ensure stable alignment after transpilation.

The runner expects the circuit metadata keys:

- ``qc.metadata["J"]``: iterable of coupling parameters (two-qubit ZZ weights)
- ``qc.metadata["h_x"]``: iterable of local X-field parameters
- ``qc.metadata["h_z"]``: iterable of local Z-field parameters
- ``qc.metadata["lam"]``: contraction parameter

These keys are produced by the circuit factory in :mod:`src.qrc.circuits`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from src.qrc.circuits.configs import BaseQRConfig

import pickle


def _to_array(dm_obj) -> np.ndarray:
    """Convert a Qiskit density-matrix-like object to a complex NumPy array.

    Parameters
    ----------
    dm_obj : Any
        A Qiskit object that may expose a ``.data`` attribute (e.g., Aer density matrix),
        or an array-like object.

    Returns
    -------
    np.ndarray
        Complex-valued array representation of the density matrix.
    """
    if hasattr(dm_obj, "data"):
        return np.asarray(dm_obj.data, dtype=complex)
    return np.asarray(dm_obj, dtype=complex)


@dataclass
class Results(ABC):
    """Abstract container for circuit execution outputs.

    Attributes
    ----------
    cfg : BaseQRConfig
        Configuration used to build/run the circuits (topology, projection, qubits, etc.).
    states : np.ndarray
        Array storing simulator outputs. The exact meaning/shape depends on the subclass.
    """

    cfg: BaseQRConfig
    states: np.ndarray

    @abstractmethod
    def save(self, file: str | Path) -> None:
        """Serialize this results object to disk.

        Parameters
        ----------
        file : str or pathlib.Path
            Destination path.

        Notes
        -----
        Subclasses should define a forward-compatible serialization format.
        """
        ...

    @staticmethod
    @abstractmethod
    def load(file: str | Path) -> "Results":
        """Load a results object from disk.

        Parameters
        ----------
        file : str or pathlib.Path
            Path to a file created by :meth:`save`.

        Returns
        -------
        Results
            A reconstructed results object.
        """
        ...


@dataclass
class ExactResults(Results):
    """Exact (density-matrix) results for a PUBS dataset.

    This class stores **reduced** density matrices on the reservoir subsystem
    for each input window and each reservoir draw.

    Let ``pubs = [(qc_i, vals_i)]_{i=1..N}``, where:
    - each ``vals_i`` has shape ``(R, P)`` with *R* reservoir parameterizations,
    - ``qc_i`` is parameterized by reservoir parameters.

    Then this class stores:

    - ``states`` of shape ``(N, R, 2**n, 2**n)``, where ``n = cfg.num_qubits``.

    Parameters
    ----------
    states : np.ndarray
        Complex array of reduced density matrices, shape ``(N, R, 2**n, 2**n)``.
    cfg : BaseQRConfig
        Configuration associated with these results.
    """

    states: np.ndarray
    cfg: BaseQRConfig

    def save(self, file: str | Path) -> None:
        """Serialize :class:`ExactResults` to disk.

        Parameters
        ----------
        file : str or pathlib.Path
            Output file path.

        Notes
        -----
        The serialization uses a small dict payload (class name, states, cfg)
        to remain future-proof if the dataclass layout changes.
        """
        path = Path(file)
        with path.open("wb") as f:
            pickle.dump(
                {"cls": "ExactResults", "states": self.states, "cfg": self.cfg},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @staticmethod
    def load(file: str | Path) -> "ExactResults":
        """Load :class:`ExactResults` from disk.

        Parameters
        ----------
        file : str or pathlib.Path
            Path created by :meth:`save`.

        Returns
        -------
        ExactResults
            Reconstructed results instance.

        Raises
        ------
        TypeError
            If the file contents are not recognized as ``ExactResults``.
        """
        path = Path(file)
        with path.open("rb") as f:
            obj = pickle.load(f)

        # Support both: whole-object pickle and dict-based payload
        if isinstance(obj, ExactResults):
            return obj

        if isinstance(obj, dict) and obj.get("cls") == "ExactResults":
            return ExactResults(states=obj["states"], cfg=obj["cfg"])

        raise TypeError(f"File {file} does not contain a valid ExactResults object.")


class BaseCircuitsRunner(ABC):
    """Interface for executing PUBS datasets.

    A runner takes a list of PUBs (parameterized circuits + parameter matrices)
    and returns a :class:`Results` object.
    """

    @abstractmethod
    def run_pubs(self, **kwargs) -> Results:
        """Execute a list of PUBs and return results.

        Returns
        -------
        Results
            Concrete results object (e.g., :class:`ExactResults`).
        """
        ...


class ExactAerCircuitsRunner(BaseCircuitsRunner):
    """Execute PUBS datasets using Qiskit Aer in density-matrix mode.

    The runner:

    1. Configures an :class:`~qiskit_aer.AerSimulator` backend with ``method="density_matrix"``.
    2. For each PUB, transpiles the circuit, ensures a density-matrix save instruction exists,
       and prepares a parameter-binds mapping.
    3. Executes all circuits in a single Aer job using ``parameter_binds``.
    4. Extracts reduced density matrices (reservoir qubits only) into an array
       of shape ``(N, R, 2**n, 2**n)``.

    Notes
    -----
    **Parameter alignment**:

    The expected column order of each ``vals`` matrix is:

    ``list(J) + list(h_x) + list(h_z) + [lam]``

    where these parameter vectors/objects are retrieved from ``qc.metadata``.
    This is intentionally *not* ``qc.parameters`` order (which can change under transpilation).

    **GPU execution**:

    If Aer is built with GPU support and exposes ``available_devices()``,
    passing ``device="GPU"`` to :meth:`run_pubs` will request GPU execution.
    The test suite should skip gracefully if GPU is unavailable.
    """

    def __init__(self, cfg: BaseQRConfig):
        """Construct the runner.

        Parameters
        ----------
        cfg : BaseQRConfig
            Circuit configuration (number of reservoir qubits, etc.).
        """
        self.cfg = cfg
        self.backend = AerSimulator(method="density_matrix")

    def run_pubs(
        self,
        pubs: List[Tuple[QuantumCircuit, np.ndarray]],
        max_threads: Optional[int] = None,
        seed_simulator: int = 0,
        optimization_level: int = 1,
        device: str = "CPU",
        max_parallel_threads: Optional[int] = None,
        max_parallel_experiments: Optional[int] = None,
        max_parallel_shots: Optional[int] = None,
    ) -> ExactResults:
        """Run a PUBS dataset and return exact reduced density matrices.

        Parameters
        ----------
        pubs : list[tuple[QuantumCircuit, np.ndarray]]
            List of PUBs. Each element is ``(qc, vals)`` where:

            - ``qc`` is a parameterized circuit with required metadata keys
              ``J``, ``h_x``, ``h_z``, ``lam``.
            - ``vals`` is a float array of shape ``(R, P)`` containing *R* parameterizations.

        max_threads : int, optional
            Requested maximum threads for Aer. If ``None`` or < 1, uses Aer convention
            ``0`` meaning “use all available”.
        seed_simulator : int, default=0
            Seed passed to Aer simulator.
        optimization_level : int, default=1
            Transpilation optimization level.
        device : {"CPU","GPU"}, default="CPU"
            Aer execution device. GPU requires Aer GPU support.
        max_parallel_threads : int, optional
            Aer parallelism option.
        max_parallel_experiments : int, optional
            Aer parallelism option.
        max_parallel_shots : int, optional
            Aer parallelism option.

        Returns
        -------
        ExactResults
            Reduced reservoir density matrices, shape ``(N, R, 2**n, 2**n)``.

        Raises
        ------
        ValueError
            If PUB formatting is inconsistent, metadata is missing, or dimensions mismatch.
        KeyError
            If Aer result objects do not contain a density matrix payload.

        Notes
        -----
        This runner always requests ``shots=1`` because it uses deterministic density-matrix
        simulation; repeated shots do not change the returned state.
        """
        N = len(pubs)
        if N == 0:
            raise ValueError("pubs must be non-empty")

        # All pubs share same R
        R = int(pubs[0][1].shape[0])

        # We save ONLY the reservoir qubits 0..n-1
        n_res = int(self.cfg.num_qubits)
        dim_res = 1 << n_res

        # Aer parallelism options
        if max_threads is None or max_threads < 1:
            max_threads = 0  # Aer convention: 0 => use all available
        if max_parallel_threads is None:
            max_parallel_threads = max_threads
        if max_parallel_experiments is None:
            max_parallel_experiments = max_parallel_threads
        if max_parallel_shots is None:
            max_parallel_shots = max_parallel_threads

        self.backend.set_options(
            device=device,
            max_parallel_threads=max_parallel_threads,
            max_parallel_experiments=max_parallel_experiments,
            max_parallel_shots=max_parallel_shots,
        )

        circuits = []
        binds_list = []

        for i, (qc, vals) in enumerate(pubs):
            if vals.ndim != 2:
                raise ValueError(f"pub[{i}] params must be 2D (R,P). Got {vals.shape}")
            if vals.shape[0] != R:
                raise ValueError(f"pub[{i}] has R={vals.shape[0]} but expected R={R}")

            # Column order is reconstructed from metadata, not qc.parameters.
            if qc.metadata is None:
                raise ValueError(f"pub[{i}] circuit has no metadata; cannot align columns.")
            try:
                J = list(qc.metadata["J"])
                hx = list(qc.metadata["h_x"])
                hz = list(qc.metadata["h_z"])
                lam = qc.metadata["lam"]
            except KeyError as e:
                raise ValueError(f"pub[{i}] missing metadata key {e}. Need J, h_x, h_z, lam.") from e

            param_order = J + hx + hz + [lam]
            P = len(param_order)
            if vals.shape[1] != P:
                raise ValueError(
                    f"pub[{i}] param cols mismatch: vals has {vals.shape[1]} cols but expected {P} "
                    f"(J,h_x,h_z,lam from qc.metadata)."
                )

            qc_work = qc.copy()

            # Ensure a save_density_matrix instruction exists.
            has_save_dm = any(getattr(inst.operation, "name", "") == "save_density_matrix" for inst in qc_work.data)
            if not has_save_dm:
                qc_work.save_density_matrix(qubits=list(range(n_res)))

            tqc = transpile(qc_work, backend=self.backend, optimization_level=optimization_level)
            circuits.append(tqc)

            # Map parameters by NAME (robust if Parameter objects differ post-transpile)
            tparam_by_name = {p.name: p for p in list(tqc.parameters)}
            order_names = [p.name for p in param_order]

            missing = [nm for nm in order_names if nm not in tparam_by_name]
            if missing:
                raise ValueError(f"pub[{i}] transpiled circuit missing parameters {missing}")

            # Vectorized binds: each param -> list length R
            bind = {tparam_by_name[nm]: vals[:, j].tolist() for j, nm in enumerate(order_names)}
            binds_list.append(bind)

        job = self.backend.run(
            circuits,
            parameter_binds=binds_list,
            shots=1,
            seed_simulator=seed_simulator,
        )
        result = job.result()

        # Aer flattens: circuit0 param0..paramR-1, circuit1 param0..paramR-1, ...
        states = np.empty((N, R, dim_res, dim_res), dtype=complex)

        for i in range(N):
            for r in range(R):
                k = i * R + r
                data_k = result.data(k)

                dm = data_k.get("density_matrix", None)
                if dm is None:
                    # fallback: find any dm key
                    for kk, vv in data_k.items():
                        if "density_matrix" in kk:
                            dm = vv
                            break
                if dm is None:
                    raise KeyError(f"No density_matrix found in result.data({k}). Keys={list(data_k.keys())}")

                dm_arr = _to_array(dm)
                if dm_arr.shape != (dim_res, dim_res):
                    raise ValueError(f"Got DM shape {dm_arr.shape}, expected {(dim_res, dim_res)} at (i={i}, r={r})")

                states[i, r] = dm_arr

        return ExactResults(states=states, cfg=self.cfg)
