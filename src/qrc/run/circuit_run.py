from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from qiskit import QuantumCircuit

from qiskit_aer import AerSimulator
from qiskit import transpile

from src.qrc.circuits.configs import BaseQRConfig

from pathlib import Path
import pickle


def _to_array(dm_obj):
    if hasattr(dm_obj, "data"):
        return np.asarray(dm_obj.data, dtype=complex)
    return np.asarray(dm_obj, dtype=complex)


@dataclass
class Results(ABC):
    cfg: BaseQRConfig

    @abstractmethod
    def save(self, file: str | Path) -> None:
        """Save results object to a file."""
        ...

    @staticmethod
    @abstractmethod
    def load(file: str | Path) -> "Results":
        """Load a results object from a file."""
        ...


@dataclass
class ExactResults(Results):
    """
    This must contain exact results which are to be exact statevectors obtained after running each circuit in each pub.
    Concretely, having a list of pubs = [pub_i : 1 <= i <= N], where each pub_i = (qc, params_i) with params_i is a
    matrix of shape (R, num_params) (R reservoirs), results must be an array of shape (N, R, 2**n, 2**n), which
    stores the different states prepared after running each circuit.
    """
    states: np.ndarray
    cfg: BaseQRConfig

    def save(self, file: str | Path) -> None:
        """Serialize ExactResults (states + cfg) to disk."""
        path = Path(file)
        with path.open("wb") as f:
            # Use an explicit dict for future-proofing instead of dumping self directly.
            pickle.dump(
                {
                    "cls": "ExactResults",
                    "states": self.states,
                    "cfg": self.cfg,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @staticmethod
    def load(file: str | Path) -> "ExactResults":
        """Load ExactResults from a file created by `save`."""
        path = Path(file)
        with path.open("rb") as f:
            obj = pickle.load(f)

        # Support both: (a) whole-object pickle, (b) dict format above
        if isinstance(obj, ExactResults):
            return obj

        if isinstance(obj, dict) and obj.get("cls") == "ExactResults":
            states = obj["states"]
            cfg = obj["cfg"]
            return ExactResults(states=states, cfg=cfg)

        raise TypeError(f"File {file} does not contain a valid ExactResults object.")


class BaseCircuitsRunner(ABC):
    """
    Interface setting the logic of how we run the circuits. This will be implemented by ExactCircuitRunner.
    The results of these are to be of the shape Results.
    """

    @abstractmethod
    def run_pubs(self, **kwargs) -> Results:
        ...


class ExactAerCircuitsRunner(BaseCircuitsRunner):
    """
    Interface setting the logic of how we run the circuits. This is to be implemented by two classes, ExactCircuitRunner
    and ShadowCircuitRunner. The results of these
    """

    def __init__(self, cfg: BaseQRConfig):
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

        N = len(pubs)
        if N == 0:
            raise ValueError("pubs must be non-empty")

        # All pubs share same R
        R = int(pubs[0][1].shape[0])

        # We save ONLY the explain qubits (reservoir qubits 0..n-1)
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

            # IMPORTANT: param_values columns follow:
            #   param_order = list(J) + list(h_x) + list(h_z) + [lam]
            # NOT qc.parameters order. So we rebuild that same order from metadata.
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

            # Save reduced density matrix on reservoir qubits
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
            bind = {}
            for j, nm in enumerate(order_names):
                bind[tparam_by_name[nm]] = vals[:, j].tolist()  # <-- length R, NOT scalar
            binds_list.append(bind)

        # Single job for all circuits (each with R parameterizations)
        job = self.backend.run(
            circuits,
            parameter_binds=binds_list,
            shots=1,
            seed_simulator=seed_simulator,
        )
        result = job.result()

        # Aer flattens results as: circuit0 param0..paramR-1, circuit1 param0..paramR-1, ...
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

                dm_arr = np.asarray(dm.data if hasattr(dm, "data") else dm, dtype=complex)
                if dm_arr.shape != (dim_res, dim_res):
                    raise ValueError(f"Got DM shape {dm_arr.shape}, expected {(dim_res, dim_res)} at (i={i}, r={r})")

                states[i, r] = dm_arr

        return ExactResults(states=states, cfg=self.cfg)
