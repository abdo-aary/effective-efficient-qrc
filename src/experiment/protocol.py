"""Typed, deterministic contracts shared by empirical protocol runners."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ChronologicalSplit:
    inner_train_stop: int
    validation_stop: int
    test_stop: int
    test_start: int | None = None

    def __post_init__(self) -> None:
        a, b, c = map(
            int, (self.inner_train_stop, self.validation_stop, self.test_stop)
        )
        test_start = b if self.test_start is None else int(self.test_start)
        if not 0 < a < b <= test_start < c:
            raise ValueError(
                "Split boundaries must satisfy 0 < inner < validation <= test_start < test."
            )
        object.__setattr__(self, "inner_train_stop", a)
        object.__setattr__(self, "validation_stop", b)
        object.__setattr__(self, "test_stop", c)
        object.__setattr__(self, "test_start", test_start)

    @property
    def inner_train(self) -> np.ndarray:
        return np.arange(0, self.inner_train_stop, dtype=np.int64)

    @property
    def validation(self) -> np.ndarray:
        return np.arange(
            self.inner_train_stop, self.validation_stop, dtype=np.int64
        )

    @property
    def outer_train(self) -> np.ndarray:
        return np.arange(0, self.validation_stop, dtype=np.int64)

    @property
    def test(self) -> np.ndarray:
        return np.arange(int(self.test_start), self.test_stop, dtype=np.int64)

    @classmethod
    def for_prefix(cls, *, prefix: int, test_start: int, test_stop: int) -> "ChronologicalSplit":
        prefix = int(prefix)
        if prefix % 5 != 0:
            raise ValueError("E1 prefixes must be divisible by five for an exact 80/20 split.")
        return cls(
            inner_train_stop=4 * prefix // 5,
            validation_stop=prefix,
            test_start=int(test_start),
            test_stop=int(test_stop),
        )

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "inner_train": self.inner_train,
            "validation": self.validation,
            "outer_train": self.outer_train,
            "test": self.test,
        }


@dataclass(frozen=True)
class E1ProtocolSpec:
    protocol_version: str
    experiment_contract: str
    scenario: str
    method: str
    mode: str
    root: int
    num_windows: int
    window_length: int
    stride: int
    num_qubits: int
    num_reservoirs: int
    locality: int
    reset_rate: float
    chunk_size: int
    task_names: tuple[str, ...]
    split: ChronologicalSplit
    prefixes: tuple[int, ...]
    nu_grid: tuple[float, ...]
    xi_grid: tuple[float, ...]
    lambda_grid: tuple[float, ...]
    path_lambda_grid: tuple[float, ...]
    tie_tolerance: float
    path_nu: float
    path_xi: float
    engine: str
    gpu_id: int = 0
    precision: str = "complex128"

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "full"}:
            raise ValueError("mode must be smoke or full.")
        if self.split.test_stop != int(self.num_windows):
            raise ValueError("The split must cover every generated window.")
        if self.precision != "complex128":
            raise ValueError("E1 protocol execution requires complex128.")
        if not self.task_names:
            raise ValueError("At least one task is required.")
        if self.engine != "cupy":
            raise ValueError("E1 exact NVIDIA execution requires the CuPy engine.")
        if self.tie_tolerance < 0.0 or min(self.path_nu, self.path_xi) <= 0.0:
            raise ValueError(
                "Readout tie tolerance must be nonnegative and path parameters positive."
            )
        if any(prefix > self.split.validation_stop for prefix in self.prefixes):
            raise ValueError("Nested prefixes must lie inside outer train.")

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "E1ProtocolSpec":
        data = cfg["data"]
        program = cfg["program"]
        backend = cfg["backend"]
        readout = cfg["readout"]
        split = cfg["split"]
        fixed_contract = {
            "program.topology": (program.get("topology"), "ring"),
            "program.projection": (program.get("projection"), "gaussian_jl"),
            "program.angle_map": (program.get("angle_map"), "tanh"),
            "backend.kind": (backend.get("kind"), "nvidia"),
            "backend.engine": (backend.get("engine"), "cupy"),
            "backend.allow_cpu_fallback": (
                backend.get("allow_cpu_fallback"),
                False,
            ),
            "readout.standardize_features": (
                readout.get("standardize_features"),
                True,
            ),
        }
        for name, (actual, expected) in fixed_contract.items():
            if actual != expected:
                raise ValueError(f"E1 requires {name}={expected!r}, got {actual!r}.")
        return cls(
            protocol_version=str(cfg["protocol_version"]),
            experiment_contract=str(cfg["experiment_contract"]),
            scenario=str(cfg["scenario"]),
            method=str(cfg["method"]),
            mode=str(cfg["mode"]),
            root=int(cfg["root"]),
            num_windows=int(data["N"]),
            window_length=int(data["window_length"]),
            stride=int(data["stride"]),
            num_qubits=int(program["num_qubits"]),
            num_reservoirs=int(program["num_reservoirs"]),
            locality=int(program["locality"]),
            reset_rate=float(program["reset_rate"]),
            chunk_size=int(backend["chunk_size"]),
            task_names=tuple(str(value) for value in data["tasks"]),
            split=ChronologicalSplit(
                int(split["inner_train_stop"]),
                int(split["validation_stop"]),
                int(split["test_stop"]),
                int(split.get("test_start", split["validation_stop"])),
            ),
            prefixes=tuple(int(value) for value in split["prefixes"]),
            nu_grid=tuple(float(value) for value in readout["nu"]),
            xi_grid=tuple(float(value) for value in readout["xi"]),
            lambda_grid=tuple(float(value) for value in readout["lambda_K"]),
            path_lambda_grid=tuple(
                float(value) for value in readout["path_lambda_K"]
            ),
            tie_tolerance=float(readout["tie_tolerance"]),
            path_nu=float(readout["path_nu"]),
            path_xi=float(readout["path_xi"]),
            engine=str(backend["engine"]),
            gpu_id=int(backend["gpu_id"]),
            precision=str(program["precision"]),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["split"] = asdict(self.split)
        return value

    def configuration_hash(
        self,
        *,
        dataset_hash: str,
        program_fingerprint: str,
        resolved_configuration: Mapping[str, Any] | None = None,
    ) -> str:
        payload = {
            "spec": self.to_dict(),
            "dataset_hash": str(dataset_hash),
            "program_fingerprint": str(program_fingerprint),
        }
        if resolved_configuration is not None:
            identity_configuration = dict(resolved_configuration)
            # These fields control orchestration and authorization only. They
            # must not split otherwise identical feature and analysis stages
            # into different cache identities.
            identity_configuration.pop("stage", None)
            identity_configuration.pop("allow_full", None)
            identity_configuration.pop("artifacts", None)
            identity_configuration.pop("hydra", None)
            payload["resolved_configuration"] = identity_configuration
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
