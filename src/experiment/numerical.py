"""Numerical providers for the Campaign I memory-versus-lag vertical slice."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from multiprocessing import get_context
import re
from time import perf_counter
from typing import Any, Sequence

import numpy as np
from scipy.special import ndtr

from src.api import (
    AerCPUBackend,
    ExactFeatureEstimator,
    ExecutionSpec,
    NvidiaBackend,
    ObservableSet,
    Precision,
    ProjectionSpec,
    QuaRKProgram,
    ResetChannelSpec,
    SeedBundle,
)
from src.core.factories import (
    gaussian_jl_projection,
    ring_topology,
    sample_balanced_reservoirs,
)
from src.models.ivanov import (
    fit_rms_matern_ivanov,
    predict_rms_matern_ivanov,
)

from .domain import (
    AcquisitionKind,
    AcquisitionSpec,
    ComparisonKind,
    ComparisonSpec,
    DataSpec,
    EvaluationSpec,
    ExperimentPlan,
    FeatureViewSpec,
    FitSpec,
    PrefixAxis,
)
from .manifest import RunManifest
from .payload import NodePayload
from .providers import ProviderIdentity
from .seeding import PathSeedTree


_MEMORY_TASK = re.compile(r"F_mem_(0|1|2|4|8|16|32|64)$")


def _parameters(value: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return dict(value)


def _memory_lags(task_ids: Sequence[str]) -> tuple[int, ...]:
    lags = []
    for task in task_ids:
        match = _MEMORY_TASK.fullmatch(str(task))
        if match is None:
            raise ValueError(f"Unsupported numerical task {task!r}; expected F_mem_L.")
        lags.append(int(match.group(1)))
    return tuple(lags)


def _execute_rate_sweep_worker(
    gpu_id: int,
    chunk_size: int,
    program: QuaRKProgram,
    windows: np.ndarray,
    reset_rate_matrix: np.ndarray,
    execution: ExecutionSpec,
) -> tuple[np.ndarray, dict[str, Any]]:
    backend = NvidiaBackend(gpu_id=gpu_id, chunk_size=chunk_size)
    result = backend.execute_exact_rate_sweep(
        program,
        windows,
        reset_rate_matrix,
        execution,
    ).as_numpy()
    return (
        np.asarray(result.values, dtype=np.float64),
        dict(result.execution_metadata.details),
    )


def execute_nvidia_rate_sweep_sharded(
    *,
    backends: Sequence[NvidiaBackend],
    program: QuaRKProgram,
    windows: np.ndarray,
    reset_rate_matrix: np.ndarray,
    execution: ExecutionSpec,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Execute contiguous window shards concurrently and restore input order."""

    backend_pool = tuple(backends)
    if not backend_pool:
        raise ValueError("A NVIDIA rate sweep requires at least one backend.")
    if len(backend_pool) == 1:
        result = backend_pool[0].execute_exact_rate_sweep(
            program,
            windows,
            reset_rate_matrix,
            execution,
        ).as_numpy()
        return (
            np.asarray(result.values, dtype=np.float64),
            dict(result.execution_metadata.details),
        )
    if execution.retain_device_array:
        raise ValueError("Multi-process GPU execution cannot return device arrays.")

    index_shards = [
        shard
        for shard in np.array_split(np.arange(windows.shape[0]), len(backend_pool))
        if shard.size
    ]
    started = perf_counter()

    with ProcessPoolExecutor(
        max_workers=len(index_shards),
        mp_context=get_context("spawn"),
    ) as pool:
        futures = [
            pool.submit(
                _execute_rate_sweep_worker,
                int(backend.gpu_id),
                int(backend.chunk_size),
                program,
                windows[indices],
                reset_rate_matrix,
                execution,
            )
            for backend, indices in zip(backend_pool, index_shards)
        ]
        shard_results = [future.result() for future in futures]

    values = np.concatenate([result[0] for result in shard_results], axis=0)
    shard_details = []
    for backend, indices, (_, details) in zip(
        backend_pool,
        index_shards,
        shard_results,
    ):
        shard_details.append(
            {
                "gpu_id": backend.gpu_id,
                "chunk_size": backend.chunk_size,
                "window_start": int(indices[0]),
                "window_stop": int(indices[-1] + 1),
                "window_count": int(indices.size),
                "execution": details,
            }
        )
    return values, {
        "engine": "cupy-multi-gpu",
        "channel_realization": "exact-shared-pure-history-rate-sweep",
        "gpu_ids": [backend.gpu_id for backend in backend_pool],
        "rate_count": int(reset_rate_matrix.shape[0]),
        "window_count": int(windows.shape[0]),
        "wall_seconds": perf_counter() - started,
        "shards": shard_details,
    }


@dataclass(frozen=True)
class ControlledLegendreDataProvider:
    manifest: RunManifest

    @property
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity(
            kind="controlled_legendre_data",
            algorithm_version="stationary-gaussian-ar1-copula/v1",
            backend_kind="numpy_cpu",
            numerical_precision="float64",
        )

    def prepare(self, spec: DataSpec, seeds: PathSeedTree) -> NodePayload:
        if spec.study_id != "memory_vs_lag":
            raise ValueError(
                "The numerical data provider currently supports only memory_vs_lag."
            )
        lags = _memory_lags(spec.task_ids)
        if max(lags) >= spec.window_length:
            raise ValueError("Every memory lag must fit inside the acquired window.")
        if spec.input_dim != 1:
            raise ValueError("The memory vertical slice requires input_dim=1.")
        rho = (
            0.0
            if spec.dependence_half_life is None
            else float(2.0 ** (-1.0 / spec.dependence_half_life))
        )
        origins = spec.window_length - 1 + np.arange(spec.sample_count, dtype=np.int64) * (
            spec.window_length + spec.gap
        )
        raw_length = int(origins[-1] + 1)
        seed_path = f"data/{spec.study_id}/{spec.split}/{spec.trajectory_id}"
        rng = seeds.generator(seed_path)
        gaussian = np.empty((raw_length, spec.input_dim), dtype=np.float64)
        gaussian[0] = rng.normal(size=spec.input_dim)
        innovations = rng.normal(size=(raw_length - 1, spec.input_dim))
        innovation_scale = np.sqrt(1.0 - rho * rho)
        for index in range(1, raw_length):
            gaussian[index] = rho * gaussian[index - 1] + innovation_scale * innovations[index - 1]
        trajectory = 2.0 * ndtr(gaussian) - 1.0
        targets = np.stack(
            [
                0.5 * (3.0 * trajectory[origins - lag, 0] ** 2 - 1.0)
                for lag in lags
            ],
            axis=1,
        )
        return NodePayload(
            metadata={
                "provider": self.identity.kind,
                "split": spec.split,
                "trajectory_id": spec.trajectory_id,
                "task_ids": list(spec.task_ids),
                "lags": list(lags),
                "sample_count": spec.sample_count,
                "window_length": spec.window_length,
                "gap": spec.gap,
                "input_dim": spec.input_dim,
                "process": "stationary_gaussian_ar1_uniform_copula",
                "rho": rho,
                "seed_path": seed_path,
                "window_origin_formula": "w-1+j*(w+g)",
            },
            assets={
                "trajectory": trajectory,
                "label_times": origins,
                "targets": targets,
            },
        )


@dataclass(frozen=True)
class ExactQuaRKRepresentationProvider:
    manifest: RunManifest
    backend: AerCPUBackend | NvidiaBackend
    backend_name: str
    gpu_id: int | None = None
    chunk_size: int | None = None
    nvidia_backends: tuple[NvidiaBackend, ...] = ()

    @property
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity(
            kind="exact_balanced_quark",
            algorithm_version="balanced-random-axis-channel/v2-multi-rate",
            backend_kind=self.backend_name,
            numerical_precision="complex128",
        )

    def _execute_nvidia_rate_sweep(
        self,
        program: QuaRKProgram,
        windows: np.ndarray,
        reset_rate_matrix: np.ndarray,
        execution: ExecutionSpec,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        backends = self.nvidia_backends or (
            (self.backend,) if isinstance(self.backend, NvidiaBackend) else ()
        )
        return execute_nvidia_rate_sweep_sharded(
            backends=backends,
            program=program,
            windows=windows,
            reset_rate_matrix=reset_rate_matrix,
            execution=execution,
        )

    def acquire(
        self,
        spec: AcquisitionSpec,
        data: NodePayload,
        seeds: PathSeedTree,
    ) -> NodePayload:
        if spec.study_id != "memory_vs_lag":
            raise ValueError(
                "The numerical representation provider supports only memory_vs_lag."
            )
        if spec.kind is not AcquisitionKind.EXACT or spec.program_key != "quark":
            raise ValueError("The memory numerical slice requires exact QuaRK acquisition.")
        resources = _parameters(spec.fixed_resources)
        n = int(resources["n"])
        if "tau_plus_values" in resources:
            tau_plus_values = tuple(float(value) for value in resources["tau_plus_values"])
        else:
            tau_plus_values = (float(resources["tau_plus"]),)
        if (
            not tau_plus_values
            or len(set(tau_plus_values)) != len(tau_plus_values)
            or tuple(sorted(tau_plus_values)) != tau_plus_values
            or tau_plus_values[0] <= 1.0
        ):
            raise ValueError("tau_plus_values must be unique, increasing, and greater than one.")
        branch_count = spec.limit(PrefixAxis.R) or int(resources.get("R", 0))
        if branch_count < 1:
            raise ValueError("Exact acquisition requires an explicit positive branch count.")

        trajectory = data.require_asset("trajectory").load()
        origins = data.require_asset("label_times").load()
        w = int(data.metadata["window_length"])
        offsets = np.arange(w - 1, -1, -1, dtype=np.int64)
        # Oldest-to-newest windows: origin-w+1, ..., origin.
        indices = origins[:, None] - offsets[None, :]
        windows = np.asarray(trajectory[indices], dtype=np.float64)

        topology = ring_topology(n)
        projection_path = f"projection/{spec.study_id}"
        mixer_path = f"{spec.randomness_path}/mixer"
        memory_path = f"{spec.randomness_path}/memory_uniforms"
        projection = gaussian_jl_projection(
            input_dim=windows.shape[2],
            encoded_width=n,
            rng=seeds.generator(projection_path),
        )
        lambda_plus_values = np.exp(-1.0 / np.asarray(tau_plus_values, dtype=np.float64))
        parameters = sample_balanced_reservoirs(
            num_reservoirs=branch_count,
            topology=topology,
            lambda_plus=float(lambda_plus_values[-1]),
            gamma=self.manifest.pre_run.mixer_gamma,
            angle_distribution=self.manifest.pre_run.mixer_angle_distribution,
            mixer_rng=seeds.generator(mixer_path),
            memory_rng=seeds.generator(memory_path),
        )
        observables = ObservableSet.cycle_complete(
            num_qubits=n, edges=topology.edges
        )
        lambda_0 = float(np.exp(-1.0))
        memory_uniforms = np.log(parameters.reset_rates / lambda_0) / np.log(
            float(lambda_plus_values[-1]) / lambda_0
        )
        reset_rate_matrix = lambda_0 * np.power(
            lambda_plus_values[:, None] / lambda_0,
            memory_uniforms[None, :],
        )

        programs = tuple(
            QuaRKProgram(
                projection=ProjectionSpec(projection, mode="gaussian_jl"),
                topology=topology,
                reservoirs=replace(parameters, reset_rates=reset_rates),
                reset_channel=ResetChannelSpec(),
                observables=observables,
                window_length=w,
                angle_map="tanh",
                angle_scale=np.pi,
            )
            for reset_rates in reset_rate_matrix
        )
        estimator = ExactFeatureEstimator(
            precision=Precision.COMPLEX128, return_states=False
        )
        execution = ExecutionSpec(
            seeds=SeedBundle.from_root(
                seeds.integer(f"execution/{spec.study_id}/{spec.split}/exact")
            ),
            chunk_size=self.chunk_size,
        )
        if len(programs) > 1 and isinstance(self.backend, NvidiaBackend):
            values, execution_details = self._execute_nvidia_rate_sweep(
                programs[-1],
                windows,
                reset_rate_matrix,
                execution,
            )
        else:
            batches = tuple(
                estimator.estimate(program, windows, self.backend, execution).as_numpy()
                for program in programs
            )
            if len(batches) == 1:
                values = np.asarray(batches[0].values, dtype=np.float64)
                execution_details = dict(batches[0].execution_metadata.details)
            else:
                values = np.stack(
                    [np.asarray(batch.values, dtype=np.float64) for batch in batches],
                    axis=1,
                )
                execution_details = [
                    dict(batch.execution_metadata.details) for batch in batches
                ]
        is_rate_sweep = len(tau_plus_values) > 1
        return NodePayload(
            metadata={
                "provider": self.identity.kind,
                "backend": self.backend_name,
                "precision": "complex128",
                "operational_gpu_id": self.gpu_id,
                "operational_gpu_ids": [
                    backend.gpu_id for backend in self.nvidia_backends
                ],
                "operational_chunk_size": self.chunk_size,
                "program_fingerprint": programs[-1].fingerprint(),
                "projection_mode": programs[-1].projection.mode,
                "projection_seed_path": projection_path,
                "mixer_seed_path": mixer_path,
                "memory_seed_path": memory_path,
                "tau_plus_values": list(tau_plus_values),
                "lambda_plus_values": lambda_plus_values.tolist(),
                "tau_axis": 1 if is_rate_sweep else None,
                "shared_history_evolution": (
                    is_rate_sweep and isinstance(self.backend, NvidiaBackend)
                ),
                "n": n,
                "R_max": branch_count,
                "window_length": w,
                "observable_labels": list(observables.labels),
                "observable_ordering": "vertex-then-cycle-edge/v1",
                "reservoir_ids": list(parameters.reservoir_ids),
                "reset_state": "plus_tensor_n",
                "history_truncation": False,
                "branch_pruning": False,
                "program_fingerprints": [
                    program.fingerprint() for program in programs
                ],
                "execution_details": execution_details,
            },
            assets={
                "features": values,
                "projection": projection,
                "local_axes": parameters.local_axes,
                "local_angles": parameters.local_angles,
                "edge_axes_left": parameters.edge_axes_left,
                "edge_axes_right": parameters.edge_axes_right,
                "edge_angles": parameters.edge_angles,
                "matching_orders": parameters.matching_orders,
                "reset_rates": (
                    reset_rate_matrix if is_rate_sweep else reset_rate_matrix[0]
                ),
                "memory_uniforms": memory_uniforms,
            },
        )

    def derive(
        self,
        spec: FeatureViewSpec,
        acquisition: NodePayload,
        data: NodePayload,
    ) -> NodePayload:
        if spec.study_id != "memory_vs_lag":
            raise ValueError("Numerical feature views currently support only memory_vs_lag.")
        if spec.observable_bank != "complete":
            raise ValueError("The memory slice requires the complete observable bank.")
        if tuple(spec.task_ids) != tuple(data.metadata["task_ids"]):
            raise ValueError("The memory slice requires the complete locked task ordering.")
        branch_count = next(
            (
                item.value
                for item in spec.prefixes
                if item.axis is PrefixAxis.R
            ),
            int(acquisition.metadata["R_max"]),
        )
        features = acquisition.require_asset("features")
        tau_plus_values = tuple(
            float(value) for value in acquisition.metadata["tau_plus_values"]
        )
        view_parameters = _parameters(spec.parameters)
        if len(tau_plus_values) == 1:
            tau_plus = tau_plus_values[0]
            view = features.view(
                (slice(None), slice(0, branch_count), slice(None))
            )
        else:
            if "tau_plus" not in view_parameters:
                raise ValueError("A multi-rate feature view requires a tau_plus selector.")
            tau_plus = float(view_parameters["tau_plus"])
            matches = [
                index
                for index, value in enumerate(tau_plus_values)
                if np.isclose(value, tau_plus, rtol=0.0, atol=1e-12)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"tau_plus={tau_plus} is not present in the acquisition cache."
                )
            tau_index = matches[0]
            view = features.view(
                (
                    slice(None),
                    slice(tau_index, tau_index + 1),
                    slice(0, branch_count),
                    slice(None),
                )
            )
        return NodePayload(
            metadata={
                "provider": self.identity.kind,
                "task_ids": list(spec.task_ids),
                "R": branch_count,
                "R_max": int(acquisition.metadata["R_max"]),
                "n": int(acquisition.metadata["n"]),
                "tau_plus": tau_plus,
                "tau_plus_values": list(tau_plus_values),
                "shared_history_evolution": bool(
                    acquisition.metadata["shared_history_evolution"]
                ),
                "observable_labels": list(acquisition.metadata["observable_labels"]),
                "feature_ordering": "reservoir-major-observable-minor/v1",
                "prefix_is_view": True,
            },
            assets={
                "features": view,
                "targets": data.require_asset("targets"),
                "label_times": data.require_asset("label_times"),
            },
        )


@dataclass(frozen=True)
class RmsMaternIvanovReadoutProvider:
    manifest: RunManifest

    @property
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity(
            kind="rms_matern_ivanov",
            algorithm_version="spectral-ivanov/v1",
            backend_kind="numpy_cpu",
            numerical_precision="float64",
        )

    def fit(self, spec: FitSpec, features: NodePayload) -> NodePayload:
        if spec.study_id != "memory_vs_lag":
            raise ValueError("Numerical readout currently supports only memory_vs_lag.")
        if spec.readout_key != "fixed_rms_matern_ivanov":
            raise ValueError(f"Unsupported readout {spec.readout_key!r}.")
        if tuple(spec.task_ids) != tuple(features.metadata["task_ids"]):
            raise ValueError("The grouped memory fit must retain the locked task ordering.")
        feature_asset = features.require_asset("features")
        X3 = feature_asset.load()
        X = np.asarray(X3).reshape(X3.shape[0], -1)
        targets = np.asarray(features.require_asset("targets").load(), dtype=np.float64)
        fit = fit_rms_matern_ivanov(
            X,
            targets,
            task_names=spec.task_ids,
            nu=self.manifest.pre_run.matern_nu,
            xi=self.manifest.pre_run.matern_xi,
            radius=self.manifest.pre_run.ivanov_lambda,
            tolerance=self.manifest.pre_run.kernel_solver_tolerance,
            spectral_clip=1e-12,
        )
        return NodePayload(
            metadata={
                "provider": self.identity.kind,
                "task_ids": list(spec.task_ids),
                "nu": fit.nu,
                "xi": fit.xi,
                "Lambda": fit.radius,
                "solver_tolerance": fit.tolerance,
                "jitter_policy": "spectral_clip_1e-12",
                "feature_standardization": False,
                "shared_gram_eigendecomposition": True,
                "feature_dimension": X.shape[1],
                "train_sample_count": X.shape[0],
                "diagnostics": list(fit.diagnostics),
                "eigenvalue_min": float(np.min(fit.eigenvalues)),
                "eigenvalue_max": float(np.max(fit.eigenvalues)),
                "effective_rank": int(
                    np.count_nonzero(
                        fit.eigenvalues
                        > 1e-12 * max(1.0, float(np.max(fit.eigenvalues)))
                    )
                ),
            },
            assets={
                "alphas": fit.alphas,
                "gram_eigenvalues": fit.eigenvalues,
                "support_features": feature_asset,
            },
        )

    def evaluate(
        self,
        spec: EvaluationSpec,
        fit: NodePayload,
        features: NodePayload,
        data: NodePayload,
    ) -> NodePayload:
        if len(spec.task_ids) != 1:
            raise ValueError("Memory evaluations must be one task per artifact.")
        task = spec.task_ids[0]
        task_order = tuple(str(item) for item in fit.metadata["task_ids"])
        try:
            task_index = task_order.index(task)
        except ValueError as exc:
            raise ValueError(f"Fitted readout has no task {task!r}.") from exc
        test_asset = features.require_asset("features")
        test3 = test_asset.load()
        train3 = fit.require_asset("support_features").load()
        test_X = np.asarray(test3).reshape(test3.shape[0], -1)
        train_X = np.asarray(train3).reshape(train3.shape[0], -1)
        alphas = fit.require_asset("alphas").load()
        predictions = predict_rms_matern_ivanov(
            test_X,
            train_X,
            np.asarray(alphas)[:, task_index : task_index + 1],
            nu=float(fit.metadata["nu"]),
            xi=float(fit.metadata["xi"]),
        )[:, 0]
        target_order = tuple(str(item) for item in data.metadata["task_ids"])
        target_index = target_order.index(task)
        targets = data.require_asset("targets").load()[:, target_index]
        mse = float(np.mean((predictions - targets) ** 2))
        variance = float(np.var(targets, ddof=0))
        if not np.isfinite(variance) or variance <= 0.0:
            raise FloatingPointError(f"Held-out variance for {task!r} is not positive.")
        reporting = _parameters(spec.parameters)
        return NodePayload(
            metadata={
                "provider": self.identity.kind,
                "task": task,
                "task_index": task_index,
                "mse": mse,
                "risk": mse,
                "heldout_label_variance_ddof0": variance,
                "denominator_key": spec.denominator_key,
                **reporting,
            },
            assets={
                "predictions": predictions,
                "targets": data.require_asset("targets").view(
                    (slice(None), slice(target_index, target_index + 1))
                ),
            },
        )

    def compare(
        self,
        spec: ComparisonSpec,
        evaluations: Sequence[NodePayload],
    ) -> NodePayload:
        if spec.kind is not ComparisonKind.NEMSE or len(evaluations) != 1:
            raise ValueError("The memory numerical slice currently supports only NEMSE.")
        evaluation = evaluations[0]
        variance = float(evaluation.metadata["heldout_label_variance_ddof0"])
        value = float(evaluation.metadata["mse"]) / variance
        return NodePayload(
            metadata={
                "provider": self.identity.kind,
                "comparison": "nemse",
                "task": evaluation.metadata["task"],
                "mse": float(evaluation.metadata["mse"]),
                "denominator": variance,
                "denominator_ddof": 0,
                "denominator_key": spec.denominator_key,
                "value": value,
                **_parameters(spec.parameters),
            }
        )


@dataclass(frozen=True)
class NumericalProviderBundle:
    data: ControlledLegendreDataProvider
    representation: ExactQuaRKRepresentationProvider
    readout: RmsMaternIvanovReadoutProvider

    @classmethod
    def create(
        cls,
        manifest: RunManifest,
        *,
        backend: str,
        gpu_id: int | None,
        gpu_ids: Sequence[int] | None = None,
        chunk_size: int | None,
    ) -> "NumericalProviderBundle":
        if manifest.pre_run.floating_precision != "float64_complex128":
            raise ValueError("Numerical execution requires float64_complex128.")
        if manifest.pre_run.numerical_jitter_policy != "spectral_clip_1e-12":
            raise ValueError("Numerical execution requires spectral_clip_1e-12.")
        if backend == "nvidia":
            selected_gpu_ids = tuple(
                int(value)
                for value in (
                    gpu_ids
                    if gpu_ids is not None
                    else ((gpu_id,) if gpu_id is not None else (0,))
                )
            )
            if (
                not selected_gpu_ids
                or len(set(selected_gpu_ids)) != len(selected_gpu_ids)
                or min(selected_gpu_ids) < 0
            ):
                raise ValueError("NVIDIA GPU IDs must be unique and nonnegative.")
            import cupy as cp

            nvidia_backends = []
            for selected_gpu_id in selected_gpu_ids:
                if chunk_size is None:
                    properties = cp.cuda.runtime.getDeviceProperties(selected_gpu_id)
                    total_memory = int(properties["totalGlobalMem"])
                    selected_chunk_size = 512 if total_memory >= 60 * 2**30 else 256
                else:
                    selected_chunk_size = int(chunk_size)
                nvidia_backends.append(
                    NvidiaBackend(
                        gpu_id=selected_gpu_id,
                        chunk_size=selected_chunk_size,
                    )
                )
            backend_pool = tuple(nvidia_backends)
            execution_backend: AerCPUBackend | NvidiaBackend = backend_pool[0]
        elif backend == "aer":
            execution_backend = AerCPUBackend(max_qubits=4, max_windows=32)
            backend_pool = ()
            selected_gpu_ids = ()
        else:
            raise ValueError("backend must be 'nvidia' or 'aer'.")
        return cls(
            data=ControlledLegendreDataProvider(manifest),
            representation=ExactQuaRKRepresentationProvider(
                manifest=manifest,
                backend=execution_backend,
                backend_name=backend,
                gpu_id=selected_gpu_ids[0] if selected_gpu_ids else None,
                chunk_size=chunk_size,
                nvidia_backends=backend_pool,
            ),
            readout=RmsMaternIvanovReadoutProvider(manifest),
        )

    def preflight(self, plan: ExperimentPlan, *, study: str | None) -> None:
        if study != "memory_vs_lag":
            raise ValueError(
                "Numerical execution requires --study memory_vs_lag in this milestone."
            )
        selected_data = [item for item in plan.data if item.study_id == study]
        selected_acquisitions = [
            item for item in plan.acquisitions if item.study_id == study
        ]
        selected_views = [item for item in plan.feature_views if item.study_id == study]
        selected_fits = [item for item in plan.fits if item.study_id == study]
        if not all((selected_data, selected_acquisitions, selected_views, selected_fits)):
            raise ValueError("The selected numerical study is incomplete.")
        for item in selected_data:
            _memory_lags(item.task_ids)
        for item in selected_acquisitions:
            resources = _parameters(item.fixed_resources)
            if item.kind is not AcquisitionKind.EXACT or item.program_key != "quark":
                raise ValueError(f"Unsupported numerical acquisition {item.id!r}.")
            n = int(resources["n"])
            if self.representation.backend_name == "aer":
                if n > self.representation.backend.max_qubits:
                    raise ValueError(
                        f"Aer canary is bounded to n<={self.representation.backend.max_qubits}; "
                        f"selected study requests n={n}."
                    )
                data = next(value for value in selected_data if value.id == item.data_id)
                if data.sample_count > self.representation.backend.max_windows:
                    raise ValueError(
                        "Aer canary exceeds its bounded window count; use NVIDIA for the locked study."
                    )
        selected_evaluations = [
            item for item in plan.evaluations if item.study_id == study
        ]
        selected_comparisons = [
            item for item in plan.comparisons if item.study_id == study
        ]
        if any(item.readout_key != "fixed_rms_matern_ivanov" for item in selected_fits):
            raise ValueError("Selected study contains an unsupported numerical readout.")
        if any(len(item.task_ids) != 1 for item in selected_evaluations):
            raise ValueError("Numerical memory evaluations must be task-local artifacts.")
        if any(item.kind is not ComparisonKind.NEMSE for item in selected_comparisons):
            raise ValueError("Numerical memory comparisons must be NEMSE.")
        if self.representation.backend_name == "nvidia":
            try:
                import cupy as cp
            except ImportError as exc:
                raise ValueError("NVIDIA numerical execution requires CuPy.") from exc
            count = int(cp.cuda.runtime.getDeviceCount())
            gpu_ids = tuple(
                backend.gpu_id for backend in self.representation.nvidia_backends
            )
            if not gpu_ids:
                gpu_ids = (self.representation.gpu_id,)
            for gpu_id in gpu_ids:
                if gpu_id is None or not 0 <= gpu_id < count:
                    raise ValueError(
                        f"Invalid NVIDIA GPU ID {gpu_id!r}; detected {count} devices."
                    )
                with cp.cuda.Device(gpu_id):
                    cp.zeros(1, dtype=cp.complex128).sum().item()
                    cp.cuda.Stream.null.synchronize()
