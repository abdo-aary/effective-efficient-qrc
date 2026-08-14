"""Strict YAML boundary for empirical-evaluation manifests and contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .domain import CampaignId, RepetitionSpec, content_digest, json_value


MANIFEST_SCHEMA_VERSION = "quark.empirical-evaluation-manifest/v1"
SUITE_NAME = "empirical_evaluation"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_PATH = PROJECT_ROOT / "experiments" / SUITE_NAME / "experiment_contract.yaml"


@dataclass(frozen=True)
class ExperimentContract:
    version: str
    theory_locked: Mapping[str, Any]
    campaign_fixed_axes: Mapping[str, Any]
    required_keys: tuple[str, ...]
    digest: str


@dataclass(frozen=True)
class PreRunConfig:
    mixer_gamma: float
    mixer_angle_distribution: str
    matern_nu: float
    matern_xi: float
    ivanov_lambda: float
    master_seed_list: tuple[int, ...]
    heldout_test_size: int
    repetitions_architecture_fidelity: int
    repetitions_pool: int
    campaign_iii_finite_shot_m: int
    mixer_cost_match_m_edge: int
    classical_baseline_definition: str
    kernel_solver_tolerance: float
    floating_precision: str
    numerical_jitter_policy: str
    frozen_pool_tasks: tuple[str, ...]
    homogeneous_memory_slices: tuple[str, ...]
    fidelity_tasks: tuple[str, ...]
    dependence_task: str

    def __post_init__(self) -> None:
        positive = (
            self.mixer_gamma,
            self.matern_nu,
            self.matern_xi,
            self.ivanov_lambda,
            self.kernel_solver_tolerance,
        )
        if min(float(value) for value in positive) <= 0.0 or self.matern_nu <= 1.0:
            raise ValueError("Mixer, Matérn, Ivanov, and solver values must be positive and nu > 1.")
        counts = (
            self.heldout_test_size,
            self.repetitions_architecture_fidelity,
            self.repetitions_pool,
            self.campaign_iii_finite_shot_m,
            self.mixer_cost_match_m_edge,
        )
        if min(int(value) for value in counts) < 1:
            raise ValueError("Pre-run counts must be positive.")
        if not self.master_seed_list or len(set(self.master_seed_list)) != len(self.master_seed_list):
            raise ValueError("master_seed_list must be nonempty and unique.")
        if any(seed < 0 for seed in self.master_seed_list):
            raise ValueError("master_seed_list values must be nonnegative.")
        if len(self.master_seed_list) < max(
            self.repetitions_architecture_fidelity, self.repetitions_pool
        ):
            raise ValueError("master_seed_list is shorter than the declared repetition counts.")
        if len(self.frozen_pool_tasks) != 6 or len(set(self.frozen_pool_tasks)) != 6:
            raise ValueError("Exactly six unique frozen-pool task IDs are required.")
        if not self.homogeneous_memory_slices or not self.fidelity_tasks or not self.dependence_task:
            raise ValueError("Representative slices and fidelity/dependence tasks must be declared.")
        if self.mixer_angle_distribution != "uniform_symmetric":
            raise ValueError("The executable contract requires uniform_symmetric mixer angles.")
        if self.floating_precision != "float64_complex128":
            raise ValueError("The executable contract requires float64_complex128.")
        if self.numerical_jitter_policy != "spectral_clip_1e-12":
            raise ValueError(
                "The executable contract permits spectral clipping only, not diagonal jitter."
            )
        for name, value in (
            ("campaign_iii_finite_shot_m", self.campaign_iii_finite_shot_m),
            ("mixer_cost_match_m_edge", self.mixer_cost_match_m_edge),
        ):
            if int(value) & (int(value) - 1):
                raise ValueError(f"{name} must be dyadic.")


@dataclass(frozen=True)
class RunManifest:
    schema_version: str
    suite: str
    mode: str
    contract_version: str
    contract_digest: str
    campaigns: tuple[CampaignId, ...]
    pre_run: PreRunConfig
    artifact_root: Path

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"Unsupported manifest schema {self.schema_version!r}.")
        if self.suite != SUITE_NAME:
            raise ValueError(f"Manifest suite must be {SUITE_NAME!r}.")
        if self.mode not in {"smoke", "production"}:
            raise ValueError("Manifest mode must be smoke or production.")
        if not self.campaigns or len(set(self.campaigns)) != len(self.campaigns):
            raise ValueError("Manifest campaigns must be nonempty and unique.")
        if self.mode == "production":
            if self.pre_run.repetitions_architecture_fidelity < 16:
                raise ValueError("Production architecture/fidelity runs require at least 16 repetitions.")
            if self.pre_run.repetitions_pool < 32:
                raise ValueError("Production pool runs require at least 32 repetitions.")

    @property
    def repetitions(self) -> tuple[RepetitionSpec, ...]:
        return tuple(
            RepetitionSpec(index=index, root_seed=seed)
            for index, seed in enumerate(self.pre_run.master_seed_list)
        )

    @property
    def digest(self) -> str:
        return content_digest(self.semantic_dict())

    def repetitions_for(self, campaign: CampaignId) -> tuple[RepetitionSpec, ...]:
        campaign = CampaignId(campaign)
        count = (
            self.pre_run.repetitions_pool
            if campaign is CampaignId.FROZEN_POOL
            else self.pre_run.repetitions_architecture_fidelity
        )
        return self.repetitions[:count]

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite": self.suite,
            "mode": self.mode,
            "contract_version": self.contract_version,
            "contract_digest": self.contract_digest,
            "campaigns": [campaign.value for campaign in self.campaigns],
            "pre_run": json_value(self.pre_run),
        }


def load_contract(path: str | Path = DEFAULT_CONTRACT_PATH) -> ExperimentContract:
    source = Path(path)
    payload = _load_yaml_mapping(source)
    _expect_keys(
        payload,
        {"contract_version", "status", "theory_locked", "campaign_fixed_axes", "pre_run_required"},
        owner="experiment contract",
    )
    required = payload["pre_run_required"]
    if not isinstance(required, Mapping):
        raise TypeError("pre_run_required must be a mapping.")
    return ExperimentContract(
        version=str(payload["contract_version"]),
        theory_locked=dict(payload["theory_locked"]),
        campaign_fixed_axes=dict(payload["campaign_fixed_axes"]),
        required_keys=tuple(str(key) for key in required),
        digest=content_digest(payload),
    )


def load_manifest(
    path: str | Path,
    *,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
) -> RunManifest:
    source = Path(path)
    payload = _load_yaml_mapping(source)
    _reject_required(payload, owner=str(source))
    _expect_keys(
        payload,
        {
            "schema_version",
            "suite",
            "mode",
            "contract_version",
            "campaigns",
            "artifact_root",
            "pre_run",
        },
        owner="run manifest",
    )
    contract = load_contract(contract_path)
    if str(payload["contract_version"]) != contract.version:
        raise ValueError("Manifest and executable contract versions differ.")
    pre_run = _parse_pre_run(payload["pre_run"])
    return RunManifest(
        schema_version=str(payload["schema_version"]),
        suite=str(payload["suite"]),
        mode=str(payload["mode"]),
        contract_version=contract.version,
        contract_digest=contract.digest,
        campaigns=tuple(CampaignId(value) for value in payload["campaigns"]),
        pre_run=pre_run,
        artifact_root=Path(str(payload["artifact_root"])),
    )


def _parse_pre_run(raw: object) -> PreRunConfig:
    if not isinstance(raw, Mapping):
        raise TypeError("pre_run must be a mapping.")
    expected = {
        "mixer_gamma",
        "mixer_angle_distribution",
        "matern_nu",
        "matern_xi",
        "ivanov_Lambda",
        "master_seed_list",
        "heldout_test_size",
        "repetitions_architecture_fidelity",
        "repetitions_pool",
        "campaign_III_finite_shot_M",
        "mixer_cost_match_M_edge",
        "classical_baseline_definition",
        "kernel_solver_tolerance",
        "floating_precision",
        "numerical_jitter_policy",
        "frozen_pool_tasks",
        "homogeneous_memory_slices",
        "fidelity_tasks",
        "dependence_task",
    }
    _expect_keys(raw, expected, owner="pre_run")
    return PreRunConfig(
        mixer_gamma=float(raw["mixer_gamma"]),
        mixer_angle_distribution=str(raw["mixer_angle_distribution"]),
        matern_nu=float(raw["matern_nu"]),
        matern_xi=float(raw["matern_xi"]),
        ivanov_lambda=float(raw["ivanov_Lambda"]),
        master_seed_list=tuple(int(value) for value in raw["master_seed_list"]),
        heldout_test_size=int(raw["heldout_test_size"]),
        repetitions_architecture_fidelity=int(raw["repetitions_architecture_fidelity"]),
        repetitions_pool=int(raw["repetitions_pool"]),
        campaign_iii_finite_shot_m=int(raw["campaign_III_finite_shot_M"]),
        mixer_cost_match_m_edge=int(raw["mixer_cost_match_M_edge"]),
        classical_baseline_definition=str(raw["classical_baseline_definition"]),
        kernel_solver_tolerance=float(raw["kernel_solver_tolerance"]),
        floating_precision=str(raw["floating_precision"]),
        numerical_jitter_policy=str(raw["numerical_jitter_policy"]),
        frozen_pool_tasks=tuple(str(value) for value in raw["frozen_pool_tasks"]),
        homogeneous_memory_slices=tuple(str(value) for value in raw["homogeneous_memory_slices"]),
        fidelity_tasks=tuple(str(value) for value in raw["fidelity_tasks"]),
        dependence_task=str(raw["dependence_task"]),
    )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path} must contain a YAML mapping.")
    return dict(payload)


def _expect_keys(payload: Mapping[str, Any], expected: set[str], *, owner: str) -> None:
    actual = set(str(key) for key in payload)
    unknown = actual - expected
    missing = expected - actual
    if unknown or missing:
        raise ValueError(f"Invalid {owner} keys; unknown={sorted(unknown)}, missing={sorted(missing)}.")


def _reject_required(value: Any, *, owner: str) -> None:
    if value == "REQUIRED":
        raise ValueError(f"{owner} contains unresolved REQUIRED values.")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_required(item, owner=owner)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_required(item, owner=owner)

