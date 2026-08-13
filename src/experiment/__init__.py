"""Public API for reproducible empirical-evaluation planning and execution."""

from .artifact_store import ArtifactStore, LocalArtifactStore
from .domain import (
    AcquisitionSpec,
    CampaignId,
    ComparisonSpec,
    DataSpec,
    EvaluationSpec,
    ExperimentPlan,
    FeatureViewSpec,
    FitSpec,
    RepetitionSpec,
    Stage,
)
from .manifest import RunManifest, load_manifest
from .planning import CampaignPlanner, plan_experiment
from .providers import DataProvider, ReadoutProvider, RepresentationProvider
from .runner import ExperimentResult, ExperimentRunner

__all__ = [
    "AcquisitionSpec",
    "ArtifactStore",
    "CampaignId",
    "CampaignPlanner",
    "ComparisonSpec",
    "DataProvider",
    "DataSpec",
    "EvaluationSpec",
    "ExperimentPlan",
    "ExperimentResult",
    "ExperimentRunner",
    "FeatureViewSpec",
    "FitSpec",
    "LocalArtifactStore",
    "ReadoutProvider",
    "RepetitionSpec",
    "RepresentationProvider",
    "RunManifest",
    "Stage",
    "load_manifest",
    "plan_experiment",
]
