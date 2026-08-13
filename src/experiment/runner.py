"""Synchronous staged executor for validated experiment plans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .artifact_store import ArtifactRef, ArtifactStore
from .domain import ExperimentPlan, Stage
from .providers import DataProvider, ReadoutProvider, RepresentationProvider
from .seeding import PathSeedTree


@dataclass(frozen=True)
class ExperimentResult:
    campaign: str
    repetition: int
    manifest_digest: str
    plan_digest: str
    completed_stage: Stage
    artifact_path: Path
    node_digests: tuple[tuple[str, str], ...]
    reused_nodes: int


class ExperimentRunner:
    def __init__(
        self,
        *,
        data_provider: DataProvider,
        representation_provider: RepresentationProvider,
        readout_provider: ReadoutProvider,
        artifact_store: ArtifactStore,
    ) -> None:
        self.data_provider = data_provider
        self.representation_provider = representation_provider
        self.readout_provider = readout_provider
        self.store = artifact_store

    def run(self, plan: ExperimentPlan, *, through: Stage = Stage.FINALIZE) -> ExperimentResult:
        through = Stage(through)
        expected_identity = (
            plan.campaign.value,
            plan.repetition.index,
            plan.manifest_digest,
            plan.digest,
        )
        if self.store.experiment_identity != expected_identity:
            raise ValueError("Artifact store is bound to a different experiment plan.")
        seeds = PathSeedTree.from_root(plan.repetition.root_seed)
        refs: dict[str, ArtifactRef] = {}
        current_stage, current_node = Stage.PREPARE, "initialize"
        last_path = self.store.run_root
        try:
            for spec in plan.data:
                current_stage, current_node = Stage.PREPARE, spec.id
                ref = self.store.execute_node(
                    stage=current_stage,
                    node_id=spec.id,
                    spec=spec,
                    upstream=(),
                    producer=lambda spec=spec: self.data_provider.prepare(spec, seeds),
                )
                refs[_key("data", spec.id)] = ref
                last_path = ref.path
            if through is Stage.PREPARE:
                return _result(plan, through, last_path, refs)

            for spec in plan.acquisitions:
                current_stage, current_node = Stage.ACQUIRE, spec.id
                data_ref = refs[_key("data", spec.data_id)]
                ref = self.store.execute_node(
                    stage=current_stage,
                    node_id=spec.id,
                    spec=spec,
                    upstream=(data_ref,),
                    producer=lambda spec=spec, data_ref=data_ref: self.representation_provider.acquire(
                        spec, data_ref.payload, seeds
                    ),
                )
                refs[_key("acquisition", spec.id)] = ref
                last_path = ref.path
            for spec in plan.feature_views:
                current_stage, current_node = Stage.ACQUIRE, spec.id
                acquisition_ref = refs[_key("acquisition", spec.acquisition_id)]
                data_ref = refs[_key("data", spec.data_id)]
                ref = self.store.execute_node(
                    stage=current_stage,
                    node_id=spec.id,
                    spec=spec,
                    upstream=(acquisition_ref, data_ref),
                    producer=lambda spec=spec, acquisition_ref=acquisition_ref, data_ref=data_ref: self.representation_provider.derive(
                        spec, acquisition_ref.payload, data_ref.payload
                    ),
                )
                refs[_key("view", spec.id)] = ref
                last_path = ref.path
            if through is Stage.ACQUIRE:
                return _result(plan, through, last_path, refs)

            for spec in plan.fits:
                current_stage, current_node = Stage.ANALYZE, spec.id
                feature_ref = refs[_key("view", spec.feature_view_id)]
                ref = self.store.execute_node(
                    stage=current_stage,
                    node_id=spec.id,
                    spec=spec,
                    upstream=(feature_ref,),
                    producer=lambda spec=spec, feature_ref=feature_ref: self.readout_provider.fit(
                        spec, feature_ref.payload
                    ),
                )
                refs[_key("fit", spec.id)] = ref
                last_path = ref.path
            for spec in plan.evaluations:
                current_stage, current_node = Stage.ANALYZE, spec.id
                fit_ref = refs[_key("fit", spec.fit_id)]
                feature_ref = refs[_key("view", spec.feature_view_id)]
                data_ref = refs[_key("data", spec.data_id)]
                ref = self.store.execute_node(
                    stage=current_stage,
                    node_id=spec.id,
                    spec=spec,
                    upstream=(fit_ref, feature_ref, data_ref),
                    producer=lambda spec=spec, fit_ref=fit_ref, feature_ref=feature_ref, data_ref=data_ref: self.readout_provider.evaluate(
                        spec, fit_ref.payload, feature_ref.payload, data_ref.payload
                    ),
                )
                refs[_key("evaluation", spec.id)] = ref
                last_path = ref.path
            for spec in plan.comparisons:
                current_stage, current_node = Stage.ANALYZE, spec.id
                evaluation_refs = tuple(refs[_key("evaluation", item)] for item in spec.evaluation_ids)
                ref = self.store.execute_node(
                    stage=current_stage,
                    node_id=spec.id,
                    spec=spec,
                    upstream=evaluation_refs,
                    producer=lambda spec=spec, evaluation_refs=evaluation_refs: self.readout_provider.compare(
                        spec, [item.payload for item in evaluation_refs]
                    ),
                )
                refs[_key("comparison", spec.id)] = ref
                last_path = ref.path
            if through is Stage.ANALYZE:
                return _result(plan, through, last_path, refs)

            current_stage, current_node = Stage.FINALIZE, "experiment"
            comparison_refs = tuple(refs[_key("comparison", spec.id)] for spec in plan.comparisons)
            final_ref = self.store.execute_node(
                stage=current_stage,
                node_id="experiment",
                spec={
                    "campaign": plan.campaign.value,
                    "repetition": plan.repetition.index,
                    "plan_digest": plan.digest,
                },
                upstream=comparison_refs,
                producer=lambda: {
                    "status": "complete",
                    "campaign": plan.campaign.value,
                    "repetition": plan.repetition.index,
                    "manifest_digest": plan.manifest_digest,
                    "plan_digest": plan.digest,
                    "comparison_count": len(plan.comparisons),
                    "comparison_digests": [item.digest for item in comparison_refs],
                },
            )
            refs[_key("final", "experiment")] = final_ref
            return _result(plan, Stage.FINALIZE, final_ref.path, refs)
        except Exception as exc:
            self.store.record_failure(stage=current_stage, node_id=current_node, error=exc)
            raise


def _key(kind: str, node_id: str) -> str:
    return f"{kind}:{node_id}"


def _result(
    plan: ExperimentPlan,
    stage: Stage,
    path: Path,
    refs: Mapping[str, ArtifactRef],
) -> ExperimentResult:
    return ExperimentResult(
        campaign=plan.campaign.value,
        repetition=plan.repetition.index,
        manifest_digest=plan.manifest_digest,
        plan_digest=plan.digest,
        completed_stage=stage,
        artifact_path=path,
        node_digests=tuple(sorted((key, ref.digest) for key, ref in refs.items())),
        reused_nodes=sum(int(ref.reused) for ref in refs.values()),
    )

