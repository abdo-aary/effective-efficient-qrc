"""Synchronous staged executor for validated experiment plans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .artifact_store import ArtifactRef, ArtifactStore
from .domain import ExperimentPlan, Stage
from .providers import (
    DataProvider,
    ProviderIdentity,
    ReadoutProvider,
    RepresentationProvider,
)
from .seeding import PathSeedTree


_FINALIZER_IDENTITY = ProviderIdentity(
    kind="experiment_finalizer",
    algorithm_version="study-aware-finalizer/v1",
    backend_kind="local",
    numerical_precision="not_applicable",
)


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
    selected_studies: tuple[str, ...]
    finalization_scope: str


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

    def run(
        self,
        plan: ExperimentPlan,
        *,
        through: Stage = Stage.FINALIZE,
        study: str | None = None,
    ) -> ExperimentResult:
        through = Stage(through)
        selected_studies = plan.study_ids if study is None else (str(study),)
        unknown = set(selected_studies) - set(plan.study_ids)
        if unknown:
            raise ValueError(
                f"Unknown study {sorted(unknown)!r}; available studies are {list(plan.study_ids)!r}."
            )
        selected = frozenset(selected_studies)
        finalization_scope = "global" if study is None else "study"
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
                if spec.study_id not in selected:
                    continue
                current_stage, current_node = Stage.PREPARE, spec.id
                ref = self.store.execute_node(
                    stage=current_stage,
                    node_id=spec.id,
                    spec=spec,
                    upstream=(),
                    producer_identity=self.data_provider.identity,
                    producer=lambda spec=spec: self.data_provider.prepare(spec, seeds),
                )
                refs[_key("data", spec.id)] = ref
                last_path = ref.path
            if through is Stage.PREPARE:
                return _result(
                    plan, through, last_path, refs, selected_studies, finalization_scope
                )

            for spec in plan.acquisitions:
                if spec.study_id not in selected:
                    continue
                current_stage, current_node = Stage.ACQUIRE, spec.id
                data_ref = refs[_key("data", spec.data_id)]
                ref = self.store.execute_node(
                    stage=current_stage,
                    node_id=spec.id,
                    spec=spec,
                    upstream=(data_ref,),
                    producer_identity=self.representation_provider.identity,
                    producer=lambda spec=spec, data_ref=data_ref: self.representation_provider.acquire(
                        spec, data_ref.payload, seeds
                    ),
                )
                refs[_key("acquisition", spec.id)] = ref
                last_path = ref.path
            for spec in plan.feature_views:
                if spec.study_id not in selected:
                    continue
                current_stage, current_node = Stage.ACQUIRE, spec.id
                acquisition_ref = refs[_key("acquisition", spec.acquisition_id)]
                data_ref = refs[_key("data", spec.data_id)]
                ref = self.store.execute_node(
                    stage=current_stage,
                    node_id=spec.id,
                    spec=spec,
                    upstream=(acquisition_ref, data_ref),
                    producer_identity=self.representation_provider.identity,
                    producer=lambda spec=spec, acquisition_ref=acquisition_ref, data_ref=data_ref: self.representation_provider.derive(
                        spec, acquisition_ref.payload, data_ref.payload
                    ),
                )
                refs[_key("view", spec.id)] = ref
                last_path = ref.path
            if through is Stage.ACQUIRE:
                return _result(
                    plan, through, last_path, refs, selected_studies, finalization_scope
                )

            for spec in plan.fits:
                if spec.study_id not in selected:
                    continue
                current_stage, current_node = Stage.ANALYZE, spec.id
                feature_ref = refs[_key("view", spec.feature_view_id)]
                ref = self.store.execute_node(
                    stage=current_stage,
                    node_id=spec.id,
                    spec=spec,
                    upstream=(feature_ref,),
                    producer_identity=self.readout_provider.identity,
                    producer=lambda spec=spec, feature_ref=feature_ref: self.readout_provider.fit(
                        spec, feature_ref.payload
                    ),
                )
                refs[_key("fit", spec.id)] = ref
                last_path = ref.path
            for spec in plan.evaluations:
                if spec.study_id not in selected:
                    continue
                current_stage, current_node = Stage.ANALYZE, spec.id
                fit_ref = refs[_key("fit", spec.fit_id)]
                feature_ref = refs[_key("view", spec.feature_view_id)]
                data_ref = refs[_key("data", spec.data_id)]
                ref = self.store.execute_node(
                    stage=current_stage,
                    node_id=spec.id,
                    spec=spec,
                    upstream=(fit_ref, feature_ref, data_ref),
                    producer_identity=self.readout_provider.identity,
                    producer=lambda spec=spec, fit_ref=fit_ref, feature_ref=feature_ref, data_ref=data_ref: self.readout_provider.evaluate(
                        spec, fit_ref.payload, feature_ref.payload, data_ref.payload
                    ),
                )
                refs[_key("evaluation", spec.id)] = ref
                last_path = ref.path
            for spec in plan.comparisons:
                if spec.study_id not in selected:
                    continue
                current_stage, current_node = Stage.ANALYZE, spec.id
                evaluation_refs = tuple(refs[_key("evaluation", item)] for item in spec.evaluation_ids)
                ref = self.store.execute_node(
                    stage=current_stage,
                    node_id=spec.id,
                    spec=spec,
                    upstream=evaluation_refs,
                    producer_identity=self.readout_provider.identity,
                    producer=lambda spec=spec, evaluation_refs=evaluation_refs: self.readout_provider.compare(
                        spec, [item.payload for item in evaluation_refs]
                    ),
                )
                refs[_key("comparison", spec.id)] = ref
                last_path = ref.path
            if through is Stage.ANALYZE:
                return _result(
                    plan, through, last_path, refs, selected_studies, finalization_scope
                )

            final_node = "experiment" if study is None else f"study/{study}"
            current_stage, current_node = Stage.FINALIZE, final_node
            selected_comparisons = tuple(
                spec for spec in plan.comparisons if spec.study_id in selected
            )
            comparison_refs = tuple(
                refs[_key("comparison", spec.id)] for spec in selected_comparisons
            )
            final_ref = self.store.execute_node(
                stage=current_stage,
                node_id=final_node,
                spec={
                    "scope": finalization_scope,
                    "studies": selected_studies,
                    "campaign": plan.campaign.value,
                    "repetition": plan.repetition.index,
                    "plan_digest": plan.digest,
                },
                upstream=comparison_refs,
                producer_identity=_FINALIZER_IDENTITY,
                producer=lambda: {
                    "status": "complete",
                    "campaign": plan.campaign.value,
                    "repetition": plan.repetition.index,
                    "manifest_digest": plan.manifest_digest,
                    "plan_digest": plan.digest,
                    "scope": finalization_scope,
                    "studies": selected_studies,
                    "comparison_count": len(selected_comparisons),
                    "comparison_digests": [item.digest for item in comparison_refs],
                },
            )
            refs[_key("final", final_node)] = final_ref
            return _result(
                plan,
                Stage.FINALIZE,
                final_ref.path,
                refs,
                selected_studies,
                finalization_scope,
            )
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
    selected_studies: tuple[str, ...],
    finalization_scope: str,
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
        selected_studies=selected_studies,
        finalization_scope=finalization_scope,
    )

