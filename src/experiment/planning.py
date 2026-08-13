"""Public campaign-planning entry point."""

from __future__ import annotations

from typing import Protocol

from .campaigns import plan_campaign_i, plan_campaign_ii, plan_campaign_iii
from .domain import CampaignId, ExperimentPlan, RepetitionSpec
from .manifest import RunManifest


class CampaignPlanner(Protocol):
    def __call__(self, manifest: RunManifest, repetition: RepetitionSpec) -> ExperimentPlan:
        ...


_PLANNERS: dict[CampaignId, CampaignPlanner] = {
    CampaignId.REPRESENTATION: plan_campaign_i,
    CampaignId.FIDELITY: plan_campaign_ii,
    CampaignId.FROZEN_POOL: plan_campaign_iii,
}


def plan_experiment(
    manifest: RunManifest,
    campaign: CampaignId | str,
    repetition: RepetitionSpec | int,
) -> ExperimentPlan:
    campaign_id = CampaignId(campaign)
    if campaign_id not in manifest.campaigns:
        raise ValueError(f"Campaign {campaign_id.value!r} is not enabled by the manifest.")
    if isinstance(repetition, int):
        if repetition < 0:
            raise ValueError("Repetition index must be nonnegative.")
        choices = manifest.repetitions_for(campaign_id)
        try:
            repetition_spec = choices[int(repetition)]
        except IndexError as exc:
            raise ValueError(f"Repetition index {repetition} is not declared for {campaign_id.value}.") from exc
    else:
        repetition_spec = repetition
        if repetition_spec not in manifest.repetitions_for(campaign_id):
            raise ValueError("Repetition is not declared for this campaign.")
    return _PLANNERS[campaign_id](manifest, repetition_spec)

