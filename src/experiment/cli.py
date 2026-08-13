"""Command-line interface for empirical-evaluation validation and execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .artifact_store import LocalArtifactStore
from .domain import CampaignId, Stage, json_value
from .manifest import load_manifest
from .planning import plan_experiment
from .providers import FakeProviders
from .runner import ExperimentRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quark-experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="Validate a run manifest and executable contract.")
    validate.add_argument("manifest", type=Path)
    plan = subparsers.add_parser("plan", help="Expand deterministic experiment plans.")
    plan.add_argument("manifest", type=Path)
    plan.add_argument("--campaign", choices=[item.value for item in CampaignId])
    plan.add_argument("--repetition", type=int, default=0)
    plan.add_argument("--full", action="store_true", help="Print the complete plan graph.")
    run = subparsers.add_parser("run", help="Execute a plan with the milestone fake providers.")
    run.add_argument("manifest", type=Path)
    run.add_argument("--campaign", required=True, choices=[item.value for item in CampaignId])
    run.add_argument("--repetition", type=int, default=0)
    run.add_argument("--through", choices=[item.value for item in Stage], default=Stage.FINALIZE.value)
    aggregate = subparsers.add_parser("aggregate", help="Index finalized immutable run artifacts.")
    aggregate.add_argument("artifact_root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "aggregate":
        print(json.dumps(LocalArtifactStore.aggregate(args.artifact_root), indent=2, sort_keys=True))
        return 0
    manifest = load_manifest(args.manifest)
    if args.command == "validate":
        _print(
            {
                "status": "valid",
                "suite": manifest.suite,
                "mode": manifest.mode,
                "manifest_digest": manifest.digest,
                "campaigns": [item.value for item in manifest.campaigns],
            }
        )
        return 0
    campaigns = (CampaignId(args.campaign),) if args.campaign else manifest.campaigns
    plans = [plan_experiment(manifest, campaign, args.repetition) for campaign in campaigns]
    if args.command == "plan":
        _print([plan.to_dict() for plan in plans] if args.full else [_plan_summary(plan) for plan in plans])
        return 0
    plan = plans[0]
    providers = FakeProviders()
    store = LocalArtifactStore(manifest.artifact_root, plan)
    runner = ExperimentRunner(
        data_provider=providers,
        representation_provider=providers,
        readout_provider=providers,
        artifact_store=store,
    )
    _print(json_value(runner.run(plan, through=Stage(args.through))))
    return 0


def _plan_summary(plan) -> dict[str, object]:
    return {
        "campaign": plan.campaign.value,
        "repetition": plan.repetition.index,
        "manifest_digest": plan.manifest_digest,
        "plan_digest": plan.digest,
        "nodes": {
            "data": len(plan.data),
            "acquisitions": len(plan.acquisitions),
            "feature_views": len(plan.feature_views),
            "fits": len(plan.fits),
            "evaluations": len(plan.evaluations),
            "comparisons": len(plan.comparisons),
        },
    }


def _print(value: object) -> None:
    print(json.dumps(json_value(value), indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())

