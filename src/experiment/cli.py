"""Command-line interface for empirical-evaluation validation and execution."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from .artifact_store import LocalArtifactStore
from .domain import CampaignId, Stage, json_value
from .manifest import load_manifest
from .numerical import NumericalProviderBundle
from .planning import plan_experiment
from .providers import FakeProviders
from .runner import ExperimentRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quark-experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser(
        "validate", help="Validate a run manifest and executable contract."
    )
    validate.add_argument("manifest", type=Path)
    plan = subparsers.add_parser("plan", help="Expand deterministic experiment plans.")
    plan.add_argument("manifest", type=Path)
    plan.add_argument("--campaign", choices=[item.value for item in CampaignId])
    plan.add_argument("--repetition", type=int, default=0)
    plan.add_argument("--full", action="store_true", help="Print the complete plan graph.")

    run = subparsers.add_parser("run", help="Execute a complete or study-scoped plan.")
    run.add_argument("manifest", type=Path)
    run.add_argument(
        "--campaign", required=True, choices=[item.value for item in CampaignId]
    )
    run.add_argument("--repetition", type=int, default=0)
    run.add_argument(
        "--through",
        choices=[item.value for item in Stage],
        default=Stage.FINALIZE.value,
    )
    run.add_argument("--provider", choices=("fake", "numerical"), default="fake")
    run.add_argument("--backend", choices=("nvidia", "aer"), default="nvidia")
    run.add_argument("--study")
    run.add_argument("--gpu-id", type=int, default=0)
    run.add_argument("--chunk-size", type=int)

    aggregate = subparsers.add_parser(
        "aggregate", help="Emit checksum-validated comparison artifacts."
    )
    aggregate.add_argument("artifact_root", type=Path)
    aggregate.add_argument("--campaign", choices=[item.value for item in CampaignId])
    aggregate.add_argument("--study")
    aggregate.add_argument("--format", choices=("json", "csv"), default="json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "aggregate":
        rows = LocalArtifactStore.aggregate_comparisons(
            args.artifact_root,
            campaign=args.campaign,
            study=args.study,
        )
        if args.format == "csv":
            _write_csv(rows)
        else:
            _print(rows)
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

    campaigns = (
        (CampaignId(args.campaign),) if args.campaign else manifest.campaigns
    )
    plans = [
        plan_experiment(manifest, campaign, args.repetition)
        for campaign in campaigns
    ]
    if args.command == "plan":
        _print(
            [plan.to_dict() for plan in plans]
            if args.full
            else [_plan_summary(plan) for plan in plans]
        )
        return 0

    plan = plans[0]
    if args.provider == "numerical":
        bundle = NumericalProviderBundle.create(
            manifest,
            backend=args.backend,
            gpu_id=args.gpu_id,
            chunk_size=args.chunk_size,
        )
        # Preflight happens before constructing a store or writing any artifact.
        bundle.preflight(plan, study=args.study)
        data_provider = bundle.data
        representation_provider = bundle.representation
        readout_provider = bundle.readout
    else:
        providers = FakeProviders()
        data_provider = providers
        representation_provider = providers
        readout_provider = providers

    store = LocalArtifactStore(manifest.artifact_root, plan)
    runner = ExperimentRunner(
        data_provider=data_provider,
        representation_provider=representation_provider,
        readout_provider=readout_provider,
        artifact_store=store,
    )
    _print(
        json_value(
            runner.run(
                plan,
                through=Stage(args.through),
                study=args.study,
            )
        )
    )
    return 0


def _plan_summary(plan) -> dict[str, object]:
    return {
        "campaign": plan.campaign.value,
        "repetition": plan.repetition.index,
        "manifest_digest": plan.manifest_digest,
        "plan_digest": plan.digest,
        "studies": list(plan.study_ids),
        "nodes": {
            "data": len(plan.data),
            "acquisitions": len(plan.acquisitions),
            "feature_views": len(plan.feature_views),
            "fits": len(plan.fits),
            "evaluations": len(plan.evaluations),
            "comparisons": len(plan.comparisons),
        },
    }


def _write_csv(rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: (
                    json.dumps(json_value(value), sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            }
        )


def _print(value: object) -> None:
    print(json.dumps(json_value(value), indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
