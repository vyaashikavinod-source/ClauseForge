"""Print a non-executing final-validation plan; never opens datasets itself."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

from clauseforge.artifacts.release import (
    FinalEvaluationBundle,
    mark_manifest_test_evaluated,
    record_test_evaluated,
)
from clauseforge.artifacts.validation import load_manifest, write_manifest
from clauseforge.artifacts.workflows import final_validation_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--authorize-held-out-test", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--test-command-json", type=Path)
    parser.add_argument("--safety-command-json", type=Path)
    parser.add_argument("--ood-command-json", type=Path)
    parser.add_argument("--test-report", type=Path)
    parser.add_argument("--safety-report", type=Path)
    parser.add_argument("--ood-report", type=Path)
    parser.add_argument("--environment-report", type=Path)
    parser.add_argument("--output-bundle", type=Path)
    args = parser.parse_args(argv)
    plan = final_validation_plan(
        args.artifact_manifest,
        authorize_test=args.authorize_held_out_test,
        lock_path=args.lock,
    )
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    required = (
        args.lock,
        args.test_command_json,
        args.safety_command_json,
        args.ood_command_json,
        args.test_report,
        args.safety_report,
        args.ood_report,
        args.environment_report,
        args.output_bundle,
    )
    if not args.authorize_held_out_test or any(value is None for value in required):
        raise ValueError(
            "execution requires authorization, lock, commands, reports, "
            "environment, and output"
        )
    assert args.lock is not None and args.output_bundle is not None
    commands = (args.test_command_json, args.safety_command_json, args.ood_command_json)
    reports = (args.test_report, args.safety_report, args.ood_report)
    results: list[dict[str, object]] = []
    for command_path, report_path in zip(commands, reports, strict=True):
        assert command_path is not None and report_path is not None
        command = json.loads(command_path.read_text(encoding="utf-8"))
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) for item in command)
        ):
            raise ValueError(
                "each execution command must be a non-empty JSON string array"
            )
        subprocess.run(command, check=True, shell=False)
        result = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("final validation report must be a JSON object")
        results.append(result)
        manifest = load_manifest(args.artifact_manifest)
        if len(results) == 1:
            record_test_evaluated(args.lock)
            mark_manifest_test_evaluated(args.artifact_manifest)
        elif len(results) == 2:
            write_manifest(
                args.artifact_manifest, replace(manifest, final_safety_evaluated=True)
            )
        else:
            write_manifest(
                args.artifact_manifest, replace(manifest, final_ood_evaluated=True)
            )
    assert args.environment_report is not None
    environment = json.loads(args.environment_report.read_text(encoding="utf-8"))
    if not isinstance(environment, dict):
        raise ValueError("environment report must be a JSON object")
    manifest = load_manifest(args.artifact_manifest)
    validation = (
        asdict(manifest.validation_summary) if manifest.validation_summary else {}
    )
    bundle = FinalEvaluationBundle(
        "clauseforge-final-evaluation-v1",
        manifest.artifact_id,
        manifest.training_commit,
        manifest.created_at,
        validation,
        results[0],
        results[1],
        results[2],
        environment,
        manifest.known_limitations,
    )
    args.output_bundle.parent.mkdir(parents=True, exist_ok=True)
    args.output_bundle.write_text(
        json.dumps(asdict(bundle), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output_bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
