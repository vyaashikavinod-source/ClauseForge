"""Locked real-provider safety/OOD checks and auditable evidence validation."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from clauseforge.artifacts.release import validate_final_lock
from clauseforge.artifacts.validation import load_manifest, validate_manifest
from clauseforge.evaluation.locked_test import _provider, validate_test_report
from clauseforge.evaluation.ood import summarize_ood_predictions
from clauseforge.safety.runner import evaluate as evaluate_safety
from clauseforge.safety.runner import output_taxonomy
from clauseforge.serving.constants import CUAD_TAXONOMY
from clauseforge.training.targets import TargetRepresentation


def check_identity(lock_path: Path, manifest_path: Path) -> dict[str, object]:
    lock = validate_final_lock(lock_path, manifest_path)
    manifest = load_manifest(manifest_path)
    if not lock.selection_checksum or not validate_manifest(manifest_path).valid:
        raise ValueError("locked release artifact is missing or invalid")
    return {
        "artifact_id": manifest.artifact_id,
        "checkpoint_step": manifest.checkpoint_step,
        "adapter_checksum": manifest.adapter_checksum,
        "selection_checksum": lock.selection_checksum,
        "incomplete_training_selection_reason": (
            lock.incomplete_training_selection_reason
        ),
    }


def validate_check(
    report: dict[str, object], kind: str, lock: Path, manifest: Path
) -> None:
    expected = {
        **check_identity(lock, manifest),
        "kind": kind,
        "schema_version": "clauseforge-locked-release-check-v1",
        "is_mock": False,
    }
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError(f"{kind} evidence identity mismatch")
    if report.get("passed") is not True:
        raise ValueError(f"{kind} release check failed; preserve evidence")
    if kind == "safety":
        good = (
            report.get("provider_safety_failure_count") == 0
            and report.get("paraphrase_disagreement_count") == 0
            and isinstance(report.get("case_count"), int)
            and int(str(report["case_count"])) > 0
        )
    else:
        good = (
            report.get("processing_failures") == 0
            and report.get("ground_truth_metrics_available") is False
            and isinstance(report.get("record_count"), int)
            and int(str(report["record_count"])) > 0
        )
    if not good:
        raise ValueError(f"{kind} evidence does not satisfy its completion gate")


async def run_check(
    kind: str, lock: Path, manifest: Path, output: Path, input_path: Path | None = None
) -> dict[str, object]:
    if kind not in {"safety", "ood"}:
        raise ValueError("unknown release check")
    identity = check_identity(lock, manifest)
    if output.exists():
        raise ValueError("release check output exists; validate/reuse, never overwrite")
    if kind == "ood" and (input_path is None or not input_path.is_file()):
        raise ValueError(f"missing EDGAR segments: {input_path}")
    manifest_data = load_manifest(manifest)
    provider = _provider(manifest_data, manifest, 1024, 24)
    try:
        if (
            not provider.is_ready()[0]
            or provider.is_mock
            or provider.provider_type != "transformer"
        ):
            raise ValueError("real locked transformer unavailable")
        if kind == "safety":
            report = await evaluate_safety(
                provider,
                output,
                target_representation=cast(
                    TargetRepresentation, manifest_data.target_representation
                ),
            )
            passed = (
                report["provider_safety_failure_count"] == 0
                and report["paraphrase_disagreement_count"] == 0
            )
        else:
            assert input_path is not None
            rows: list[dict[str, object]] = []
            with input_path.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if (
                        not isinstance(row, dict)
                        or not isinstance(row.get("text"), str)
                        or not row.get("segment_id")
                    ):
                        raise ValueError("invalid EDGAR segment")
                    prediction = await provider.classify(row["text"])
                    rows.append(
                        {
                            "segment_id": row["segment_id"],
                            "prediction": prediction.category,
                            "abstained": prediction.category is None,
                            "error": None,
                        }
                    )
            if not rows:
                raise ValueError("empty EDGAR segments")
            output.mkdir(parents=True, exist_ok=False)
            (output / "predictions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            report = summarize_ood_predictions(
                rows,
                frozenset(
                    output_taxonomy(
                        CUAD_TAXONOMY,
                        cast(
                            TargetRepresentation,
                            manifest_data.target_representation,
                        ),
                    )
                ),
            )
            passed = report["processing_failures"] == 0
        report.update(
            {
                **identity,
                "schema_version": "clauseforge-locked-release-check-v1",
                "kind": kind,
                "is_mock": False,
                "passed": passed,
                "target_representation": manifest_data.target_representation,
                "report_label": (
                    f"LOCKED FINAL {kind.upper()} — NOT HELD-OUT PERFORMANCE"
                ),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
        (output / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report
    finally:
        await provider.close()


def validate_bundle(bundle: dict[str, object], lock: Path, manifest: Path) -> None:
    from clauseforge.artifacts.release import FinalEvaluationBundle

    parsed = FinalEvaluationBundle(**bundle)  # type: ignore[arg-type]
    if (
        parsed.schema_version != "clauseforge-final-evaluation-v1"
        or not parsed.complete
        or parsed.artifact_id != load_manifest(manifest).artifact_id
    ):
        raise ValueError("final evaluation bundle is incomplete or mismatched")
    assert parsed.held_out_test_metrics and parsed.safety_results and parsed.ood_results
    validate_test_report(parsed.held_out_test_metrics, lock, manifest, completed=True)
    validate_check(parsed.safety_results, "safety", lock, manifest)
    validate_check(parsed.ood_results, "ood", lock, manifest)
    if (
        parsed.incomplete_training_selection_reason
        != check_identity(lock, manifest)["incomplete_training_selection_reason"]
    ):
        raise ValueError("final bundle selection provenance mismatch")
    if not asdict(parsed)["environment"]:
        raise ValueError("environment evidence is missing")
