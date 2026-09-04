"""Final lock, one-time test authorization, and evaluation bundle schemas."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from clauseforge.artifacts.models import ArtifactManifest
from clauseforge.artifacts.validation import (
    load_manifest,
    sha256_file,
    validate_manifest,
    write_manifest,
)


@dataclass(frozen=True, slots=True)
class FinalModelLock:
    schema_version: str
    artifact_id: str
    experiment_id: str
    training_commit: str
    config_checksum: str
    taxonomy_version: str
    prompt_version: str
    target_representation_version: str
    manifest_checksum: str
    locked_at: str
    test_authorized: bool = False
    test_evaluated: bool = False
    incomplete_training_selection_reason: str | None = None
    full_training_completed: bool = True
    checkpoint_step: int | None = None
    adapter_checksum: str | None = None
    selection_checksum: str | None = None
    validation_report_checksum: str | None = None


@dataclass(frozen=True, slots=True)
class FinalEvaluationBundle:
    schema_version: str
    artifact_id: str
    commit: str
    timestamp: str
    validation_metrics: dict[str, object]
    held_out_test_metrics: dict[str, object] | None
    safety_results: dict[str, object] | None
    ood_results: dict[str, object] | None
    environment: dict[str, object]
    limitations: tuple[str, ...]
    incomplete_training_selection_reason: str | None = None

    @property
    def complete(self) -> bool:
        return all((self.held_out_test_metrics, self.safety_results, self.ood_results))


def lock_final_model(
    manifest_path: Path,
    validation_report: Path,
    output: Path,
    locked_at: str,
    *,
    incomplete_training_selection_reason: str | None = None,
    selected_artifact_id: str | None = None,
) -> FinalModelLock:
    if output.exists():
        raise ValueError("final lock already exists; refusing to reset test protection")
    manifest = load_manifest(manifest_path)
    if manifest.validation_summary is None:
        raise ValueError("final lock requires full training and validation")
    if not manifest.full_training_completed:
        _validate_incomplete_selection(
            manifest_path,
            validation_report,
            incomplete_training_selection_reason,
            selected_artifact_id,
        )
    elif incomplete_training_selection_reason is not None:
        raise ValueError(
            "incomplete-training authorization is only for incomplete training"
        )
    if not validation_report.is_file():
        raise ValueError("validation report is required")
    lock = FinalModelLock(
        "clauseforge-final-lock-v1",
        manifest.artifact_id,
        manifest.adapter_experiment_id,
        manifest.training_commit,
        manifest.config_checksum,
        manifest.taxonomy_version,
        manifest.prompt_version,
        manifest.target_representation_version,
        sha256_file(manifest_path),
        locked_at,
        incomplete_training_selection_reason=incomplete_training_selection_reason,
        full_training_completed=manifest.full_training_completed,
        checkpoint_step=manifest.checkpoint_step,
        adapter_checksum=manifest.adapter_checksum,
        selection_checksum=selection_checksum(manifest),
        validation_report_checksum=sha256_file(validation_report),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(lock), indent=2, sort_keys=True) + "\n")
    return lock


def selection_checksum(manifest: ArtifactManifest) -> str:
    """Freeze identity and selection evidence; exclude later release-gate results."""
    values = manifest.to_dict()
    for field in (
        "test_evaluated",
        "final_safety_evaluated",
        "final_ood_evaluated",
        "quantized",
        "deployed",
        "real_serving_benchmark_completed",
        "container_smoke_test_completed",
        "deployment_manifest_valid",
        "release_checklist_complete",
        "release_status",
        "final_release",
    ):
        values.pop(field)
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def approved_incomplete_selection(
    lock: FinalModelLock, manifest: ArtifactManifest
) -> bool:
    """An explicit lock grants only the training-completion gate, for this identity."""
    return bool(
        lock.incomplete_training_selection_reason
        and lock.incomplete_training_selection_reason.strip()
        and not lock.full_training_completed
        and not manifest.full_training_completed
        and lock.artifact_id == manifest.artifact_id
        and lock.checkpoint_step == manifest.checkpoint_step
        and lock.adapter_checksum == manifest.adapter_checksum
        and lock.validation_report_checksum
        and lock.selection_checksum == selection_checksum(manifest)
    )


def _validate_incomplete_selection(
    manifest_path: Path,
    report_path: Path,
    reason: str | None,
    selected_id: str | None,
) -> None:
    if not reason or not reason.strip():
        raise ValueError(
            "incomplete training requires explicit release authorization and reason"
        )
    manifest = load_manifest(manifest_path)
    summary = manifest.validation_summary
    if (
        selected_id != manifest.artifact_id
        or manifest.release_status != "release_candidate"
        or not manifest.validation_selection_completed
        or manifest.test_evaluated
        or summary is None
        or summary.split != "validation"
        or summary.validation_examples <= 0
        or not manifest.adapter_path
        or not manifest.adapter_checksum
    ):
        raise ValueError(
            "incomplete selection requires a fixed validation-only release_candidate"
        )
    validation = validate_manifest(manifest_path)
    if not validation.valid:
        raise ValueError(
            "selected artifact validation failed: " + "; ".join(validation.errors)
        )
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    expected = {
        "split": "validation",
        "test_evaluated": False,
        "checkpoint_step": manifest.checkpoint_step,
        "experiment_id": manifest.adapter_experiment_id,
        "prompt_version": manifest.prompt_version,
        "target_representation": manifest.target_representation,
        "target_representation_version": manifest.target_representation_version,
        "stable_id_map_checksum": manifest.stable_id_map_checksum,
        "accuracy": summary.accuracy,
        "macro_f1": summary.macro_f1,
        "weighted_f1": summary.weighted_f1,
        "invalid_output_rate": summary.invalid_output_rate,
        "validation_loss": summary.validation_loss,
    }
    if not isinstance(raw, dict) or any(raw.get(k) != v for k, v in expected.items()):
        raise ValueError("validation evidence does not match selected candidate")
    if (
        raw.get("total_examples", raw.get("validation_examples"))
        != summary.validation_examples
        or raw.get("exact_id_rate", raw.get("exact_id_output_rate"))
        != summary.exact_id_rate
        or ("artifact_id" in raw and raw["artifact_id"] != manifest.artifact_id)
    ):
        raise ValueError("validation evidence counts or identity mismatch")


def authorize_test_once(lock_path: Path) -> FinalModelLock:
    raw = json.loads(lock_path.read_text(encoding="utf-8"))
    lock = FinalModelLock(**raw)
    if lock.test_authorized or lock.test_evaluated:
        raise ValueError("held-out test is already authorized or evaluated")
    updated = replace(lock, test_authorized=True)
    lock_path.write_text(
        json.dumps(asdict(updated), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return updated


def record_test_evaluated(lock_path: Path) -> FinalModelLock:
    """Consume an existing authorization after the test result is persisted."""
    raw = json.loads(lock_path.read_text(encoding="utf-8"))
    lock = FinalModelLock(**raw)
    if not lock.test_authorized or lock.test_evaluated:
        raise ValueError("held-out test is not authorized or was already evaluated")
    updated = replace(lock, test_authorized=False, test_evaluated=True)
    lock_path.write_text(
        json.dumps(asdict(updated), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return updated


def validate_final_lock(lock_path: Path, manifest_path: Path) -> FinalModelLock:
    """Reject identity or configuration changes after candidate locking."""
    raw = json.loads(lock_path.read_text(encoding="utf-8"))
    lock = FinalModelLock(**raw)
    manifest = load_manifest(manifest_path)
    values = (
        (lock.artifact_id, manifest.artifact_id),
        (lock.experiment_id, manifest.adapter_experiment_id),
        (lock.training_commit, manifest.training_commit),
        (lock.config_checksum, manifest.config_checksum),
        (lock.taxonomy_version, manifest.taxonomy_version),
        (lock.prompt_version, manifest.prompt_version),
        (
            lock.target_representation_version,
            manifest.target_representation_version,
        ),
    )
    if any(expected != current for expected, current in values):
        raise ValueError("locked candidate identity or configuration changed")
    if lock.selection_checksum and lock.selection_checksum != selection_checksum(
        manifest
    ):
        raise ValueError("locked candidate selection or lineage changed")
    if (
        lock.incomplete_training_selection_reason is not None
        and not approved_incomplete_selection(lock, manifest)
    ):
        raise ValueError("invalid incomplete-training selection provenance")
    return lock


def mark_manifest_test_evaluated(path: Path) -> ArtifactManifest:
    manifest = load_manifest(path)
    if manifest.test_evaluated:
        raise ValueError("held-out test has already been evaluated")
    updated = replace(manifest, test_evaluated=True)
    write_manifest(path, updated)
    return updated


def authorize_deployment(path: Path) -> ArtifactManifest:
    """Record deployment authorization only after every preceding release gate."""
    manifest = load_manifest(path)
    checks = {
        "final candidate": manifest.release_status == "final_candidate",
        "test": manifest.test_evaluated,
        "safety": manifest.final_safety_evaluated,
        "OOD": manifest.final_ood_evaluated,
        "quantization": manifest.quantized,
        "deployment bundle": manifest.artifact_type == "deployment_bundle",
        "benchmark": manifest.real_serving_benchmark_completed,
        "container smoke": manifest.container_smoke_test_completed,
        "deployment manifest": manifest.deployment_manifest_valid,
        "release checklist": manifest.release_checklist_complete,
    }
    blocked = [name for name, complete in checks.items() if not complete]
    if blocked:
        raise ValueError("deployment authorization blocked: " + ", ".join(blocked))
    updated = replace(manifest, deployed=True)
    write_manifest(path, updated)
    return updated
