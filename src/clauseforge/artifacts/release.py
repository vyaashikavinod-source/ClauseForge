"""Final lock, one-time test authorization, and evaluation bundle schemas."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from clauseforge.artifacts.models import ArtifactManifest
from clauseforge.artifacts.validation import load_manifest, sha256_file, write_manifest


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

    @property
    def complete(self) -> bool:
        return all((self.held_out_test_metrics, self.safety_results, self.ood_results))


def lock_final_model(
    manifest_path: Path, validation_report: Path, output: Path, locked_at: str
) -> FinalModelLock:
    manifest = load_manifest(manifest_path)
    if not manifest.full_training_completed or manifest.validation_summary is None:
        raise ValueError("final lock requires full training and validation")
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
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(asdict(lock), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return lock


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
