"""CPU-safe plans for import, lineage, deployment, and final validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from clauseforge.artifacts.models import ArtifactManifest
from clauseforge.artifacts.validation import (
    load_manifest,
    sha256_file,
    sha256_path,
    validate_manifest,
)


@dataclass(frozen=True, slots=True)
class ImportReport:
    valid: bool
    artifact_id: str
    adapter: str
    checksum: str
    copied: bool = False


def validate_import(adapter: Path, manifest_path: Path) -> ImportReport:
    manifest = load_manifest(manifest_path)
    if manifest.artifact_type != "adapter":
        raise ValueError("import requires an adapter artifact manifest")
    if not adapter.exists():
        raise ValueError("adapter does not exist")
    checksum = sha256_path(adapter)
    if manifest.adapter_checksum != checksum:
        raise ValueError("adapter checksum mismatch")
    report = validate_manifest(manifest_path, require_files=False)
    if not report.valid:
        raise ValueError("invalid manifest: " + "; ".join(report.errors))
    return ImportReport(True, manifest.artifact_id, str(adapter.resolve()), checksum)


def validate_parent_lineage(child: ArtifactManifest, parent_path: Path) -> None:
    parent = load_manifest(parent_path)
    if child.parent_artifact_id != parent.artifact_id:
        raise ValueError("parent artifact identity mismatch")
    if child.parent_artifact_checksum != sha256_file(parent_path):
        raise ValueError("parent artifact checksum mismatch")
    identity_fields = (
        "base_model",
        "base_revision",
        "adapter_experiment_id",
        "target_representation_version",
        "prompt_version",
        "taxonomy_version",
        "stable_id_map_checksum",
    )
    mismatches = [
        name
        for name in identity_fields
        if getattr(child, name) != getattr(parent, name)
    ]
    if mismatches:
        raise ValueError("lineage incompatibility: " + ", ".join(mismatches))


def final_validation_plan(
    manifest_path: Path,
    *,
    authorize_test: bool = False,
    lock_path: Path | None = None,
) -> dict[str, object]:
    report = validate_manifest(manifest_path)
    if not report.valid:
        raise ValueError("artifact validation failed: " + "; ".join(report.errors))
    manifest = load_manifest(manifest_path)
    if manifest.release_status != "final_candidate":
        raise ValueError("final validation requires a locked final_candidate")
    test_step = "held_out_test_blocked"
    if authorize_test:
        if lock_path is None:
            raise ValueError("authorized test requires a final lock")
        from clauseforge.artifacts.release import validate_final_lock

        lock = validate_final_lock(lock_path, manifest_path)
        if not lock.test_authorized or lock.test_evaluated or manifest.test_evaluated:
            raise ValueError("held-out test is not authorized or was already evaluated")
        test_step = "held_out_test_once"
    return {
        "artifact_id": manifest.artifact_id,
        "executed": False,
        "test_authorized": authorize_test,
        "steps": [
            "validate_artifact",
            test_step,
            "final_safety_harness",
            "final_edgar_ood",
            "build_final_evaluation_bundle",
        ],
    }


def build_deployment_bundle(
    output: Path,
    model_manifest: Path,
    deployment_manifest: dict[str, object],
    documentation: str,
) -> Path:
    model_report = validate_manifest(model_manifest)
    if not model_report.valid:
        raise ValueError("model artifact is invalid")
    required = {"artifact_id", "backend", "quantization", "model_manifest_checksum"}
    if required - deployment_manifest.keys():
        raise ValueError("deployment manifest is incomplete")
    if deployment_manifest["artifact_id"] != model_report.artifact_id:
        raise ValueError("deployment/model artifact identity mismatch")
    if deployment_manifest["model_manifest_checksum"] != sha256_file(model_manifest):
        raise ValueError("deployment/model checksum mismatch")
    output.mkdir(parents=True, exist_ok=False)
    model_target = output / "model_manifest.json"
    model_target.write_text(
        model_manifest.read_text(encoding="utf-8"), encoding="utf-8"
    )
    deployment_target = output / "deployment_manifest.json"
    deployment_target.write_text(
        json.dumps(deployment_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "checksums.json").write_text(
        json.dumps(
            {
                "model_manifest.json": sha256_file(model_target),
                "deployment_manifest.json": sha256_file(deployment_target),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    docs = output / "docs"
    docs.mkdir()
    (docs / "README.md").write_text(documentation, encoding="utf-8")
    (output / "config").mkdir()
    return output


def serialize_import(report: ImportReport) -> str:
    return json.dumps(asdict(report), indent=2, sort_keys=True)
