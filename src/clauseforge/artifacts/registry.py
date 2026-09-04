"""Portable JSON registry, promotion gates, active pointer, and rollback."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from clauseforge.artifacts.models import ArtifactManifest, ReleaseStatus
from clauseforge.artifacts.release import (
    FinalModelLock,
    approved_incomplete_selection,
    validate_final_lock,
)
from clauseforge.artifacts.validation import (
    load_manifest,
    validate_manifest,
    write_manifest,
)

_TRANSITIONS: dict[ReleaseStatus, ReleaseStatus] = {
    "pilot": "release_candidate",
    "release_candidate": "final_candidate",
    "final_candidate": "released",
}


@dataclass(frozen=True, slots=True)
class GateReport:
    allowed: bool
    target_status: ReleaseStatus
    blocking_gates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ActiveModelPointer:
    schema_version: str
    current_manifest: str
    current_artifact_id: str
    current_manifest_checksum: str
    previous_manifest: str | None = None
    previous_artifact_id: str | None = None
    previous_manifest_checksum: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "current_manifest": self.current_manifest,
            "current_artifact_id": self.current_artifact_id,
            "current_manifest_checksum": self.current_manifest_checksum,
            "previous_manifest": self.previous_manifest,
            "previous_artifact_id": self.previous_artifact_id,
            "previous_manifest_checksum": self.previous_manifest_checksum,
        }


def promotion_gates(
    manifest: ArtifactManifest,
    target: ReleaseStatus,
    *,
    lock: FinalModelLock | None = None,
) -> GateReport:
    expected = _TRANSITIONS.get(manifest.release_status)
    blocked: list[str] = []
    if expected != target:
        blocked.append(f"invalid transition {manifest.release_status}->{target}")
    if target == "final_candidate":
        checks = {
            "full training completed": manifest.full_training_completed
            or (lock is not None and approved_incomplete_selection(lock, manifest)),
            "model config locked": bool(manifest.config_checksum),
            "validation selection completed": manifest.validation_selection_completed,
            "held-out test evaluated once": manifest.test_evaluated,
            "final safety evaluation completed": manifest.final_safety_evaluated,
            "final EDGAR OOD evaluation completed": manifest.final_ood_evaluated,
        }
        blocked.extend(name for name, complete in checks.items() if not complete)
    elif target == "released":
        checks = {
            "merged/quantized artifact validated": manifest.quantized,
            "deployment bundle": manifest.artifact_type == "deployment_bundle",
            "real serving benchmark completed": (
                manifest.real_serving_benchmark_completed
            ),
            "container smoke test completed": manifest.container_smoke_test_completed,
            "deployment manifest valid": manifest.deployment_manifest_valid,
            "release checklist complete": manifest.release_checklist_complete,
            "deployment authorized": manifest.deployed,
        }
        blocked.extend(name for name, complete in checks.items() if not complete)
    return GateReport(not blocked, target, tuple(blocked))


class ArtifactRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.manifests = self.root / "manifests"
        self.active_path = self.root / "active.json"

    def register(self, source: Path) -> Path:
        report = validate_manifest(source)
        if not report.valid:
            raise ValueError("invalid artifact: " + "; ".join(report.errors))
        manifest = load_manifest(source)
        destination = self.manifests / f"{manifest.artifact_id}.json"
        if destination.exists():
            raise ValueError("artifact_id already registered")
        write_manifest(destination, manifest)
        return destination

    def list(self) -> tuple[ArtifactManifest, ...]:
        if not self.manifests.exists():
            return ()
        return tuple(
            load_manifest(path) for path in sorted(self.manifests.glob("*.json"))
        )

    def inspect(self, artifact_id: str) -> ArtifactManifest:
        return load_manifest(self.manifests / f"{artifact_id}.json")

    def promote(
        self,
        artifact_id: str,
        target: ReleaseStatus,
        *,
        lock_path: Path | None = None,
    ) -> ArtifactManifest:
        path = self.manifests / f"{artifact_id}.json"
        report = validate_manifest(path)
        if not report.valid:
            raise ValueError("promotion blocked: artifact or lineage is invalid")
        manifest = load_manifest(path)
        lock = validate_final_lock(lock_path, path) if lock_path is not None else None
        gates = promotion_gates(manifest, target, lock=lock)
        if not gates.allowed:
            raise ValueError("promotion blocked: " + "; ".join(gates.blocking_gates))
        promoted = replace(
            manifest, release_status=target, final_release=target == "released"
        )
        write_manifest(path, promoted)
        return promoted

    def activate(self, artifact_id: str) -> ActiveModelPointer:
        from clauseforge.artifacts.validation import sha256_file

        manifest_path = self.manifests / f"{artifact_id}.json"
        report = validate_manifest(manifest_path)
        if not report.valid:
            raise ValueError("cannot activate invalid artifact")
        manifest = load_manifest(manifest_path)
        if (
            manifest.artifact_type == "adapter"
            and manifest.release_status == "release_candidate"
            and manifest.validation_summary is None
        ):
            raise ValueError(
                "cannot activate trained candidate without validation evidence"
            )
        previous = self.read_active() if self.active_path.exists() else None
        pointer = ActiveModelPointer(
            "clauseforge-active-model-v1",
            str(manifest_path.relative_to(self.root)).replace("\\", "/"),
            artifact_id,
            sha256_file(manifest_path),
            previous.current_manifest if previous else None,
            previous.current_artifact_id if previous else None,
            previous.current_manifest_checksum if previous else None,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.active_path.write_text(
            json.dumps(pointer.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return pointer

    def read_active(self) -> ActiveModelPointer:
        from clauseforge.artifacts.validation import sha256_file

        raw = json.loads(self.active_path.read_text(encoding="utf-8"))
        if (
            not isinstance(raw, dict)
            or raw.get("schema_version") != "clauseforge-active-model-v1"
        ):
            raise ValueError("invalid active-model pointer")
        manifest_ref = str(raw["current_manifest"])
        candidate = Path(manifest_ref)
        target = (self.root / candidate).resolve()
        if candidate.is_absolute() or self.root not in target.parents:
            raise ValueError("unsafe active-model pointer")
        checksum = str(raw["current_manifest_checksum"])
        if sha256_file(target) != checksum:
            raise ValueError("active-model manifest checksum mismatch")
        manifest = load_manifest(target)
        if manifest.artifact_id != str(raw["current_artifact_id"]):
            raise ValueError("active-model identity mismatch")
        return ActiveModelPointer(
            str(raw["schema_version"]),
            manifest_ref,
            manifest.artifact_id,
            checksum,
            _optional_string(raw.get("previous_manifest")),
            _optional_string(raw.get("previous_artifact_id")),
            _optional_string(raw.get("previous_manifest_checksum")),
        )

    def rollback(self) -> ActiveModelPointer:
        active = self.read_active()
        if active.previous_artifact_id is None:
            raise ValueError("no previous model is available")
        return self.activate(active.previous_artifact_id)


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)
