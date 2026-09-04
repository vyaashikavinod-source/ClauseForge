"""Deterministic single-surface release status and operator guidance."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from clauseforge.artifacts.models import ArtifactManifest
from clauseforge.artifacts.release import validate_final_lock
from clauseforge.artifacts.validation import load_manifest, validate_manifest


@dataclass(frozen=True, slots=True)
class GateStatus:
    name: str
    complete: bool
    state: str
    reason: str
    next_action: str


@dataclass(frozen=True, slots=True)
class ReleaseStatusReport:
    active_candidate: str
    candidate_lifecycle_state: str
    artifact_manifest_valid: bool
    gates: tuple[GateStatus, ...]
    release_allowed: bool
    label: str = "FINAL RELEASE STATUS — NO PERFORMANCE INFERENCE"
    incomplete_training_selection_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_release_status(
    manifest_path: Path, *, lock_path: Path | None = None
) -> ReleaseStatusReport:
    validation = validate_manifest(manifest_path, require_files=False)
    if not validation.valid:
        return ReleaseStatusReport(
            validation.artifact_id or "unknown",
            "unknown",
            False,
            (
                _gate(
                    "artifact_manifest",
                    False,
                    "Manifest validation failed",
                    f"Fix and validate {manifest_path}",
                ),
            ),
            False,
        )
    manifest = load_manifest(manifest_path)
    locked = False
    selection_reason = None
    if lock_path is not None and lock_path.is_file():
        try:
            lock = validate_final_lock(lock_path, manifest_path)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            locked = False
        else:
            locked = True
            selection_reason = lock.incomplete_training_selection_reason
    gates = _manifest_gates(manifest, locked)
    return ReleaseStatusReport(
        manifest.artifact_id,
        manifest.release_status,
        True,
        gates,
        all(gate.complete for gate in gates) and manifest.release_status == "released",
        incomplete_training_selection_reason=selection_reason,
    )


def _manifest_gates(manifest: ArtifactManifest, locked: bool) -> tuple[GateStatus, ...]:
    return (
        _gate(
            "final_model_locked",
            locked,
            "Selected candidate has not been locked",
            "Run scripts/lock_final_model.py after validation selection",
        ),
        _gate(
            "validation_complete",
            manifest.validation_selection_completed,
            "Full validation selection is incomplete",
            "Complete rank-8 validation and select the winning checkpoint",
        ),
        _gate(
            "held_out_test",
            manifest.test_evaluated,
            "Final candidate has not received its one authorized test run"
            if locked
            else "Final candidate is not locked",
            "Authorize the held-out test, then run final validation"
            if locked
            else "Lock the selected candidate first",
        ),
        _gate(
            "final_safety",
            manifest.final_safety_evaluated,
            "Final-model safety evaluation is incomplete",
            "Run the final safety harness on the locked artifact",
        ),
        _gate(
            "edgar_ood",
            manifest.final_ood_evaluated,
            "Final EDGAR OOD evaluation is incomplete",
            "Run EDGAR OOD on the locked artifact",
        ),
        _gate(
            "merge",
            manifest.artifact_type
            in {"merged_model", "awq", "gguf", "deployment_bundle"},
            "Adapter has not been merged",
            "Run scripts/merge_adapter.py for the selected adapter",
        ),
        _gate(
            "quantization",
            manifest.quantized,
            "No validated quantized artifact is recorded",
            "Run the selected AWQ/GGUF plan and validate its parent lineage",
        ),
        _gate(
            "benchmark",
            manifest.real_serving_benchmark_completed,
            "Real serving benchmark is incomplete",
            "Run scripts/benchmark_serving.py against the real artifact",
        ),
        _gate(
            "container_smoke",
            manifest.container_smoke_test_completed,
            "Real-model container smoke is incomplete",
            "Run the documented real-model container smoke test",
        ),
        _gate(
            "deployment_bundle",
            manifest.deployment_manifest_valid,
            "Deployment bundle has not been validated",
            "Build and validate the deployment bundle",
        ),
        _gate(
            "deployment_authorized",
            manifest.deployed,
            "Deployment has not been explicitly authorized",
            "Record deployment authorization only after every release gate passes",
        ),
        _gate(
            "release_checklist",
            manifest.release_checklist_complete,
            "Release checklist is incomplete",
            "Complete docs/release_checklist.md with evidence",
        ),
    )


def _gate(name: str, complete: bool, reason: str, next_action: str) -> GateStatus:
    return GateStatus(
        name,
        complete,
        "COMPLETE" if complete else "BLOCKED",
        "complete" if complete else reason,
        "none" if complete else next_action,
    )


def serialize_status(report: ReleaseStatusReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)


def human_status(report: ReleaseStatusReport) -> str:
    lines = [
        report.label,
        f"candidate: {report.active_candidate}",
        f"lifecycle: {report.candidate_lifecycle_state}",
        f"artifact_valid: {str(report.artifact_manifest_valid).lower()}",
        "incomplete_training_selection_reason: "
        f"{report.incomplete_training_selection_reason}",
    ]
    for gate in report.gates:
        lines.extend(
            (
                f"\n{gate.name}: {gate.state}",
                f"  reason: {gate.reason}",
                f"  next_action: {gate.next_action}",
            )
        )
    lines.append(f"\nrelease_allowed: {str(report.release_allowed).lower()}")
    return "\n".join(lines)
