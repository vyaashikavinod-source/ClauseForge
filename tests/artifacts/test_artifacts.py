from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from clauseforge.artifacts.models import ArtifactManifest
from clauseforge.artifacts.registry import ArtifactRegistry, promotion_gates
from clauseforge.artifacts.release import (
    FinalEvaluationBundle,
    authorize_test_once,
    lock_final_model,
    validate_final_lock,
)
from clauseforge.artifacts.validation import (
    load_manifest,
    sha256_file,
    validate_manifest,
    write_manifest,
)
from clauseforge.artifacts.workflows import (
    build_deployment_bundle,
    final_validation_plan,
    validate_import,
    validate_parent_lineage,
)

RC0 = Path("configs/artifacts/clauseforge-qwen25-7b-r8-rc0.json")


def test_rc0_schema_serialization_and_truthful_blocked_status(tmp_path: Path) -> None:
    manifest = load_manifest(RC0)
    copy = tmp_path / "rc0.json"
    write_manifest(copy, manifest)
    assert load_manifest(copy) == manifest
    report = validate_manifest(RC0)
    assert report.valid
    assert manifest.release_status == "release_candidate"
    assert not manifest.final_release and not manifest.test_evaluated
    assert manifest.validation_summary is not None
    assert "NOT FINAL MODEL PERFORMANCE" in manifest.validation_summary.label
    gates = promotion_gates(manifest, "final_candidate")
    assert not gates.allowed
    assert "full training completed" in gates.blocking_gates


def test_checksum_and_safe_required_paths(tmp_path: Path) -> None:
    manifest = load_manifest(RC0)
    payload = tmp_path / "adapter.bin"
    payload.write_bytes(b"tiny synthetic fixture")
    local = replace(
        manifest,
        adapter_path="adapter.bin",
        adapter_checksum=sha256_file(payload),
        required_files={"adapter.bin": sha256_file(payload)},
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, local)
    assert validate_manifest(path).valid
    payload.write_bytes(b"corrupt")
    assert "checksum mismatch: adapter.bin" in validate_manifest(path).errors
    unsafe = replace(
        local,
        adapter_path=None,
        adapter_checksum=None,
        required_files={"../outside": "0" * 64},
    )
    write_manifest(path, unsafe)
    assert any("unsafe" in error for error in validate_manifest(path).errors)


def test_import_validates_adapter_without_copying(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter.bin"
    adapter.write_bytes(b"metadata-only test adapter")
    manifest = replace(load_manifest(RC0), adapter_checksum=sha256_file(adapter))
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    report = validate_import(adapter, path)
    assert report.valid and not report.copied
    adapter.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        validate_import(adapter, path)


def test_registry_activation_switch_and_rollback(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path / "registry")
    first = replace(load_manifest(RC0), artifact_id="first")
    second = replace(first, artifact_id="second")
    one = tmp_path / "one.json"
    two = tmp_path / "two.json"
    write_manifest(one, first)
    write_manifest(two, second)
    registry.register(one)
    registry.register(two)
    registry.activate("first")
    active = registry.activate("second")
    assert active.current_artifact_id == "second"
    assert active.previous_artifact_id == "first"
    assert registry.rollback().current_artifact_id == "first"
    manifest_path = registry.manifests / "first.json"
    manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        registry.read_active()


def test_final_lock_and_one_time_test_gate(tmp_path: Path) -> None:
    validation = tmp_path / "validation.json"
    validation.write_text("{}", encoding="utf-8")
    candidate = replace(load_manifest(RC0), full_training_completed=True)
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, candidate)
    lock_path = tmp_path / "lock.json"
    lock = lock_final_model(manifest, validation, lock_path, "2026-09-02T00:00:00Z")
    assert not lock.test_evaluated
    assert validate_final_lock(lock_path, manifest) == lock
    authorization = authorize_test_once(lock_path)
    assert authorization.test_authorized and not authorization.test_evaluated
    with pytest.raises(ValueError, match="already"):
        authorize_test_once(lock_path)
    write_manifest(manifest, replace(candidate, config_checksum="1" * 64))
    with pytest.raises(ValueError, match="changed"):
        validate_final_lock(lock_path, manifest)


def test_lineage_rejects_wrong_parent_and_preserves_quantization_chain(
    tmp_path: Path,
) -> None:
    parent = load_manifest(RC0)
    parent_path = tmp_path / "parent.json"
    write_manifest(parent_path, parent)
    child = replace(
        parent,
        artifact_id="merged",
        artifact_type="merged_model",
        parent_artifact_id=parent.artifact_id,
        parent_artifact_checksum=sha256_file(parent_path),
    )
    validate_parent_lineage(child, parent_path)
    with pytest.raises(ValueError, match="identity"):
        validate_parent_lineage(replace(child, parent_artifact_id="wrong"), parent_path)


def test_deployment_bundle_validation(tmp_path: Path) -> None:
    manifest = tmp_path / "model.json"
    write_manifest(manifest, load_manifest(RC0))
    deployment: dict[str, object] = {
        "artifact_id": "clauseforge-qwen25-7b-r8-rc0",
        "backend": "transformer",
        "quantization": "none",
        "model_manifest_checksum": sha256_file(manifest),
    }
    output = build_deployment_bundle(
        tmp_path / "bundle", manifest, deployment, "External model files required.\n"
    )
    checksums = json.loads((output / "checksums.json").read_text(encoding="utf-8"))
    assert set(checksums) == {"model_manifest.json", "deployment_manifest.json"}


def test_no_fake_performance_in_empty_evaluation_bundle() -> None:
    manifest: ArtifactManifest = load_manifest(RC0)
    assert not manifest.test_evaluated
    assert not manifest.final_safety_evaluated
    assert not manifest.final_ood_evaluated
    assert not manifest.quantized and not manifest.deployed
    bundle = FinalEvaluationBundle(
        "clauseforge-final-evaluation-v1",
        manifest.artifact_id,
        manifest.training_commit,
        "2026-09-02T00:00:00Z",
        {"source": "validation"},
        None,
        None,
        None,
        {"execution": "not_run"},
        ("GPU work pending",),
    )
    assert not bundle.complete


def test_final_validation_plan_is_non_executing(tmp_path: Path) -> None:
    candidate = replace(
        load_manifest(RC0),
        release_status="final_candidate",
        full_training_completed=True,
        validation_selection_completed=True,
    )
    path = tmp_path / "candidate.json"
    write_manifest(path, candidate)
    plan = final_validation_plan(path)
    assert plan["executed"] is False
    assert plan["steps"] == [
        "validate_artifact",
        "held_out_test_blocked",
        "final_safety_harness",
        "final_edgar_ood",
        "build_final_evaluation_bundle",
    ]
