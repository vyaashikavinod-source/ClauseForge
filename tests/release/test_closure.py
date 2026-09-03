from __future__ import annotations

import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from scripts.authorize_held_out_test import main as authorize_test_main
from scripts.release_status import main as release_status_main

from clauseforge.artifacts.archive import package_gpu_artifacts, restore_gpu_artifacts
from clauseforge.artifacts.release import (
    authorize_deployment,
    authorize_test_once,
    lock_final_model,
    record_test_evaluated,
)
from clauseforge.artifacts.validation import load_manifest, write_manifest
from clauseforge.release.manifest import FinalReleaseManifest
from clauseforge.release.selection import select_final_candidate
from clauseforge.release.status import (
    build_release_status,
    human_status,
    serialize_status,
)

RC0 = Path("configs/artifacts/clauseforge-qwen25-7b-r8-rc0.json")


def test_rc0_release_status_and_exact_actions() -> None:
    report = build_release_status(RC0)
    assert report.active_candidate == "clauseforge-qwen25-7b-r8-rc0"
    assert report.candidate_lifecycle_state == "release_candidate"
    assert report.artifact_manifest_valid
    assert not report.release_allowed
    gates = {gate.name: gate for gate in report.gates}
    assert gates["held_out_test"].state == "BLOCKED"
    assert gates["held_out_test"].reason == "Final candidate is not locked"
    assert gates["held_out_test"].next_action == "Lock the selected candidate first"
    assert "release_allowed: false" in human_status(report)


def test_release_status_json_is_machine_readable() -> None:
    payload = json.loads(serialize_status(build_release_status(RC0)))
    assert payload["artifact_manifest_valid"] is True
    assert payload["release_allowed"] is False
    assert all(item["next_action"] for item in payload["gates"])


def test_release_status_command_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert release_status_main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out)["release_allowed"] is False


def test_test_authorization_requires_explicit_confirmation() -> None:
    with pytest.raises(SystemExit):
        authorize_test_main(
            ["--lock", "lock.json", "--artifact-manifest", "model.json"]
        )


def _experiment(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "experiment"
    root.mkdir()
    manifest = load_manifest(RC0)
    write_manifest(root / "artifact_manifest.json", manifest)
    (root / "validation_metrics.json").write_text(
        '{"split":"validation"}', encoding="utf-8"
    )
    adapter = root / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (root / "cache.bin").write_bytes(b"excluded")
    return root, manifest.adapter_experiment_id


def test_gpu_backup_manifest_and_verified_restore(tmp_path: Path) -> None:
    experiment, experiment_id = _experiment(tmp_path)
    archive = tmp_path / "backup.tar.gz"
    package = package_gpu_artifacts(experiment, archive)
    assert "cache.bin" not in cast(dict[str, object], package["files"])
    assert package["target_representation_version"] == "cuad-category-id-v1"
    restored = tmp_path / "restored"
    result = restore_gpu_artifacts(
        archive, restored, expected_experiment_id=experiment_id
    )
    assert result["experiment_id"] == experiment_id
    assert not (restored / "cache.bin").exists()


def test_archive_path_traversal_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        payload = b"bad"
        info = tarfile.TarInfo("../escape")
        info.size = len(payload)
        stream.addfile(info, io.BytesIO(payload))
    with pytest.raises(ValueError, match="unsafe archive"):
        restore_gpu_artifacts(archive, tmp_path / "out", expected_experiment_id="x")


def test_restore_rejects_wrong_experiment(tmp_path: Path) -> None:
    experiment, _ = _experiment(tmp_path)
    archive = tmp_path / "backup.tar.gz"
    package_gpu_artifacts(experiment, archive)
    with pytest.raises(ValueError, match="wrong experiment"):
        restore_gpu_artifacts(archive, tmp_path / "out", expected_experiment_id="wrong")


def _metrics(
    root: Path, step: int, macro: float, invalid: float, exact: float, loss: float
) -> None:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "validation_metrics.json").write_text(
        json.dumps(
            {
                "split": "validation",
                "global_step": step,
                "validation_macro_f1": macro,
                "invalid_output_rate": invalid,
                "exact_id_rate": exact,
                "validation_loss": loss,
            }
        ),
        encoding="utf-8",
    )


def test_final_selection_uses_documented_validation_ties(tmp_path: Path) -> None:
    _metrics(tmp_path, 20, 0.5, 0.2, 0.8, 0.4)
    _metrics(tmp_path, 40, 0.5, 0.1, 0.8, 0.5)
    _metrics(tmp_path, 60, 0.5, 0.1, 0.9, 0.6)
    _metrics(tmp_path, 80, 0.5, 0.1, 0.9, 0.3)
    _metrics(tmp_path, 100, 0.5, 0.1, 0.9, 0.3)
    assert select_final_candidate(tmp_path).global_step == 80


def test_explicit_test_authorization_and_consumption(tmp_path: Path) -> None:
    validation = tmp_path / "selection.json"
    validation.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, replace(load_manifest(RC0), full_training_completed=True))
    lock_path = tmp_path / "lock.json"
    lock_final_model(manifest, validation, lock_path, "2026-09-03T00:00:00Z")
    authorized = authorize_test_once(lock_path)
    assert authorized.test_authorized and not authorized.test_evaluated
    with pytest.raises(ValueError, match="already"):
        authorize_test_once(lock_path)
    consumed = record_test_evaluated(lock_path)
    assert consumed.test_evaluated and not consumed.test_authorized


def test_final_release_template_is_incomplete_but_schema_valid() -> None:
    raw = json.loads(
        Path("configs/artifacts/final_release_manifest.template.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = FinalReleaseManifest(**raw)
    manifest.validate()
    with pytest.raises(ValueError, match="evidence"):
        manifest.validate(final=True)


def test_deployment_authorization_fails_closed() -> None:
    with pytest.raises(ValueError, match="blocked"):
        authorize_deployment(RC0)
