"""Allowlisted, checksummed GPU experiment backup and safe restoration."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath

from clauseforge.artifacts.validation import load_manifest, validate_manifest

_ROOT_FILES = {
    "artifact_manifest.json",
    "experiment_config.json",
    "pilot_config.json",
    "resume_state.json",
    "training_metrics.json",
    "validation_metrics.json",
    "environment.json",
    "selected_train_examples.json",
    "selected_validation_examples.json",
}
_ROOT_DIRS = {"adapter"}


def package_gpu_artifacts(experiment: Path, output: Path) -> dict[str, object]:
    if not experiment.is_dir():
        raise ValueError("experiment directory does not exist")
    artifact_path = experiment / "artifact_manifest.json"
    report = validate_manifest(artifact_path, require_files=False)
    if not report.valid:
        raise ValueError("experiment artifact manifest is invalid")
    artifact = load_manifest(artifact_path)
    files = _selected_files(experiment)
    if not files:
        raise ValueError("experiment contains no allowlisted artifact files")
    checksums = {
        path.relative_to(experiment).as_posix(): _sha256(path) for path in files
    }
    package = {
        "schema_version": "clauseforge-gpu-backup-v1",
        "experiment_id": artifact.adapter_experiment_id,
        "artifact_id": artifact.artifact_id,
        "target_representation_version": artifact.target_representation_version,
        "prompt_version": artifact.prompt_version,
        "stable_id_map_checksum": artifact.stable_id_map_checksum,
        "lora_rank": artifact.lora_rank,
        "lora_targets": list(artifact.lora_targets),
        "files": checksums,
    }
    payload = (json.dumps(package, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        info = tarfile.TarInfo("backup_manifest.json")
        info.size = len(payload)
        info.mode = 0o600
        archive.addfile(info, io.BytesIO(payload))
        for path in files:
            archive.add(
                path, arcname=path.relative_to(experiment).as_posix(), recursive=False
            )
    return package


def restore_gpu_artifacts(
    archive_path: Path, output: Path, *, expected_experiment_id: str
) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        raise ValueError("restore output must be absent or empty")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            _validate_member(member)
        manifest_member = archive.getmember("backup_manifest.json")
        stream = archive.extractfile(manifest_member)
        if stream is None:
            raise ValueError("backup manifest is unreadable")
        raw = json.loads(stream.read())
        if (
            not isinstance(raw, dict)
            or raw.get("schema_version") != "clauseforge-gpu-backup-v1"
        ):
            raise ValueError("invalid backup manifest")
        if raw.get("experiment_id") != expected_experiment_id:
            raise ValueError("wrong experiment ID")
        declared = raw.get("files")
        if not isinstance(declared, dict):
            raise ValueError("backup checksums are missing")
        actual_names = {member.name for member in members if member.isfile()} - {
            "backup_manifest.json"
        }
        if actual_names != set(declared):
            raise ValueError("archive contents do not match backup manifest")
        for name, checksum in declared.items():
            member = archive.getmember(str(name))
            content = archive.extractfile(member)
            if (
                content is None
                or hashlib.sha256(content.read()).hexdigest() != checksum
            ):
                raise ValueError(f"backup checksum mismatch: {name}")
        output.mkdir(parents=True, exist_ok=True)
        archive.extractall(output, members=members, filter="data")
    artifact_path = output / "artifact_manifest.json"
    report = validate_manifest(artifact_path, require_files=False)
    if not report.valid:
        raise ValueError("restored artifact lineage is invalid")
    artifact = load_manifest(artifact_path)
    lineage = {
        "experiment_id": artifact.adapter_experiment_id,
        "target_representation_version": artifact.target_representation_version,
        "prompt_version": artifact.prompt_version,
        "stable_id_map_checksum": artifact.stable_id_map_checksum,
        "lora_rank": artifact.lora_rank,
        "lora_targets": list(artifact.lora_targets),
    }
    for key, value in lineage.items():
        if raw.get(key) != value:
            raise ValueError(f"restored lineage mismatch: {key}")
    return raw


def _selected_files(root: Path) -> list[Path]:
    selected = [root / name for name in sorted(_ROOT_FILES) if (root / name).is_file()]
    for child in sorted(root.iterdir()):
        allowed_dir = child.name in _ROOT_DIRS or (
            child.is_dir() and child.name.startswith("checkpoint-")
        )
        if allowed_dir:
            selected.extend(sorted(path for path in child.rglob("*") if path.is_file()))
    return selected


def _validate_member(member: tarfile.TarInfo) -> None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
        raise ValueError(f"unsafe archive member: {member.name}")
    if not member.isfile() and not member.isdir():
        raise ValueError(f"unsupported archive member: {member.name}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
