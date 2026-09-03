"""Offline schema, checksum, path, compatibility, and lineage validation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from clauseforge.artifacts.models import ArtifactManifest
from clauseforge.serving.constants import TAXONOMY_VERSION
from clauseforge.training.targets import stable_id_map_checksum, validate_target_version

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ArtifactValidationReport:
    valid: bool
    errors: tuple[str, ...]
    artifact_id: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash one file or a directory tree including normalized relative names."""
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise ValueError("artifact path is not a file or directory")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError("artifact directory is empty")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def load_manifest(path: Path) -> ArtifactManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("artifact manifest must be an object")
    return ArtifactManifest.from_dict(raw)


def write_manifest(path: Path, manifest: ArtifactManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_manifest(
    path: Path, *, require_files: bool = True
) -> ArtifactValidationReport:
    try:
        manifest = load_manifest(path)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return ArtifactValidationReport(False, (str(exc),), None)
    errors: list[str] = []
    if manifest.schema_version != "clauseforge-artifact-v1":
        errors.append("unsupported schema_version")
    if not manifest.artifact_id or not manifest.adapter_experiment_id:
        errors.append("artifact and experiment identity are required")
    if manifest.taxonomy_version != TAXONOMY_VERSION:
        errors.append("taxonomy_version is incompatible")
    if manifest.stable_id_map_checksum != stable_id_map_checksum():
        errors.append("stable_id_map_checksum is incompatible")
    try:
        validate_target_version(
            manifest.target_representation,  # type: ignore[arg-type]
            manifest.target_representation_version,
            manifest.prompt_version,
        )
    except (ValueError, KeyError):
        errors.append("target/prompt versions are incompatible")
    if manifest.lora_rank <= 0 or manifest.lora_alpha <= 0 or not manifest.lora_targets:
        errors.append("LoRA metadata is incomplete")
    for checksum_name, checksum in (
        ("adapter_checksum", manifest.adapter_checksum),
        ("config_checksum", manifest.config_checksum),
        ("parent_artifact_checksum", manifest.parent_artifact_checksum),
    ):
        if checksum is not None and not _SHA256.fullmatch(checksum):
            errors.append(f"{checksum_name} is not SHA-256")
    if manifest.artifact_type != "adapter" and (
        manifest.parent_artifact_id is None or manifest.parent_artifact_checksum is None
    ):
        errors.append("derived artifact requires parent lineage")
    if manifest.release_status == "released" and not manifest.final_release:
        errors.append("released artifact must set final_release=true")
    if manifest.test_evaluated and manifest.release_status in {"development", "pilot"}:
        errors.append("test evaluation requires a locked release candidate")
    if require_files:
        _validate_files(path, manifest, errors)
    return ArtifactValidationReport(not errors, tuple(errors), manifest.artifact_id)


def _validate_files(path: Path, manifest: ArtifactManifest, errors: list[str]) -> None:
    root = path.parent.resolve()
    for relative, expected in manifest.required_files.items():
        candidate = Path(relative)
        target = (root / candidate).resolve()
        if candidate.is_absolute() or (target != root and root not in target.parents):
            errors.append(f"unsafe required path: {relative}")
        elif not target.is_file():
            errors.append(f"required file missing: {relative}")
        elif sha256_file(target) != expected:
            errors.append(f"checksum mismatch: {relative}")
    if manifest.adapter_path:
        candidate = Path(manifest.adapter_path)
        target = (root / candidate).resolve()
        if candidate.is_absolute() or (target != root and root not in target.parents):
            errors.append("adapter_path must be relative to the manifest")
        elif not target.exists():
            errors.append("adapter_path does not exist")
        elif (
            manifest.adapter_checksum
            and sha256_path(target) != manifest.adapter_checksum
        ):
            errors.append("adapter checksum mismatch")
