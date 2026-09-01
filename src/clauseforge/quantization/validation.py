"""Offline release artifact schema, checksum, taxonomy, and backend validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from clauseforge.quantization.manifest import DeploymentManifest, deployment_from_dict
from clauseforge.serving.constants import TAXONOMY_VERSION
from clauseforge.training.templates import PROMPT_TEMPLATE_VERSION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_deployment(path: Path) -> DeploymentManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("deployment manifest must be an object")
    manifest = deployment_from_dict(raw)
    if (
        manifest.taxonomy_version != TAXONOMY_VERSION
        or manifest.prompt_version != PROMPT_TEMPLATE_VERSION
    ):
        raise ValueError("deployment taxonomy or prompt version is incompatible")
    if manifest.backend not in {"transformer", "vllm", "llamacpp"}:
        raise ValueError("deployment backend is unsupported")
    root = path.parent.resolve()
    for relative, checksum in manifest.required_files.items():
        target = (root / relative).resolve()
        if root not in target.parents or not target.is_file():
            raise ValueError(
                f"required deployment file is missing or unsafe: {relative}"
            )
        if sha256_file(target) != checksum:
            raise ValueError(f"checksum mismatch: {relative}")
    return manifest
