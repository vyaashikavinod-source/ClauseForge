"""Artifact and deployment lineage schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import cast


@dataclass(frozen=True, slots=True)
class QuantizationManifest:
    artifact_id: str
    base_model: str
    base_revision: str
    adapter_experiment_id: str
    adapter_revision: str
    adapter_checksum: str
    taxonomy_version: str
    prompt_version: str
    merge_commit: str
    quantization_method: str
    quantization_parameters: dict[str, object]
    output_format: str
    context_length: int
    created_at: str
    validation_status: str
    known_limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DeploymentManifest:
    deployment_id: str
    model_name: str
    base_revision: str
    adapter_experiment_id: str
    quantization_artifact_id: str
    backend: str
    taxonomy_version: str
    prompt_version: str
    context_length: int
    max_new_tokens: int
    build_commit: str
    environment: str
    created_at: str
    validation_status: str
    known_limitations: tuple[str, ...]
    required_files: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def deployment_from_dict(value: dict[str, object]) -> DeploymentManifest:
    required = {
        field.name for field in DeploymentManifest.__dataclass_fields__.values()
    }
    missing = required - value.keys()
    if missing:
        raise ValueError(f"deployment manifest missing: {', '.join(sorted(missing))}")
    limitations = cast(list[object], value["known_limitations"])
    files = cast(dict[object, object], value["required_files"])
    return DeploymentManifest(
        deployment_id=str(value["deployment_id"]),
        model_name=str(value["model_name"]),
        base_revision=str(value["base_revision"]),
        adapter_experiment_id=str(value["adapter_experiment_id"]),
        quantization_artifact_id=str(value["quantization_artifact_id"]),
        backend=str(value["backend"]),
        taxonomy_version=str(value["taxonomy_version"]),
        prompt_version=str(value["prompt_version"]),
        context_length=int(cast(int | str, value["context_length"])),
        max_new_tokens=int(cast(int | str, value["max_new_tokens"])),
        build_commit=str(value["build_commit"]),
        environment=str(value["environment"]),
        created_at=str(value["created_at"]),
        validation_status=str(value["validation_status"]),
        known_limitations=tuple(str(x) for x in limitations),
        required_files={str(k): str(v) for k, v in files.items()},
    )
