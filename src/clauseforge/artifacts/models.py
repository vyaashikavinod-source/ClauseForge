"""Strict schemas for portable ClauseForge model artifact metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, cast

ArtifactType = Literal["adapter", "merged_model", "awq", "gguf", "deployment_bundle"]
ReleaseStatus = Literal[
    "development", "pilot", "release_candidate", "final_candidate", "released"
]
_ARTIFACT_TYPES = {"adapter", "merged_model", "awq", "gguf", "deployment_bundle"}
_RELEASE_STATES = {
    "development",
    "pilot",
    "release_candidate",
    "final_candidate",
    "released",
}


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    label: str
    split: str
    train_examples: int
    validation_examples: int
    optimizer_steps: int
    examples_seen: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    exact_id_rate: float
    invalid_output_rate: float
    validation_loss: float | None

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> ValidationSummary:
        return cls(
            label=str(value["label"]),
            split=str(value["split"]),
            train_examples=int(cast(int | str, value["train_examples"])),
            validation_examples=int(cast(int | str, value["validation_examples"])),
            optimizer_steps=int(cast(int | str, value["optimizer_steps"])),
            examples_seen=int(cast(int | str, value["examples_seen"])),
            accuracy=float(cast(float | str, value["accuracy"])),
            macro_f1=float(cast(float | str, value["macro_f1"])),
            weighted_f1=float(cast(float | str, value["weighted_f1"])),
            exact_id_rate=float(cast(float | str, value["exact_id_rate"])),
            invalid_output_rate=float(cast(float | str, value["invalid_output_rate"])),
            validation_loss=(
                None
                if value.get("validation_loss") is None
                else float(cast(float | str, value["validation_loss"]))
            ),
        )


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    schema_version: str
    artifact_id: str
    artifact_type: ArtifactType
    base_model: str
    base_revision: str
    adapter_experiment_id: str
    adapter_path: str | None
    adapter_checksum: str | None
    parent_artifact_id: str | None
    parent_artifact_checksum: str | None
    target_representation: str
    target_representation_version: str
    prompt_version: str
    taxonomy_version: str
    stable_id_map_checksum: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: tuple[str, ...]
    quantization_mode: str
    quantization_type: str
    double_quantization: bool
    precision: str
    checkpoint_step: int
    training_commit: str
    config_checksum: str
    validation_summary: ValidationSummary | None
    full_training_completed: bool
    validation_selection_completed: bool
    test_evaluated: bool
    final_safety_evaluated: bool
    final_ood_evaluated: bool
    quantized: bool
    deployed: bool
    real_serving_benchmark_completed: bool
    container_smoke_test_completed: bool
    deployment_manifest_valid: bool
    release_checklist_complete: bool
    release_status: ReleaseStatus
    final_release: bool
    created_at: str
    known_limitations: tuple[str, ...]
    required_files: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> ArtifactManifest:
        required = {field.name for field in cls.__dataclass_fields__.values()}
        missing = required - value.keys()
        extra = value.keys() - required
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing: {', '.join(sorted(missing))}")
            if extra:
                details.append(f"unknown: {', '.join(sorted(extra))}")
            raise ValueError(
                "artifact manifest schema error (" + "; ".join(details) + ")"
            )
        artifact_type = str(value["artifact_type"])
        release_status = str(value["release_status"])
        if (
            artifact_type not in _ARTIFACT_TYPES
            or release_status not in _RELEASE_STATES
        ):
            raise ValueError("unknown artifact type or release status")
        summary_raw = value["validation_summary"]
        if summary_raw is not None and not isinstance(summary_raw, dict):
            raise ValueError("validation_summary must be an object or null")
        targets = cast(list[object], value["lora_targets"])
        limitations = cast(list[object], value["known_limitations"])
        files = cast(dict[object, object], value["required_files"])
        return cls(
            schema_version=str(value["schema_version"]),
            artifact_id=str(value["artifact_id"]),
            artifact_type=cast(ArtifactType, artifact_type),
            base_model=str(value["base_model"]),
            base_revision=str(value["base_revision"]),
            adapter_experiment_id=str(value["adapter_experiment_id"]),
            adapter_path=None
            if value["adapter_path"] is None
            else str(value["adapter_path"]),
            adapter_checksum=None
            if value["adapter_checksum"] is None
            else str(value["adapter_checksum"]),
            parent_artifact_id=None
            if value["parent_artifact_id"] is None
            else str(value["parent_artifact_id"]),
            parent_artifact_checksum=None
            if value["parent_artifact_checksum"] is None
            else str(value["parent_artifact_checksum"]),
            target_representation=str(value["target_representation"]),
            target_representation_version=str(value["target_representation_version"]),
            prompt_version=str(value["prompt_version"]),
            taxonomy_version=str(value["taxonomy_version"]),
            stable_id_map_checksum=str(value["stable_id_map_checksum"]),
            lora_rank=int(cast(int | str, value["lora_rank"])),
            lora_alpha=int(cast(int | str, value["lora_alpha"])),
            lora_dropout=float(cast(float | str, value["lora_dropout"])),
            lora_targets=tuple(str(item) for item in targets),
            quantization_mode=str(value["quantization_mode"]),
            quantization_type=str(value["quantization_type"]),
            double_quantization=bool(value["double_quantization"]),
            precision=str(value["precision"]),
            checkpoint_step=int(cast(int | str, value["checkpoint_step"])),
            training_commit=str(value["training_commit"]),
            config_checksum=str(value["config_checksum"]),
            validation_summary=None
            if summary_raw is None
            else ValidationSummary.from_dict(summary_raw),
            full_training_completed=bool(value["full_training_completed"]),
            validation_selection_completed=bool(
                value["validation_selection_completed"]
            ),
            test_evaluated=bool(value["test_evaluated"]),
            final_safety_evaluated=bool(value["final_safety_evaluated"]),
            final_ood_evaluated=bool(value["final_ood_evaluated"]),
            quantized=bool(value["quantized"]),
            deployed=bool(value["deployed"]),
            real_serving_benchmark_completed=bool(
                value["real_serving_benchmark_completed"]
            ),
            container_smoke_test_completed=bool(
                value["container_smoke_test_completed"]
            ),
            deployment_manifest_valid=bool(value["deployment_manifest_valid"]),
            release_checklist_complete=bool(value["release_checklist_complete"]),
            release_status=cast(ReleaseStatus, release_status),
            final_release=bool(value["final_release"]),
            created_at=str(value["created_at"]),
            known_limitations=tuple(str(item) for item in limitations),
            required_files={str(k): str(v) for k, v in files.items()},
        )
