"""Generic trained-checkpoint manifest creation and evidence attachment."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from clauseforge.artifacts.models import ArtifactManifest, ValidationSummary
from clauseforge.artifacts.validation import sha256_file, sha256_path, write_manifest
from clauseforge.serving.constants import TAXONOMY_VERSION
from clauseforge.training.targets import (
    TargetRepresentation,
    stable_id_map_checksum,
    validate_target_version,
)


def create_candidate_manifest(
    checkpoint: Path,
    artifact_id: str,
    output: Path,
    training_commit: str,
) -> ArtifactManifest:
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    if not checkpoint.is_dir():
        raise ValueError("checkpoint must be an existing directory")
    if not artifact_id.strip() or not training_commit.strip():
        raise ValueError("artifact ID and historical training commit are required")
    try:
        adapter_path = checkpoint.relative_to(output.parent).as_posix()
    except ValueError as exc:
        raise ValueError(
            "checkpoint must be contained by the manifest directory"
        ) from exc

    metadata = _object(checkpoint / "checkpoint_metadata.json")
    resume = _object(checkpoint / "resume_state.json")
    adapter = _object(checkpoint / "adapter_config.json")
    experiment_path = checkpoint.parent / "experiment_config.json"
    experiment = _object(experiment_path)
    model = _mapping(experiment, "model")
    lora = _mapping(experiment, "lora")

    step = _same_int(
        "checkpoint step", metadata.get("global_step"), resume.get("global_step")
    )
    experiment_id = _same_string(
        "experiment ID", metadata.get("experiment_id"), resume.get("experiment_id")
    )
    _expect(
        checkpoint.name == f"checkpoint-{step}", "checkpoint directory/step mismatch"
    )
    rank = _same_int(
        "LoRA rank", metadata.get("lora_rank"), lora.get("rank"), adapter.get("r")
    )
    alpha = _same_int(
        "LoRA alpha",
        metadata.get("lora_alpha"),
        lora.get("alpha"),
        adapter.get("lora_alpha"),
    )
    targets = _same_set(
        "target modules",
        metadata.get("target_modules"),
        lora.get("target_modules"),
        adapter.get("target_modules"),
    )
    representation = _same_string(
        "target representation",
        metadata.get("target_representation"),
        experiment.get("target_representation"),
    )
    representation_version = _same_string(
        "target representation version",
        metadata.get("target_representation_version"),
        experiment.get("target_representation_version"),
    )
    prompt_version = _same_string(
        "prompt version",
        metadata.get("prompt_template_version"),
        experiment.get("prompt_template_version"),
    )
    validate_target_version(
        cast(TargetRepresentation, representation),
        representation_version,
        prompt_version,
    )
    stable_checksum = _same_string(
        "stable-ID checksum",
        metadata.get("stable_id_map_checksum"),
        stable_id_map_checksum(),
    )
    taxonomy = str(metadata.get("taxonomy_version", TAXONOMY_VERSION))
    _expect(taxonomy == TAXONOMY_VERSION, "taxonomy version mismatch")

    required_names = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "checkpoint_metadata.json",
    ]
    if (checkpoint / "resume_state.json").is_file():
        required_names.append("resume_state.json")
    for name in required_names:
        _expect(
            (checkpoint / name).is_file(), f"required checkpoint file missing: {name}"
        )
    required = {
        f"{adapter_path}/{name}": sha256_file(checkpoint / name)
        for name in required_names
    }
    base_model = str(model["name"])
    adapter_base = adapter.get("base_model_name_or_path")
    if adapter_base is not None:
        _expect(str(adapter_base) == base_model, "adapter base model mismatch")

    manifest = ArtifactManifest(
        schema_version="clauseforge-artifact-v1",
        artifact_id=artifact_id,
        artifact_type="adapter",
        base_model=base_model,
        base_revision=str(model["revision"]),
        adapter_experiment_id=experiment_id,
        adapter_path=adapter_path,
        adapter_checksum=sha256_path(checkpoint),
        parent_artifact_id=None,
        parent_artifact_checksum=None,
        target_representation=representation,
        target_representation_version=representation_version,
        prompt_version=prompt_version,
        taxonomy_version=taxonomy,
        stable_id_map_checksum=stable_checksum,
        lora_rank=rank,
        lora_alpha=alpha,
        lora_dropout=float(cast(float | str, lora["dropout"])),
        lora_targets=tuple(sorted(targets)),
        quantization_mode=str(model["quantization"]),
        quantization_type=str(model["quant_type"]),
        double_quantization=bool(model["double_quant"]),
        precision=str(model["precision"]),
        checkpoint_step=step,
        training_commit=training_commit,
        config_checksum=sha256_file(experiment_path),
        validation_summary=None,
        full_training_completed=False,
        validation_selection_completed=False,
        test_evaluated=False,
        final_safety_evaluated=False,
        final_ood_evaluated=False,
        quantized=False,
        deployed=False,
        real_serving_benchmark_completed=False,
        container_smoke_test_completed=False,
        deployment_manifest_valid=False,
        release_checklist_complete=False,
        release_status="release_candidate",
        final_release=False,
        created_at=datetime.now(UTC).isoformat(),
        known_limitations=(
            "REAL TRAINED MODEL CANDIDATE — NOT FINAL RELEASE",
            "Validation evidence must be attached before comparison or activation",
            "Held-out test, final safety/OOD, merge, quantization, and release "
            "are not run",
        ),
        required_files=required,
    )
    write_manifest(output, manifest)
    return manifest


def attach_validation_evidence(
    manifest_path: Path, evidence_path: Path, output: Path
) -> ArtifactManifest:
    from clauseforge.artifacts.validation import load_manifest, validate_manifest

    if output.resolve().parent != manifest_path.resolve().parent:
        raise ValueError("updated manifest must remain beside its checkpoint")
    report = validate_manifest(manifest_path)
    if not report.valid:
        raise ValueError("candidate manifest is invalid: " + "; ".join(report.errors))
    manifest = load_manifest(manifest_path)
    evidence = _object(evidence_path)
    _expect(evidence.get("split") == "validation", "evidence must use validation split")
    _expect(
        evidence.get("test_evaluated") is False, "held-out test evidence is forbidden"
    )
    evidence_step = int(
        cast(
            int | str, evidence.get("global_step", evidence.get("checkpoint_step", -1))
        )
    )
    _expect(
        evidence_step == manifest.checkpoint_step, "validation checkpoint step mismatch"
    )
    summary = ValidationSummary(
        label=str(evidence["label"]),
        split="validation",
        train_examples=int(cast(int | str, evidence["train_examples"])),
        validation_examples=int(cast(int | str, evidence["validation_examples"])),
        optimizer_steps=int(cast(int | str, evidence["optimizer_steps"])),
        examples_seen=int(cast(int | str, evidence["examples_seen"])),
        accuracy=float(cast(float | str, evidence["accuracy"])),
        macro_f1=float(cast(float | str, evidence["macro_f1"])),
        weighted_f1=float(cast(float | str, evidence["weighted_f1"])),
        exact_id_rate=float(
            cast(
                float | str,
                evidence.get("exact_id_rate")
                if evidence.get("exact_id_rate") is not None
                else evidence["exact_id_output_rate"],
            )
        ),
        invalid_output_rate=float(cast(float | str, evidence["invalid_output_rate"])),
        validation_loss=float(cast(float | str, evidence["validation_loss"])),
    )
    updated = replace(
        manifest,
        validation_summary=summary,
        validation_selection_completed=True,
        known_limitations=tuple(
            item
            for item in manifest.known_limitations
            if "must be attached" not in item
        ),
    )
    write_manifest(output, updated)
    return updated


def _object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid or missing metadata file: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"metadata file must contain an object: {path.name}")
    return cast(dict[str, object], value)


def _mapping(value: dict[str, object], name: str) -> dict[str, object]:
    result = value.get(name)
    if not isinstance(result, dict):
        raise ValueError(f"experiment config is missing {name}")
    return cast(dict[str, object], result)


def _same_int(name: str, *values: object) -> int:
    try:
        converted = [int(cast(int | str, item)) for item in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is missing or invalid") from exc
    _expect(len(set(converted)) == 1, f"{name} mismatch")
    return converted[0]


def _same_string(name: str, *values: object) -> str:
    converted = [str(item) for item in values]
    _expect(None not in values and len(set(converted)) == 1, f"{name} mismatch")
    return converted[0]


def _same_set(name: str, *values: object) -> set[str]:
    try:
        converted = [
            {str(item) for item in cast(list[object], value)} for value in values
        ]
    except TypeError as exc:
        raise ValueError(f"{name} is missing or invalid") from exc
    _expect(all(value == converted[0] for value in converted[1:]), f"{name} mismatch")
    return converted[0]


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
