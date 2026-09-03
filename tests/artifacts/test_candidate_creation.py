from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from clauseforge.artifacts.candidate import (
    attach_validation_evidence,
    create_candidate_manifest,
)
from clauseforge.artifacts.registry import ArtifactRegistry
from clauseforge.artifacts.validation import validate_manifest, write_manifest
from clauseforge.training.targets import stable_id_map_checksum


def _checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir()
    experiment_id = "qwen2.5-7b-instruct_lora-r8_seed42_0ad1c2c19e59"
    shared = {
        "experiment_id": experiment_id,
        "global_step": step,
    }
    (checkpoint / "resume_state.json").write_text(
        json.dumps({**shared, "examples_seen": step * 16}), encoding="utf-8"
    )
    (checkpoint / "checkpoint_metadata.json").write_text(
        json.dumps(
            {
                **shared,
                "target_representation": "category_id",
                "target_representation_version": "cuad-category-id-v1",
                "prompt_template_version": "cuad-classification-id-v2",
                "taxonomy_version": "cuad-v1-41",
                "stable_id_map_checksum": stable_id_map_checksum(),
                "lora_rank": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"offline fixture")
    (root / "experiment_config.json").write_text(
        json.dumps(
            {
                "model": {
                    "name": "Qwen/Qwen2.5-7B-Instruct",
                    "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
                    "precision": "float16",
                    "quantization": "4bit",
                    "quant_type": "nf4",
                    "double_quant": True,
                },
                "lora": {
                    "rank": 8,
                    "alpha": 16,
                    "dropout": 0.05,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                },
                "target_representation": "category_id",
                "target_representation_version": "cuad-category-id-v1",
                "prompt_template_version": "cuad-classification-id-v2",
            }
        ),
        encoding="utf-8",
    )
    return checkpoint


@pytest.mark.parametrize("step", [700, 800, 1400, 2100, 975])
def test_generic_checkpoint_candidate_creation(tmp_path: Path, step: int) -> None:
    checkpoint = _checkpoint(tmp_path, step)
    output = tmp_path / f"step{step}-manifest.json"
    manifest = create_candidate_manifest(
        checkpoint,
        f"clauseforge-qwen25-7b-r8-step{step}",
        output,
        "a" * 40,
    )
    assert manifest.checkpoint_step == step
    assert manifest.adapter_path == str(checkpoint.resolve())
    assert manifest.adapter_checksum
    assert manifest.validation_summary is None
    assert not manifest.test_evaluated
    assert validate_manifest(output).valid


def test_checkpoint_lineage_mismatch_is_rejected(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, 800)
    config_path = checkpoint / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["r"] = 16
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="LoRA rank mismatch"):
        create_candidate_manifest(
            checkpoint, "candidate-800", tmp_path / "m.json", "a" * 40
        )


def test_validation_evidence_is_separate_and_required_for_activation(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path, 800)
    manifest_path = tmp_path / "candidate.json"
    manifest = create_candidate_manifest(
        checkpoint, "candidate-800", manifest_path, "a" * 40
    )
    registry = ArtifactRegistry(tmp_path / "registry")
    metadata_only = replace(
        manifest,
        adapter_path=None,
        adapter_checksum=None,
        required_files={},
    )
    registry_source = tmp_path / "registry-source.json"
    write_manifest(registry_source, metadata_only)
    registry.register(registry_source)
    with pytest.raises(ValueError, match="validation evidence"):
        registry.activate("candidate-800")

    evidence = tmp_path / "validation.json"
    evidence.write_text(
        json.dumps(
            {
                "label": "VALIDATION ONLY — NOT FINAL MODEL PERFORMANCE",
                "split": "validation",
                "test_evaluated": False,
                "global_step": 800,
                "experiment_id": "qwen2.5-7b-instruct_lora-r8_seed42_0ad1c2c19e59",
                "prompt_version": "cuad-classification-id-v2",
                "target_representation": "category_id",
                "target_representation_version": "cuad-category-id-v1",
                "stable_id_map_checksum": stable_id_map_checksum(),
                "train_examples": 1000,
                "validation_examples": 128,
                "optimizer_steps": 800,
                "examples_seen": 12800,
                "accuracy": 0.5,
                "macro_f1": 0.48,
                "weighted_f1": 0.49,
                "exact_id_output_rate": 0.9,
                "invalid_output_rate": 0.1,
                "validation_loss": 1.2,
            }
        ),
        encoding="utf-8",
    )
    updated = attach_validation_evidence(manifest_path, evidence, manifest_path)
    assert updated.validation_summary is not None
    assert updated.validation_summary.macro_f1 == 0.48
    assert updated.validation_selection_completed
    assert not updated.test_evaluated
    validated_registry = ArtifactRegistry(tmp_path / "validated-registry")
    validated_registry.register(manifest_path)
    active = validated_registry.activate("candidate-800")
    assert active.current_artifact_id == "candidate-800"


def test_test_split_evidence_is_rejected(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, 800)
    manifest_path = tmp_path / "candidate.json"
    create_candidate_manifest(checkpoint, "candidate-800", manifest_path, "a" * 40)
    evidence = tmp_path / "test.json"
    evidence.write_text(
        json.dumps({"split": "test", "test_evaluated": True}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="validation split"):
        attach_validation_evidence(manifest_path, evidence, manifest_path)
