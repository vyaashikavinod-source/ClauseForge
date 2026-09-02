from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from clauseforge.taxonomy import load_taxonomy_metadata
from clauseforge.training.compatibility import (
    CompatibilityReport,
    checkpoint_evaluation_compatibility,
    historical_experiment_id,
)
from clauseforge.training.config import TrainingConfig, load_config
from clauseforge.training.dataset import build_training_dataset
from clauseforge.training.pilot import (
    build_pilot_dataset,
    combined_selection_checksum,
    diagnostic_validation_selection,
    restore_pilot_selection,
)

FIXTURE = Path("tests/fixtures/phase3b_historical_checkpoint.json")
CONFIG = Path("training/configs/phase3b/qwen25_7b_qlora_r8.yaml")
V2_CONFIG = Path("training/configs/phase3b_v2/qwen25_7b_qlora_id_r8.yaml")
CHECKSUM = "a287835bc6954916c2d9066e7a4b76b18c6946419290abf839f6996bc73ba458"
DEFAULT_RECONSTRUCTION_CHECKSUM = (
    "d16b99769131b050fde76cd4a8016f6cf394d030b9e8e338664006e25ff9bbf6"
)


def _fixture() -> dict[str, dict[str, object]]:
    return cast(
        dict[str, dict[str, object]],
        json.loads(FIXTURE.read_text(encoding="utf-8")),
    )


def _report(
    config: TrainingConfig,
    *,
    mutate: tuple[str, str] | None = None,
    taxonomy: tuple[str, ...] | None = None,
) -> CompatibilityReport:
    fixture = copy.deepcopy(_fixture())
    if mutate is not None:
        section, field = mutate
        fixture[section][field] = "different"
    canonical = tuple(item.canonical for item in load_taxonomy_metadata())
    return checkpoint_evaluation_compatibility(
        config,
        fixture["experiment_config"],
        fixture["pilot_config"],
        fixture["resume_state"],
        fixture["adapter_metadata"],
        expected_run_mode="pilot",
        expected_subset_checksum=CHECKSUM,
        historical_taxonomy=list(taxonomy or canonical),
        current_taxonomy=canonical,
        historical_training_subset={"selected_count": 256, "checksum": "train"},
        historical_validation_subset={
            "selected_count": 128,
            "checksum": "validation",
        },
        evaluation_validation_subset={
            "selected_count": 128,
            "checksum": "validation",
        },
        evaluation_override=False,
    )


def test_historical_diagnostics_only_field_is_evaluation_compatible() -> None:
    fixture = _fixture()
    config = load_config(CONFIG)
    assert historical_experiment_id(fixture["experiment_config"]).endswith(
        "ebbf06b3b6cf"
    )
    assert config.experiment_id.endswith("2fb2d147fc8e")
    report = _report(config)
    assert report.compatible
    assert report.ignored_nonstructural_differences == (
        "data.validation_max_new_tokens",
    )
    assert report.evaluation_overrides["validation_max_new_tokens"] == 160
    assert not report.blocking_differences
    assert report.training_subset_match and report.validation_subset_match
    assert not report.evaluation_override


def test_model_revision_rank_and_targets_block_evaluation() -> None:
    config = load_config(CONFIG)
    revision = replace(config, model=replace(config.model, revision="different"))
    rank = replace(config, lora=replace(config.lora, rank=16))
    targets = replace(config, lora=replace(config.lora, target_modules=("q_proj",)))
    assert "model.revision" in _report(revision).blocking_differences
    assert "lora.rank" in _report(rank).blocking_differences
    assert "lora.target_modules" in _report(targets).blocking_differences


def test_identity_subset_prompt_and_taxonomy_block_evaluation() -> None:
    config = load_config(CONFIG)
    identity = _report(config, mutate=("resume_state", "experiment_id"))
    subset = _report(config, mutate=("resume_state", "subset_checksum"))
    prompt = _report(replace(config, prompt_template_version="incompatible-template"))
    taxonomy = tuple(item.canonical for item in load_taxonomy_metadata())[:-1]
    assert "experiment_identity" in identity.blocking_differences
    assert "selected_subset_checksum" in subset.blocking_differences
    assert "prompt_template_version" in prompt.blocking_differences
    assert "taxonomy" in _report(config, taxonomy=taxonomy).blocking_differences


def test_historical_and_default_reconstruction_checksums_are_explicit() -> None:
    source = build_training_dataset(Path("data/processed/cuad/1.0.0-run-a"))
    _, historical_train, historical_validation = build_pilot_dataset(
        source, train_count=256, validation_count=128, seed=42
    )
    _, default_train, default_validation = build_pilot_dataset(
        source, train_count=512, validation_count=128, seed=42
    )
    assert (
        combined_selection_checksum(historical_train, historical_validation) == CHECKSUM
    )
    assert (
        combined_selection_checksum(default_train, default_validation)
        == DEFAULT_RECONSTRUCTION_CHECKSUM
    )
    assert (
        restore_pilot_selection(
            source.train, historical_train.metadata(), split="train"
        ).selected_example_ids
        == historical_train.selected_example_ids
    )


def test_corrupted_persisted_selection_ids_and_checksums_fail() -> None:
    source = build_training_dataset(Path("data/processed/cuad/1.0.0-run-a"))
    _, train, _ = build_pilot_dataset(
        source, train_count=256, validation_count=128, seed=42
    )
    bad_checksum = train.metadata()
    bad_checksum["checksum"] = "corrupt"
    with pytest.raises(ValueError, match="stored=corrupt, calculated="):
        restore_pilot_selection(source.train, bad_checksum, split="train")
    bad_ids = train.metadata()
    ids = cast(list[str], bad_ids["selected_example_ids"])
    ids[0] = "unknown-clause"
    with pytest.raises(ValueError, match="unknown IDs"):
        restore_pilot_selection(source.train, bad_ids, split="train")


def test_validation_size_matches_history_or_is_labeled_override() -> None:
    source = build_training_dataset(Path("data/processed/cuad/1.0.0-run-a"))
    _, _, historical = build_pilot_dataset(
        source, train_count=256, validation_count=128, seed=42
    )
    same, same_override = diagnostic_validation_selection(
        source.validation, historical, requested_count=128, seed=42
    )
    changed, changed_override = diagnostic_validation_selection(
        source.validation, historical, requested_count=64, seed=42
    )
    assert same.selected_example_ids == historical.selected_example_ids
    assert not same_override
    assert changed.requested_count == 64 and changed_override
    assert changed.checksum != historical.checksum


def test_target_representation_is_model_critical() -> None:
    v1 = load_config(CONFIG)
    v2 = load_config(V2_CONFIG)
    fixture = _fixture()
    assert _report(v1).compatible
    assert "target_representation" in _report(v2).blocking_differences

    historical = v2.to_dict()
    resume = copy.deepcopy(fixture["resume_state"])
    resume["experiment_id"] = historical_experiment_id(historical)
    adapter = copy.deepcopy(fixture["adapter_metadata"])
    adapter.update(
        {
            "target_representation": "category_id",
            "target_representation_version": "cuad-category-id-v1",
        }
    )
    from clauseforge.training.targets import stable_id_map_checksum

    adapter["stable_id_map_checksum"] = stable_id_map_checksum()
    report = checkpoint_evaluation_compatibility(
        v2,
        historical,
        fixture["pilot_config"],
        resume,
        adapter,
        expected_run_mode="pilot",
        expected_subset_checksum=CHECKSUM,
        historical_taxonomy=[item.canonical for item in load_taxonomy_metadata()],
        current_taxonomy=tuple(item.canonical for item in load_taxonomy_metadata()),
        historical_training_subset={"selected_count": 256},
        historical_validation_subset={"selected_count": 128},
        evaluation_validation_subset={"selected_count": 128},
        evaluation_override=False,
    )
    assert report.compatible
    reverse = checkpoint_evaluation_compatibility(
        v1,
        historical,
        fixture["pilot_config"],
        resume,
        adapter,
        expected_run_mode="pilot",
        expected_subset_checksum=CHECKSUM,
        historical_taxonomy=[item.canonical for item in load_taxonomy_metadata()],
        current_taxonomy=tuple(item.canonical for item in load_taxonomy_metadata()),
        historical_training_subset={"selected_count": 256},
        historical_validation_subset={"selected_count": 128},
        evaluation_validation_subset={"selected_count": 128},
        evaluation_override=False,
    )
    assert "target_representation" in reverse.blocking_differences
    adapter["stable_id_map_checksum"] = "different"
    incompatible = checkpoint_evaluation_compatibility(
        v2,
        historical,
        fixture["pilot_config"],
        resume,
        adapter,
        expected_run_mode="pilot",
        expected_subset_checksum=CHECKSUM,
        historical_taxonomy=[item.canonical for item in load_taxonomy_metadata()],
        current_taxonomy=tuple(item.canonical for item in load_taxonomy_metadata()),
        historical_training_subset={"selected_count": 256},
        historical_validation_subset={"selected_count": 128},
        evaluation_validation_subset={"selected_count": 128},
        evaluation_override=False,
    )
    assert "stable_id_map_checksum" in incompatible.blocking_differences
