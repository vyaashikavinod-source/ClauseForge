from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

from clauseforge.taxonomy import load_taxonomy_metadata
from clauseforge.training.compatibility import (
    CompatibilityReport,
    checkpoint_evaluation_compatibility,
    historical_experiment_id,
)
from clauseforge.training.config import TrainingConfig, load_config

FIXTURE = Path("tests/fixtures/phase3b_historical_checkpoint.json")
CONFIG = Path("training/configs/phase3b/qwen25_7b_qlora_r8.yaml")
CHECKSUM = "a287835bc6954916c2d9066e7a4b76b18c6946419290abf839f6996bc73ba458"


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
    )


def test_historical_diagnostics_only_field_is_evaluation_compatible() -> None:
    fixture = _fixture()
    config = load_config(CONFIG)
    assert historical_experiment_id(fixture["experiment_config"]).endswith(
        "ebbf06b3b6cf"
    )
    assert config.experiment_id.endswith("2b4ab5f08030")
    report = _report(config)
    assert report.compatible
    assert report.ignored_nonstructural_differences == (
        "data.validation_max_new_tokens",
    )
    assert report.evaluation_overrides["validation_max_new_tokens"] == 160
    assert not report.blocking_differences


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
