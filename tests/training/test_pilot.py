from __future__ import annotations

from pathlib import Path

import pytest

from clauseforge.training.cli import build_parser, run
from clauseforge.training.config import load_config
from clauseforge.training.dataset import build_training_dataset
from clauseforge.training.phase3b import ResumeState, validate_resume_compatibility
from clauseforge.training.pilot import (
    PILOT_LABEL,
    build_pilot_dataset,
    combined_selection_checksum,
    stratified_selection,
)
from clauseforge.training.templates import TrainingExample


def _examples(split: str) -> tuple[TrainingExample, ...]:
    return tuple(
        TrainingExample(
            str(index), f"contract-{index}", "text", f"category-{index % 4}", split
        )
        for index in range(20)
    )


def test_stratified_pilot_selection_is_deterministic_and_covers_categories() -> None:
    first = stratified_selection(_examples("train"), count=8, seed=42, split="train")
    second = stratified_selection(_examples("train"), count=8, seed=42, split="train")
    assert first.selected_example_ids == second.selected_example_ids
    assert first.checksum == second.checksum
    assert first.covered_category_count == first.supported_category_count == 4
    assert set(first.category_counts.values()) == {2}
    assert first.metadata()["label"] == PILOT_LABEL


def test_pilot_selection_rejects_wrong_split_and_oversize() -> None:
    with pytest.raises(ValueError, match="another split"):
        stratified_selection(_examples("validation"), count=4, seed=42, split="train")
    with pytest.raises(ValueError, match="exceeds"):
        stratified_selection(_examples("train"), count=21, seed=42, split="train")


def test_real_pilot_uses_train_and_validation_only() -> None:
    source = build_training_dataset(Path("data/processed/cuad/1.0.0-run-a"))
    pilot, train, validation = build_pilot_dataset(source)
    assert len(pilot.train) == 512 and len(pilot.validation) == 256
    assert all(item.split == "train" for item in pilot.train)
    assert all(item.split == "validation" for item in pilot.validation)
    assert train.covered_category_count == 41
    assert (
        validation.covered_category_count == validation.supported_category_count == 40
    )
    assert train.metadata()["test_examples_selected"] == 0
    assert validation.metadata()["test_examples_selected"] == 0
    assert "test" not in {item.split for item in pilot.train + pilot.validation}


def test_pilot_cli_and_offline_metadata() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "training/configs/phase3b/qwen25_7b_qlora_r8.yaml",
            "--data",
            "data/processed/cuad/1.0.0-run-a",
            "--pilot",
            "--pilot-train-examples",
            "512",
            "--pilot-validation-examples",
            "256",
            "--max-steps",
            "12",
        ]
    )
    assert args.pilot and args.max_steps == 12
    config = load_config(args.config)
    result = run(config, args.data, dry_run=True, pilot=True)
    assert result["status"] == "phase3b-pilot-config-valid"
    manifest = result["dataset_manifest"]
    assert isinstance(manifest, dict) and manifest["label"] == PILOT_LABEL
    assert result["test_evaluated"] is False
    assert "macro_f1" not in result and "accuracy" not in result


def test_resume_requires_same_pilot_selection() -> None:
    train = stratified_selection(_examples("train"), count=8, seed=42, split="train")
    validation = stratified_selection(
        _examples("validation"), count=8, seed=42, split="validation"
    )
    checksum = combined_selection_checksum(train, validation)
    state = ResumeState(5, 80, 0, "experiment", "pilot", checksum)
    validate_resume_compatibility(state, "experiment", "pilot", checksum)
    with pytest.raises(ValueError, match="incompatible"):
        validate_resume_compatibility(state, "experiment", "pilot", "different")
    with pytest.raises(ValueError, match="incompatible"):
        validate_resume_compatibility(state, "experiment", "full", checksum)
