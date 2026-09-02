from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from clauseforge.taxonomy import category_by_id
from clauseforge.training.config import load_config
from clauseforge.training.dataset import build_training_dataset
from clauseforge.training.diagnostics import aggregate_diagnostics, build_prediction
from clauseforge.training.targets import (
    UnknownTargetError,
    resolve_generated_target,
    stable_id_map,
    stable_id_map_checksum,
    target_for_canonical,
)
from clauseforge.training.templates import render_example
from clauseforge.training.trainer import build_training_batch

CONFIG = Path("training/configs/phase3b_v2/qwen25_7b_qlora_id_r8.yaml")


def test_stable_ids_are_unique_reversible_and_checksummed() -> None:
    categories = stable_id_map()
    assert len(categories) == len({item.category_id for item in categories}) == 41
    assert all(
        re.fullmatch(r"[a-z][a-z0-9_]*", item.category_id) for item in categories
    )
    assert all(
        target_for_canonical(item.canonical, "category_id") == item.category_id
        and resolve_generated_target(item.category_id, "category_id") == item
        for item in categories
    )
    assert stable_id_map_checksum() == (
        "6a1f61e698a84c0051c7356ae60e52c2e7d57f38300ea213d50a110ada847a8d"
    )


def test_v2_dataset_uses_ids_and_never_loads_test() -> None:
    config = load_config(CONFIG)
    dataset = build_training_dataset(
        Path("data/processed/cuad/1.0.0-run-a"),
        max_train_examples=3,
        max_validation_examples=2,
        target_representation=config.target_representation,
        target_representation_version=config.target_representation_version,
        prompt_template_version=config.prompt_template_version,
    )
    allowed = {item.category_id for item in stable_id_map()}
    assert {item.target_label for item in dataset.train + dataset.validation} <= allowed
    assert {item.split for item in dataset.train} == {"train"}
    assert {item.split for item in dataset.validation} == {"validation"}
    assert "no examples materialized" in str(dataset.manifest["test_split_access"])
    assert render_example(dataset.train[0]).endswith(dataset.train[0].target_label)
    assert "exactly one valid CUAD category ID" in render_example(dataset.train[0])


def test_id_validation_and_diagnostics_remain_strict() -> None:
    renewal = category_by_id("renewal_term")
    assert resolve_generated_target(" renewal_term\n", "category_id") == renewal
    for invalid in (
        "Renewal Term",
        renewal.canonical,
        "The category is renewal_term",
        "unknown_category",
        "renewal term",
    ):
        with pytest.raises(UnknownTargetError):
            resolve_generated_target(invalid, "category_id")
    records = [
        build_prediction(
            clause_id=f"validation-{index}",
            canonical_target="renewal_term",
            raw_generated_text=raw,
            generated_token_count=4,
            target_token_count=2,
            generation_limit=24,
            target_representation="category_id",
        )
        for index, raw in enumerate(
            (
                "renewal_term",
                "Renewal Term",
                renewal.canonical,
                "The category is renewal_term",
                "bad_id",
            )
        )
    ]
    counts = aggregate_diagnostics(records)
    assert counts["exact_id_matches"] == 1
    assert counts["short_name_outputs"] == 1
    assert counts["canonical_question_outputs"] == 1
    assert counts["commentary_wrapped_ids"] == 1
    assert counts["invalid_ids"] == 1
    assert sum(item.exact_match for item in records) == 1


class _Tokenizer:
    eos_token_id = 999
    pad_token_id = 0

    def encode(self, text: str, **_: object) -> list[int]:
        return list(range(len(text)))

    def __call__(self, text: str, **_: object) -> dict[str, torch.Tensor]:
        values = torch.tensor([self.encode(text)])
        return {"input_ids": values, "attention_mask": torch.ones_like(values)}


def test_assistant_only_masking_still_applies_to_id_target() -> None:
    dataset = build_training_dataset(
        Path("data/processed/cuad/1.0.0-run-a"),
        max_train_examples=1,
        max_validation_examples=1,
        target_representation="category_id",
        target_representation_version="cuad-category-id-v1",
        prompt_template_version="cuad-classification-id-v2",
    )
    batch = build_training_batch(dataset.train[0], _Tokenizer(), 1024)
    labels = batch["labels"][0].tolist()
    assert -100 in labels
    assert any(value != -100 for value in labels)
