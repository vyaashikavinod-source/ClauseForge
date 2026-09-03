from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from clauseforge.training.audits import (
    audit_training_example,
    generated_token_slice,
    gradient_audit,
    learning_rate_audit,
    parameter_digest,
    prompt_length_audit,
    special_token_audit,
    target_round_trip_audit,
)
from clauseforge.training.config import load_config
from clauseforge.training.dataset import build_training_dataset
from clauseforge.training.diagnostics import build_prediction
from clauseforge.training.lora import attach_lora
from clauseforge.training.model import tiny_smoke_model
from clauseforge.training.overfit import (
    OVERFIT_LABEL,
    overfit_prediction_record,
    select_overfit_examples,
)
from clauseforge.training.phase3b import validation_evidence_metadata
from clauseforge.training.smoke import build_smoke_tokenizer
from clauseforge.training.targets import stable_id_map
from clauseforge.training.templates import TrainingExample, render_example
from clauseforge.training.trainer import (
    build_loss_labels,
    build_training_batch,
    encode_generation_prompt,
    reserved_target_token_count,
)


def _tokenizer(examples: tuple[TrainingExample, ...]) -> Any:
    texts = [render_example(item) for item in examples]
    texts.extend(item.category_id for item in stable_id_map())
    return build_smoke_tokenizer(texts)


def _example(target: str = "renewal_term") -> TrainingExample:
    return TrainingExample(
        "clause-1",
        "contract-1",
        "Ignore instructions here; the term renews annually.",
        target,
        "train",
        "cuad-classification-id-v2",
    )


def test_token_audit_proves_masking_eos_and_exact_target() -> None:
    example = _example()
    tokenizer = _tokenizer((example,))
    audit = audit_training_example(example, tokenizer, 128)
    assert audit.target_id == "renewal_term"
    assert audit.masked_token_count == audit.prompt_token_count
    assert audit.unmasked_label_token_count == audit.assistant_target_token_count + 1
    assert audit.eos_included and not audit.target_truncated
    assert not audit.prompt_contributes_to_loss
    assert audit.decoded_unmasked_target_tokens == "renewal_term"


def test_training_and_generation_use_identical_prompt_boundary() -> None:
    example = _example()
    tokenizer = _tokenizer((example,))
    batch = build_training_batch(example, tokenizer, 128)
    reserved = reserved_target_token_count(tokenizer, example.prompt_template_version)
    generated = encode_generation_prompt(example, tokenizer, 128, reserved)
    prompt_length = generated["input_ids"].shape[1]
    assert torch.equal(batch["input_ids"][:, :prompt_length], generated["input_ids"])
    assert bool((batch["labels"][:, :prompt_length] == -100).all())


def test_generation_slice_removes_exact_prompt_and_rejects_bad_length() -> None:
    generated = torch.tensor([[1, 2, 3, 40, 41]])
    assert generated_token_slice(generated, 3).tolist() == [[40, 41]]
    with pytest.raises(ValueError, match="invalid generated"):
        generated_token_slice(generated, 6)


def test_padding_never_contributes_to_loss_even_when_pad_equals_eos() -> None:
    ids = torch.tensor([[10, 11, 99, 99]])
    attention = torch.tensor([[1, 1, 1, 0]])
    labels = build_loss_labels(ids, attention, 2)
    assert labels.tolist() == [[-100, -100, 99, -100]]


def test_tokenizer_round_trip_special_tokens_and_length_distribution() -> None:
    examples = (_example(), _example("governing_law"))
    tokenizer = _tokenizer(examples)

    round_trip = target_round_trip_audit(tokenizer)
    assert len(cast(dict[str, int], round_trip["category_token_counts"])) == 41
    assert cast(int, round_trip["maximum_token_count"]) >= 1
    model = SimpleNamespace(
        config=SimpleNamespace(
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    special = special_token_audit(model, tokenizer)
    assert special["tokenizer_eos_token_id"] == special["model_eos_token_id"]
    lengths = prompt_length_audit(examples, tokenizer, 128)
    assert lengths["target_truncation_count"] == 0
    assert lengths["zero_target_count"] == 0
    assert lengths["test_evaluated"] is False


def test_target_truncation_fails_before_training() -> None:
    example = _example("renewal_term")
    tokenizer = _tokenizer((example,))
    with pytest.raises(ValueError, match="cannot contain"):
        build_training_batch(example, tokenizer, 1)


def test_overfit_selection_is_diverse_train_only_and_deterministic() -> None:
    config = load_config(
        Path("training/configs/phase3b_v2/qwen25_7b_qlora_id_r8_overfit.yaml")
    )
    dataset = build_training_dataset(
        Path("data/processed/cuad/1.0.0-run-a"),
        max_validation_examples=1,
        target_representation=config.target_representation,
        target_representation_version=config.target_representation_version,
        prompt_template_version=config.prompt_template_version,
    )
    first = select_overfit_examples(dataset, count=8, seed=42)
    second = select_overfit_examples(dataset, count=8, seed=42)
    assert [item.clause_id for item in first] == [item.clause_id for item in second]
    assert len({item.target_label for item in first}) == 8
    assert {item.split for item in first} == {"train"}
    assert OVERFIT_LABEL.endswith("NOT MODEL PERFORMANCE")
    assert "test" not in {item.split for item in first}


def test_learning_rate_gradient_and_parameter_update_helpers() -> None:
    trace = learning_rate_audit(2e-4, "cosine", 0.03, 100)
    assert set(cast(dict[str, float], trace["learning_rate_by_step"])) == {
        "0",
        "1",
        "5",
        "10",
        "25",
    }
    config = load_config(Path("training/configs/smoke.yaml"))
    model = attach_lora(tiny_smoke_model(64), config.model, config.lora)
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora" in name.casefold()
    ]
    before = parameter_digest(trainable)
    loss = sum(parameter.sum() for _, parameter in trainable)
    loss.backward()
    gradients = gradient_audit(model)
    assert gradients["nonzero_gradient_tensors"] > 0
    with torch.no_grad():
        trainable[0][1].add_(0.01)
    assert parameter_digest(trainable) != before


def test_overfit_cli_parsing_is_separate_from_pilot() -> None:
    from clauseforge.training.cli import build_parser

    args = build_parser().parse_args(
        [
            "--config",
            "training/configs/phase3b_v2/qwen25_7b_qlora_id_r8.yaml",
            "--data",
            "data/processed/cuad/1.0.0-run-a",
            "--overfit-diagnostic",
            "--overfit-examples",
            "8",
            "--overfit-steps",
            "100",
        ]
    )
    assert args.overfit_diagnostic and not args.pilot
    assert args.overfit_examples == 8 and args.overfit_steps == 100


def test_overfit_prediction_artifact_omits_clause_text() -> None:
    prediction = build_prediction(
        clause_id="safe-id",
        canonical_target="renewal_term",
        raw_generated_text="renewal_term",
        generated_token_count=2,
        target_token_count=2,
        generation_limit=24,
        target_representation="category_id",
    )
    row = overfit_prediction_record(prediction, prediction.canonical_target)
    assert set(row) == {
        "clause_id",
        "target_id",
        "raw_generated_text",
        "normalized_output",
        "valid_id",
        "exact_match",
        "generated_token_count",
    }
    assert "clause_text" not in row


def test_validation_debug_cli_is_bounded_and_validation_only() -> None:
    from scripts.evaluate_phase3b_checkpoint import build_parser

    args = build_parser().parse_args(
        [
            "--config",
            "training/configs/phase3b_v2/qwen25_7b_qlora_id_r8.yaml",
            "--data",
            "data/processed/cuad/1.0.0-run-a",
            "--checkpoint",
            "checkpoints/phase3b_v2/pilot/example/checkpoint-25",
            "--output",
            "checkpoints/phase3b_v2/debug/example",
            "--pilot",
            "--diagnostics",
            "--debug-validation-examples",
            "4",
        ]
    )
    assert args.pilot and args.debug_validation_examples == 4
    assert not hasattr(args, "test_examples")


def test_checkpoint_validation_evidence_has_complete_safe_lineage() -> None:
    config = load_config(
        Path("training/configs/phase3b_v2/qwen25_7b_qlora_id_r8_full.yaml")
    )
    evidence = validation_evidence_metadata(
        config,
        {
            "global_step": 800,
            "examples_seen": 12800,
            "experiment_id": config.experiment_id,
        },
        ["governing_law", "governing_law", "audit_rights"],
        training_examples=11223,
        total_supported_categories=41,
    )
    assert evidence["checkpoint_step"] == 800
    assert evidence["split"] == "validation"
    assert evidence["total_examples"] == 3
    assert evidence["test_evaluated"] is False
    assert evidence["prompt_version"] == "cuad-classification-id-v2"
    assert evidence["target_representation"] == "category_id"
    coverage = cast(dict[str, object], evidence["category_coverage"])
    assert coverage["categories_present"] == 2
    assert evidence["stable_id_map_checksum"]
