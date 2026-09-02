"""GPU-only tiny memorization harness for Phase 3B pipeline diagnosis."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

import torch

from clauseforge.training.audits import (
    audit_training_example,
    gradient_audit,
    learning_rate_audit,
    parameter_digest,
    prompt_length_audit,
    special_token_audit,
    target_round_trip_audit,
)
from clauseforge.training.checkpoints import save_adapter, write_json
from clauseforge.training.config import TrainingConfig
from clauseforge.training.dataset import TrainingDataset, build_training_dataset
from clauseforge.training.diagnostics import ValidationPrediction
from clauseforge.training.environment import environment_metadata
from clauseforge.training.lora import attach_lora, prepare_qlora_base
from clauseforge.training.model import load_production_model
from clauseforge.training.phase3b import (
    ResumeState,
    _append_log,
    _checkpoint,
    _generate_label,
    _optimizer,
    _scheduler,
    configure_gradient_checkpointing,
)
from clauseforge.training.pilot import stratified_selection
from clauseforge.training.targets import stable_id_map_checksum
from clauseforge.training.trainer import build_training_batch, seed_everything

OVERFIT_LABEL = "TRAINING MEMORIZATION DIAGNOSTIC — NOT MODEL PERFORMANCE"


def select_overfit_examples(
    dataset: TrainingDataset, *, count: int, seed: int
) -> tuple[Any, ...]:
    return stratified_selection(
        dataset.train, count=count, seed=seed, split="train"
    ).selected


def _named_parameters(model: Any, *, trainable: bool) -> list[tuple[str, Any]]:
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad is trainable
        and (("lora" in name.casefold()) if trainable else True)
    ][:1]


def overfit_prediction_record(
    record: ValidationPrediction, accepted: str | None
) -> dict[str, object]:
    return {
        "clause_id": record.clause_id,
        "target_id": record.canonical_target_id,
        "raw_generated_text": record.raw_generated_text,
        "normalized_output": record.normalized_generated_text,
        "valid_id": accepted is not None,
        "exact_match": record.exact_match,
        "generated_token_count": record.generated_token_count,
    }


def _safe_memorization_evaluation(
    model: Any,
    tokenizer: Any,
    examples: tuple[Any, ...],
    dataset: TrainingDataset,
    config: TrainingConfig,
    step: int,
    first_exact: dict[str, int],
    output: Path,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    exact = invalid = 0
    model.eval()
    for example in examples:
        accepted, record = _generate_label(
            model,
            tokenizer,
            example,
            dataset.taxonomy,
            config.data.max_sequence_length,
            config.data.validation_max_new_tokens,
            config.target_representation,
        )
        is_exact = record.exact_match
        exact += int(is_exact)
        invalid += int(accepted is None)
        if is_exact and example.clause_id not in first_exact:
            first_exact[example.clause_id] = step
        rows.append(overfit_prediction_record(record, accepted))
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    model.train()
    return {
        "step": step,
        "example_count": len(examples),
        "exact_id_accuracy": exact / len(examples),
        "exact_id_outputs": exact,
        "invalid_output_rate": invalid / len(examples),
        "first_exact_step": dict(sorted(first_exact.items())),
        "artifact": output.name,
        "label": OVERFIT_LABEL,
        "test_evaluated": False,
    }


def run_overfit_diagnostic(
    config: TrainingConfig,
    data_dir: Path,
    *,
    examples_count: int = 8,
    steps: int = 100,
    eval_every: int = 10,
) -> dict[str, object]:
    """Run bounded GPU memorization; this function must not be called locally."""
    if config.target_representation != "category_id":
        raise ValueError("overfit diagnostics require category-ID targets")
    if min(examples_count, steps, eval_every) <= 0:
        raise ValueError("overfit counts must be positive")
    source = build_training_dataset(
        data_dir,
        max_train_examples=config.data.max_train_examples,
        max_validation_examples=1,
        target_representation=config.target_representation,
        target_representation_version=config.target_representation_version,
        prompt_template_version=config.prompt_template_version,
    )
    examples = select_overfit_examples(source, count=examples_count, seed=config.seed)
    experiment_dir = config.output_dir / "overfit" / config.experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    write_json(experiment_dir / "experiment_config.json", config.to_dict())
    write_json(experiment_dir / "environment.json", environment_metadata())
    write_json(
        experiment_dir / "overfit_config.json",
        {
            "label": OVERFIT_LABEL,
            "examples": examples_count,
            "steps": steps,
            "eval_every": eval_every,
            "selected_example_ids": [item.clause_id for item in examples],
            "stable_id_map_checksum": stable_id_map_checksum(),
            "selection_split": "train",
            "test_evaluated": False,
        },
    )
    seed_everything(config.seed)
    model, tokenizer = load_production_model(config.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model = attach_lora(prepare_qlora_base(model), config.model, config.lora)
    configure_gradient_checkpointing(model, config)
    write_json(
        experiment_dir / "special_tokens.json", special_token_audit(model, tokenizer)
    )
    write_json(
        experiment_dir / "target_token_audit.json", target_round_trip_audit(tokenizer)
    )
    audits = [
        audit_training_example(
            item, tokenizer, config.data.max_sequence_length
        ).to_dict()
        for item in examples
    ]
    write_json(experiment_dir / "training_token_audit.json", {"examples": audits})
    write_json(
        experiment_dir / "prompt_length_audit.json",
        prompt_length_audit(source.train, tokenizer, config.data.max_sequence_length),
    )
    write_json(
        experiment_dir / "learning_rate_audit.json",
        learning_rate_audit(
            config.optimization.learning_rate,
            config.optimization.scheduler,
            config.optimization.warmup_ratio,
            steps,
        ),
    )
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(optimizer, steps, config)
    accumulation = config.optimization.gradient_accumulation
    trace_path = experiment_dir / "loss_trace.jsonl"
    first_exact: dict[str, int] = {}
    evaluations: list[dict[str, object]] = []
    losses: list[float] = []
    parameter_updates: list[dict[str, object]] = []
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad()
    state = ResumeState(0, 0, 0, config.experiment_id, "overfit", "")
    for step in range(1, steps + 1):
        step_losses: list[float] = []
        for offset in range(accumulation):
            example = examples[((step - 1) * accumulation + offset) % len(examples)]
            batch = build_training_batch(
                example, tokenizer, config.data.max_sequence_length
            )
            batch = {key: value.to(model.device) for key, value in batch.items()}
            output = model(**batch)
            (output.loss / accumulation).backward()
            step_losses.append(float(output.loss.detach()))
        gradients = gradient_audit(model)
        gradient_norm = float(gradients["aggregate_gradient_norm"])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        trainable_before = parameter_digest(_named_parameters(model, trainable=True))
        frozen_before = parameter_digest(_named_parameters(model, trainable=False))
        effective_lr = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        trainable_after = parameter_digest(_named_parameters(model, trainable=True))
        frozen_after = parameter_digest(_named_parameters(model, trainable=False))
        update: dict[str, object] = {
            "step": step,
            "effective_learning_rate": effective_lr,
            "trainable_parameter_changed": trainable_before != trainable_after,
            "frozen_parameter_unchanged": frozen_before == frozen_after,
        }
        parameter_updates.append(update)
        if frozen_before != frozen_after:
            raise ValueError("a frozen base parameter changed during optimization")
        scheduler.step()
        optimizer.zero_grad()
        loss = sum(step_losses) / len(step_losses)
        losses.append(loss)
        state = ResumeState(
            step, step * accumulation, 0, config.experiment_id, "overfit", ""
        )
        _append_log(
            trace_path,
            {
                "step": step,
                "loss": loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "gradient_norm": gradient_norm,
                **gradients,
            },
        )
        if step % eval_every == 0 or step == steps:
            evaluation = _safe_memorization_evaluation(
                model,
                tokenizer,
                examples,
                source,
                config,
                step,
                first_exact,
                experiment_dir / f"memorization_predictions_step-{step}.jsonl",
            )
            evaluations.append(evaluation)
            _checkpoint(experiment_dir, model, optimizer, scheduler, state)
    if not any(
        item["trainable_parameter_changed"]
        for item in parameter_updates
        if cast(float, item["effective_learning_rate"]) > 0
    ):
        raise ValueError("LoRA parameters did not change after optimization")
    save_adapter(model, experiment_dir)
    result = {
        "label": OVERFIT_LABEL,
        "training_loss": sum(losses) / len(losses),
        "steps": steps,
        "examples_seen": steps * accumulation,
        "runtime_seconds": time.perf_counter() - started,
        "evaluations": evaluations,
        "parameter_update_audit": parameter_updates,
        "test_evaluated": False,
    }
    write_json(experiment_dir / "overfit_metrics.json", result)
    return result
