"""Small transparent training loop used for reproducible adapter experiments."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import torch

from clauseforge.training.config import OptimizationConfig
from clauseforge.training.targets import stable_id_map
from clauseforge.training.templates import (
    ID_PROMPT_TEMPLATE_VERSION,
    TrainingExample,
    render_prompt,
)

_TARGET_TOKEN_RESERVATIONS: dict[tuple[int, str], int] = {}


@dataclass(frozen=True, slots=True)
class TrainingMetrics:
    training_loss: float
    validation_loss: float
    learning_rate: float
    epochs: int
    steps: int
    runtime_seconds: float
    examples_per_second: float
    peak_gpu_memory_bytes: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mask_prompt_tokens(labels: Any, prompt_length: int) -> Any:
    """Return labels with prompt positions excluded from causal-LM loss."""
    masked = labels.clone()
    masked[:, : min(prompt_length, masked.shape[1])] = -100
    return masked


def build_loss_labels(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_length: int
) -> torch.Tensor:
    labels = mask_prompt_tokens(input_ids, prompt_length)
    labels[attention_mask == 0] = -100
    return labels


def target_token_ids(tokenizer: Any, target: str) -> list[int]:
    values = tokenizer.encode(target, add_special_tokens=False)
    if not values:
        raise ValueError("assistant target tokenizes to zero tokens")
    return cast(list[int], values)


def reserved_target_token_count(tokenizer: Any, prompt_template_version: str) -> int:
    key = (id(tokenizer), prompt_template_version)
    cached = _TARGET_TOKEN_RESERVATIONS.get(key)
    if cached is not None:
        return cached
    targets = (
        [item.category_id for item in stable_id_map()]
        if prompt_template_version == ID_PROMPT_TEMPLATE_VERSION
        else [item.canonical for item in stable_id_map()]
    )
    count = max(len(target_token_ids(tokenizer, target)) for target in targets)
    _TARGET_TOKEN_RESERVATIONS[key] = count
    return count


def prompt_token_ids(
    example: TrainingExample,
    tokenizer: Any,
    max_length: int,
    target_length: int,
) -> tuple[list[int], bool]:
    """Encode the shared generation prompt while reserving the complete target."""
    eos_count = 1 if tokenizer.eos_token_id is not None else 0
    budget = max_length - target_length - eos_count
    if budget <= 0:
        raise ValueError("max sequence length cannot contain the assistant target")
    values = cast(
        list[int],
        tokenizer.encode(
            render_prompt(example.clause_text, example.prompt_template_version),
            add_special_tokens=True,
        ),
    )
    return values[:budget], len(values) > budget


def encode_generation_prompt(
    example: TrainingExample,
    tokenizer: Any,
    max_length: int,
    reserved_target_tokens: int,
) -> dict[str, Any]:
    ids, _ = prompt_token_ids(example, tokenizer, max_length, reserved_target_tokens)
    input_ids = torch.tensor([ids], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def build_training_batch(
    example: TrainingExample, tokenizer: Any, max_length: int
) -> dict[str, Any]:
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer EOS token is required for supervised training")
    target_ids = target_token_ids(tokenizer, example.target_label)
    reserved = reserved_target_token_count(tokenizer, example.prompt_template_version)
    prompt_ids, _ = prompt_token_ids(example, tokenizer, max_length, reserved)
    eos_ids = [int(tokenizer.eos_token_id)]
    sequence = prompt_ids + target_ids + eos_ids
    input_ids = torch.tensor([sequence], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = build_loss_labels(input_ids, attention_mask, len(prompt_ids))
    unmasked = labels[0][labels[0] != -100].tolist()
    expected = target_ids + eos_ids
    if unmasked != expected:
        raise ValueError(f"assistant target alignment failed for {example.clause_id}")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _mean_loss(
    model: Any, examples: tuple[TrainingExample, ...], tokenizer: Any, max_length: int
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for example in examples:
            output = model(**build_training_batch(example, tokenizer, max_length))
            losses.append(float(output.loss.detach()))
    return sum(losses) / len(losses)


def train_adapter(
    model: Any,
    tokenizer: Any,
    train: tuple[TrainingExample, ...],
    validation: tuple[TrainingExample, ...],
    optimization: OptimizationConfig,
    max_length: int,
    seed: int,
) -> TrainingMetrics:
    if not train or not validation:
        raise ValueError("training and validation examples are required")
    seed_everything(seed)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=optimization.learning_rate,
        weight_decay=optimization.weight_decay,
    )
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad()
    losses: list[float] = []
    steps = 0
    for _epoch in range(optimization.epochs):
        for index, example in enumerate(train, 1):
            output = model(**build_training_batch(example, tokenizer, max_length))
            loss = output.loss / optimization.gradient_accumulation
            loss.backward()
            losses.append(float(output.loss.detach()))
            if index % optimization.gradient_accumulation == 0 or index == len(train):
                optimizer.step()
                optimizer.zero_grad()
                steps += 1
    runtime = time.perf_counter() - started
    return TrainingMetrics(
        training_loss=sum(losses) / len(losses),
        validation_loss=_mean_loss(model, validation, tokenizer, max_length),
        learning_rate=optimization.learning_rate,
        epochs=optimization.epochs,
        steps=steps,
        runtime_seconds=runtime,
        examples_per_second=len(train) * optimization.epochs / runtime,
        peak_gpu_memory_bytes=(
            torch.cuda.max_memory_allocated(0) if torch.cuda.is_available() else None
        ),
    )
