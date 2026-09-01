"""Small transparent training loop used for reproducible adapter experiments."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import torch

from clauseforge.training.config import OptimizationConfig
from clauseforge.training.templates import (
    TrainingExample,
    render_example,
    render_prompt,
)


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


def build_training_batch(
    example: TrainingExample, tokenizer: Any, max_length: int
) -> dict[str, Any]:
    encoded = tokenizer(
        render_example(example),
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    prompt_ids = tokenizer.encode(
        render_prompt(example.clause_text), add_special_tokens=True
    )
    encoded["labels"] = mask_prompt_tokens(encoded["input_ids"], len(prompt_ids))
    if bool((encoded["labels"] != -100).sum() == 0):
        raise ValueError(
            f"max sequence length truncates the complete target for {example.clause_id}"
        )
    return cast(dict[str, Any], encoded)


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
