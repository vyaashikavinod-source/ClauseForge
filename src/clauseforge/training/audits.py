"""Safe CPU/GPU diagnostics for causal-LM target alignment and optimization."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

import torch

from clauseforge.training.targets import stable_id_map
from clauseforge.training.templates import TrainingExample, render_prompt
from clauseforge.training.trainer import (
    build_training_batch,
    prompt_token_ids,
    reserved_target_token_count,
    target_token_ids,
)


@dataclass(frozen=True, slots=True)
class TokenAudit:
    clause_id: str
    target_id: str
    raw_prompt_text_length: int
    prompt_token_count: int
    full_sequence_token_count: int
    assistant_target_token_count: int
    masked_token_count: int
    unmasked_label_token_count: int
    eos_included: bool
    prompt_truncated: bool
    target_truncated: bool
    prompt_contributes_to_loss: bool
    decoded_unmasked_target_tokens: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_training_example(
    example: TrainingExample, tokenizer: Any, max_length: int
) -> TokenAudit:
    target_ids = target_token_ids(tokenizer, example.target_label)
    reserved = reserved_target_token_count(tokenizer, example.prompt_template_version)
    prompt_ids, prompt_truncated = prompt_token_ids(
        example, tokenizer, max_length, reserved
    )
    batch = build_training_batch(example, tokenizer, max_length)
    labels = batch["labels"][0]
    unmasked = labels[labels != -100].tolist()
    eos_id = tokenizer.eos_token_id
    eos_included = eos_id is not None and bool(unmasked) and unmasked[-1] == eos_id
    decoded_ids = unmasked[:-1] if eos_included else unmasked
    decoded = str(tokenizer.decode(decoded_ids, skip_special_tokens=True)).strip()
    audit = TokenAudit(
        clause_id=example.clause_id,
        target_id=example.target_label,
        raw_prompt_text_length=len(
            render_prompt(example.clause_text, example.prompt_template_version)
        ),
        prompt_token_count=len(prompt_ids),
        full_sequence_token_count=int(batch["input_ids"].shape[1]),
        assistant_target_token_count=len(target_ids),
        masked_token_count=int((labels == -100).sum()),
        unmasked_label_token_count=int((labels != -100).sum()),
        eos_included=eos_included,
        prompt_truncated=prompt_truncated,
        target_truncated=False,
        prompt_contributes_to_loss=bool((labels[: len(prompt_ids)] != -100).any()),
        decoded_unmasked_target_tokens=decoded,
    )
    if audit.prompt_contributes_to_loss or decoded != example.target_label:
        raise ValueError(f"training mask invariant failed for {example.clause_id}")
    if audit.unmasked_label_token_count < len(target_ids):
        raise ValueError(f"assistant target was truncated for {example.clause_id}")
    return audit


def target_round_trip_audit(tokenizer: Any) -> dict[str, object]:
    counts: dict[str, int] = {}
    failures: list[str] = []
    for category in stable_id_map():
        ids = target_token_ids(tokenizer, category.category_id)
        decoded = str(tokenizer.decode(ids, skip_special_tokens=True)).strip()
        counts[category.category_id] = len(ids)
        if decoded != category.category_id:
            failures.append(category.category_id)
    if failures:
        raise ValueError(f"category IDs fail tokenizer round trip: {failures}")
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "category_token_counts": dict(sorted(counts.items())),
        "token_length_frequency": dict(sorted(Counter(counts.values()).items())),
        "maximum_token_count": ordered[0][1],
        "unusually_long_ids": [
            name for name, count in ordered if count == ordered[0][1]
        ],
    }


def _percentile(values: list[int], fraction: float) -> int:
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def prompt_length_audit(
    examples: Iterable[TrainingExample], tokenizer: Any, max_length: int
) -> dict[str, object]:
    prompts: list[int] = []
    full: list[int] = []
    targets: list[int] = []
    prompt_truncated = target_truncated = zero_targets = 0
    for example in examples:
        target_ids = target_token_ids(tokenizer, example.target_label)
        reserved = reserved_target_token_count(
            tokenizer, example.prompt_template_version
        )
        prompt_ids, truncated = prompt_token_ids(
            example, tokenizer, max_length, reserved
        )
        prompts.append(len(prompt_ids))
        targets.append(len(target_ids))
        full.append(
            len(prompt_ids) + len(target_ids) + (tokenizer.eos_token_id is not None)
        )
        prompt_truncated += int(truncated)
        target_truncated += int(len(target_ids) + 1 > max_length)
        zero_targets += int(not target_ids)

    def distribution(values: list[int]) -> dict[str, int | float]:
        return {
            "min": min(values),
            "median": median(values),
            "p90": _percentile(values, 0.90),
            "p95": _percentile(values, 0.95),
            "p99": _percentile(values, 0.99),
            "max": max(values),
        }

    count = len(prompts)
    approaching = sum(value >= int(max_length * 0.9) for value in full)
    return {
        "prompt_tokens": distribution(prompts),
        "full_sequence_tokens": distribution(full),
        "target_tokens": distribution(targets),
        "approaching_max_count": approaching,
        "approaching_max_percentage": 100.0 * approaching / count,
        "prompt_truncation_count": prompt_truncated,
        "target_truncation_count": target_truncated,
        "zero_target_count": zero_targets,
        "example_count": count,
        "test_evaluated": False,
    }


def generated_token_slice(generated: torch.Tensor, input_length: int) -> torch.Tensor:
    if generated.ndim != 2 or input_length < 0 or input_length > generated.shape[1]:
        raise ValueError("invalid generated sequence or prompt length")
    return generated[:, input_length:]


def special_token_audit(model: Any, tokenizer: Any) -> dict[str, object]:
    values = {
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
        "model_eos_token_id": getattr(model.config, "eos_token_id", None),
        "model_pad_token_id": getattr(model.config, "pad_token_id", None),
    }
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer EOS token is required for training diagnostics")
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer PAD token is required for generation diagnostics")
    model_eos = values["model_eos_token_id"]
    model_eos_values = model_eos if isinstance(model_eos, list | tuple) else [model_eos]
    if model_eos is not None and tokenizer.eos_token_id not in model_eos_values:
        raise ValueError("model and tokenizer EOS token IDs differ")
    return values


def gradient_audit(model: Any) -> dict[str, int | float]:
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora" in name.casefold()
    ]
    nonzero = [
        parameter
        for parameter in parameters
        if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
    ]
    norm = math.sqrt(
        sum(float(parameter.grad.detach().float().norm()) ** 2 for parameter in nonzero)
    )
    result: dict[str, int | float] = {
        "trainable_lora_tensors": len(parameters),
        "nonzero_gradient_tensors": len(nonzero),
        "aggregate_gradient_norm": norm,
    }
    if not parameters or not nonzero:
        raise ValueError("no trainable LoRA tensor received a nonzero gradient")
    return result


def parameter_digest(parameters: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    found = False
    for name, parameter in parameters:
        found = True
        value = parameter.detach().reshape(-1)[:1024].float().cpu()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    if not found:
        raise ValueError("parameter digest requires at least one tensor")
    return digest.hexdigest()


def learning_rate_audit(
    configured_lr: float,
    scheduler: str,
    warmup_ratio: float,
    total_steps: int,
    points: tuple[int, ...] = (0, 1, 5, 10, 25),
) -> dict[str, object]:
    warmup = int(total_steps * warmup_ratio)

    def factor(step: int) -> float:
        if warmup and step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        if scheduler == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return max(0.0, 1.0 - progress)

    return {
        "configured_learning_rate": configured_lr,
        "scheduler": scheduler,
        "warmup_steps": warmup,
        "learning_rate_by_step": {
            str(step): configured_lr * factor(step) for step in points
        },
    }
