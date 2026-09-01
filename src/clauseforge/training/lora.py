"""PEFT adapter construction with explicit, reviewed module targets."""

from __future__ import annotations

from typing import Any

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from clauseforge.training.config import LoraSettings, ModelConfig
from clauseforge.training.model import require_qlora_runtime, resolve_target_modules


def build_lora_config(model: ModelConfig, settings: LoraSettings) -> LoraConfig:
    return LoraConfig(
        r=settings.rank,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=list(
            resolve_target_modules(model.family, settings.target_modules)
        ),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def attach_lora(model_instance: Any, model: ModelConfig, settings: LoraSettings) -> Any:
    if model.quantization == "4bit":
        require_qlora_runtime()
        model_instance = prepare_model_for_kbit_training(model_instance)
    return get_peft_model(model_instance, build_lora_config(model, settings))
