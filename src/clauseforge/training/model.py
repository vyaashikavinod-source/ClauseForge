"""Model-family-aware loading with a deliberately local smoke model."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
)

from clauseforge.training.config import ConfigurationError, ModelConfig

SUPPORTED_TARGETS: dict[str, tuple[str, ...]] = {
    "gpt2": ("c_attn",),
}
EXPERIMENTAL_TARGETS: dict[str, tuple[str, ...]] = {
    "llama": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mistral": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "qwen2": ("q_proj", "k_proj", "v_proj", "o_proj"),
}


def resolve_target_modules(family: str, explicit: tuple[str, ...]) -> tuple[str, ...]:
    if explicit:
        return explicit
    if family in SUPPORTED_TARGETS:
        return SUPPORTED_TARGETS[family]
    if family in EXPERIMENTAL_TARGETS:
        return EXPERIMENTAL_TARGETS[family]
    raise ConfigurationError(
        f"no reviewed LoRA targets for model family {family!r}; "
        "configure them explicitly"
    )


def require_qlora_runtime() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires a CUDA GPU; use quantization: none on CPU")
    if find_spec("bitsandbytes") is None:
        raise RuntimeError(
            "QLoRA requires the optional bitsandbytes package on a supported CUDA host"
        )


def build_4bit_config(precision: str) -> Any:
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=(
            torch.bfloat16 if precision == "bfloat16" else torch.float16
        ),
    )


def tiny_smoke_model(vocab_size: int) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=128,
        n_ctx=128,
        n_embd=32,
        n_layer=1,
        n_head=1,
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=0,
    )
    return GPT2LMHeadModel(config)


def load_production_model(config: ModelConfig) -> tuple[Any, Any]:
    if config.name == "local/tiny-gpt2":
        raise ConfigurationError(
            "the local smoke model is constructed by the smoke runner"
        )
    kwargs: dict[str, object] = {"revision": config.revision}
    if config.quantization == "4bit":
        require_qlora_runtime()
        kwargs["quantization_config"] = build_4bit_config(config.precision)
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name, revision=config.revision
    )
    model = AutoModelForCausalLM.from_pretrained(config.name, **kwargs)
    return model, tokenizer
