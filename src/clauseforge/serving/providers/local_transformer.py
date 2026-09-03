"""Manifest-driven Qwen QLoRA inference with fail-closed startup."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from clauseforge.serving.errors import ProviderUnavailableError
from clauseforge.serving.providers.base import ProviderResult
from clauseforge.training.templates import render_prompt


class LocalTransformerProvider:
    name = "active-trained-candidate"
    provider_type = "transformer"
    is_mock = False

    def __init__(
        self,
        model_path: Path | None,
        adapter_path: Path | None,
        tokenizer_path: Path | None,
        device: str,
        max_sequence_length: int,
        target_representation: str = "canonical_question",
        target_representation_version: str = "cuad-canonical-question-v1",
        prompt_template_version: str = "cuad-classification-v1",
        base_revision: str | None = None,
        max_new_tokens: int = 160,
        artifact_id: str | None = None,
        checkpoint_step: int | None = None,
        candidate_status: str | None = None,
    ) -> None:
        self.model_id = str(model_path) if model_path else "unconfigured"
        self.artifact_id = artifact_id
        self.checkpoint_step = checkpoint_step
        self.candidate_status = candidate_status
        self.base_model = str(model_path) if model_path else None
        self._model_path = model_path
        self._adapter_path = adapter_path
        self._tokenizer_path = tokenizer_path or model_path
        self._device = device
        self._max_sequence_length = max_sequence_length
        self._prompt_template_version = prompt_template_version
        self._base_revision = base_revision
        self._max_new_tokens = max_new_tokens
        self.target_representation = target_representation
        self.target_representation_version = target_representation_version
        self._model: Any = None
        self._tokenizer: Any = None
        self._load_error: str | None = None
        self._load_if_supported()

    def _load_if_supported(self) -> None:
        if self._device != "cuda":
            self._load_error = (
                "real Qwen adapter serving requires CLAUSEFORGE_DEVICE=cuda"
            )
            return
        if self._model_path is None or self._adapter_path is None:
            self._load_error = "required real model artifacts are not configured"
            return
        if not self._adapter_path.exists():
            self._load_error = "configured adapter is unavailable"
            return
        try:
            import torch
            from peft import PeftModel
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )

            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            source = str(self._model_path)
            tokenizer_source = str(self._tokenizer_path or self._model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source, revision=self._base_revision
            )
            base = AutoModelForCausalLM.from_pretrained(
                source,
                revision=self._base_revision,
                quantization_config=quantization,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self._model = PeftModel.from_pretrained(base, str(self._adapter_path))
            self._model.eval()
        except Exception as exc:  # optional GPU stack is reported through readiness
            self._load_error = f"real adapter load failed ({type(exc).__name__})"
            self._model = None
            self._tokenizer = None

    def is_ready(self) -> tuple[bool, str | None]:
        if self._model is None or self._tokenizer is None:
            return False, self._load_error or "real adapter is not loaded"
        return True, None

    def _classify_sync(self, text: str) -> ProviderResult:
        if self._model is None or self._tokenizer is None:
            raise ProviderUnavailableError
        prompt = render_prompt(text, self._prompt_template_version)
        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_sequence_length,
        )
        device = next(self._model.parameters()).device
        encoded = {name: value.to(device) for name, value in encoded.items()}
        input_length = int(encoded["input_ids"].shape[-1])
        generated = self._model.generate(
            **encoded,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        raw = self._tokenizer.decode(
            generated[0][input_length:], skip_special_tokens=True
        )
        return ProviderResult(raw, raw.strip(), None)

    async def classify(self, text: str) -> ProviderResult:
        return await asyncio.to_thread(self._classify_sync, text)

    async def close(self) -> None:
        self._model = None
        self._tokenizer = None
