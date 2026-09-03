"""Explicit provider construction; no implicit fallback behavior."""

from __future__ import annotations

from clauseforge.config import Settings
from clauseforge.serving.providers.base import ClauseClassifierProvider
from clauseforge.serving.providers.llamacpp import LlamaCppProvider
from clauseforge.serving.providers.local_transformer import LocalTransformerProvider
from clauseforge.serving.providers.mock import MockDevelopmentProvider
from clauseforge.serving.providers.vllm import VllmProvider


def build_provider(
    settings: Settings, taxonomy: tuple[str, ...]
) -> ClauseClassifierProvider:
    if settings.model_provider == "mock":
        return MockDevelopmentProvider(taxonomy)
    if settings.model_provider == "vllm":
        return VllmProvider(
            settings.model_path,
            settings.vllm_base_url,
            taxonomy,
            settings.max_new_tokens,
            settings.temperature,
            settings.inference_timeout_seconds,
            settings.target_representation,
            settings.target_representation_version,
            settings.prompt_template_version,
        )
    if settings.model_provider == "llamacpp":
        return LlamaCppProvider(
            settings.gguf_path,
            settings.llamacpp_base_url,
            taxonomy,
            settings.max_new_tokens,
            settings.temperature,
            settings.inference_timeout_seconds,
            settings.target_representation,
            settings.target_representation_version,
            settings.prompt_template_version,
        )
    return LocalTransformerProvider(
        settings.model_path,
        settings.adapter_path,
        settings.tokenizer_path,
        settings.device,
        settings.max_sequence_length,
        settings.target_representation,
        settings.target_representation_version,
        settings.prompt_template_version,
        settings.base_revision,
        settings.max_new_tokens,
        settings.artifact_id,
        settings.checkpoint_step,
        settings.candidate_status,
    )
