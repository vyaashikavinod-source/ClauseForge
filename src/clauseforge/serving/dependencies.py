"""Explicit provider construction; no implicit fallback behavior."""

from __future__ import annotations

from clauseforge.config import Settings
from clauseforge.serving.providers.base import ClauseClassifierProvider
from clauseforge.serving.providers.local_transformer import LocalTransformerProvider
from clauseforge.serving.providers.mock import MockDevelopmentProvider


def build_provider(
    settings: Settings, taxonomy: tuple[str, ...]
) -> ClauseClassifierProvider:
    if settings.model_provider == "mock":
        return MockDevelopmentProvider(taxonomy)
    return LocalTransformerProvider(
        settings.model_path,
        settings.adapter_path,
        settings.tokenizer_path,
        settings.device,
        settings.max_sequence_length,
    )
