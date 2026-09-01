"""Readiness-safe skeleton for a future local Phase 3A transformer adapter."""

from __future__ import annotations

from pathlib import Path

from clauseforge.serving.errors import ProviderUnavailableError
from clauseforge.serving.providers.base import ProviderResult


class LocalTransformerProvider:
    name = "local-transformer"
    provider_type = "transformer"
    is_mock = False

    def __init__(
        self,
        model_path: Path | None,
        adapter_path: Path | None,
        tokenizer_path: Path | None,
        device: str,
        max_sequence_length: int,
    ) -> None:
        self.model_id = model_path.name if model_path else "unconfigured"
        self._paths = (model_path, adapter_path, tokenizer_path)
        self._device = device
        self._max_sequence_length = max_sequence_length

    def is_ready(self) -> tuple[bool, str | None]:
        if any(path is None or not path.exists() for path in self._paths):
            return False, "required local model artifacts are not configured"
        return False, "transformer loading is pending a real trained adapter"

    async def classify(self, text: str) -> ProviderResult:
        raise ProviderUnavailableError

    async def close(self) -> None:
        return None
