"""Optional llama.cpp server provider for a provisioned GGUF artifact."""

from __future__ import annotations

import asyncio
import json
import urllib.request
from pathlib import Path

from clauseforge.serving.errors import ProviderUnavailableError
from clauseforge.serving.providers.base import ProviderResult
from clauseforge.training.templates import render_prompt


class LlamaCppProvider:
    name = "llama.cpp"
    provider_type = "llamacpp"
    is_mock = False

    def __init__(
        self,
        gguf: Path | None,
        base_url: str | None,
        taxonomy: tuple[str, ...],
        max_new_tokens: int,
        temperature: float,
        timeout: float,
        target_representation: str = "canonical_question",
        target_representation_version: str = "cuad-canonical-question-v1",
        prompt_template_version: str = "cuad-classification-v1",
    ) -> None:
        self.model_id = gguf.name if gguf else "unconfigured"
        self._gguf = gguf
        self._url = base_url.rstrip("/") if base_url else None
        self._taxonomy = taxonomy
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._timeout = timeout
        self.target_representation = target_representation
        self.target_representation_version = target_representation_version
        self._prompt_template_version = prompt_template_version

    def is_ready(self) -> tuple[bool, str | None]:
        if self._gguf is None or not self._gguf.is_file():
            return False, "GGUF artifact is not configured"
        if not self._url:
            return False, "llama.cpp server URL is not configured"
        try:
            with urllib.request.urlopen(
                self._url + "/health", timeout=min(self._timeout, 2.0)
            ) as response:
                if response.status != 200:
                    return False, "llama.cpp health check failed"
        except OSError:
            return False, "llama.cpp service is unavailable"
        return True, None

    def _complete(self, text: str) -> str:
        if self._url is None:
            raise ProviderUnavailableError
        data = json.dumps(
            {
                "prompt": render_prompt(text, self._prompt_template_version),
                "n_predict": self._max_new_tokens,
                "temperature": self._temperature,
            }
        ).encode()
        request = urllib.request.Request(
            self._url + "/completion",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                value = json.loads(response.read())
            return str(value["content"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ProviderUnavailableError from exc

    async def classify(self, text: str) -> ProviderResult:
        raw = await asyncio.to_thread(self._complete, text)
        category = raw.strip()
        if self.target_representation == "category_id":
            return ProviderResult(raw, category, None)
        return ProviderResult(
            raw, category if category in self._taxonomy else None, None
        )

    async def close(self) -> None:
        return None

    def metadata(self) -> dict[str, object]:
        return {
            "provider": self.name,
            "backend": self.provider_type,
            "model": self.model_id,
            "scores_available": False,
            "target_representation": self.target_representation,
            "target_representation_version": self.target_representation_version,
        }
