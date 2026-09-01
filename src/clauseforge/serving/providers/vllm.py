"""Optional vLLM OpenAI-compatible HTTP provider."""

from __future__ import annotations

import asyncio
import json
import urllib.request
from pathlib import Path

from clauseforge.serving.errors import ProviderUnavailableError
from clauseforge.serving.providers.base import ProviderResult
from clauseforge.training.templates import render_prompt


class VllmProvider:
    name = "vllm"
    provider_type = "vllm"
    is_mock = False

    def __init__(
        self,
        model: Path | None,
        base_url: str | None,
        taxonomy: tuple[str, ...],
        max_new_tokens: int,
        temperature: float,
        timeout: float,
    ) -> None:
        self.model_id = model.name if model else "unconfigured"
        self._url = base_url.rstrip("/") if base_url else None
        self._taxonomy = taxonomy
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._timeout = timeout

    def is_ready(self) -> tuple[bool, str | None]:
        if not self._url:
            return (
                False,
                "vLLM base URL is not configured; optional service unavailable",
            )
        try:
            with urllib.request.urlopen(
                self._url + "/health", timeout=min(self._timeout, 2.0)
            ) as response:
                if response.status != 200:
                    return False, "vLLM health check failed"
        except OSError:
            return False, "vLLM service is unavailable"
        return True, None

    def _complete(self, text: str) -> str:
        if self._url is None:
            raise ProviderUnavailableError
        data = json.dumps(
            {
                "model": self.model_id,
                "prompt": render_prompt(text),
                "max_tokens": self._max_new_tokens,
                "temperature": self._temperature,
            }
        ).encode()
        request = urllib.request.Request(
            self._url + "/v1/completions",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                value = json.loads(response.read())
            return str(value["choices"][0]["text"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ProviderUnavailableError from exc

    async def classify(self, text: str) -> ProviderResult:
        raw = await asyncio.to_thread(self._complete, text)
        category = raw.strip()
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
        }
