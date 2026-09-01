"""Provider contract independent of HTTP and model frameworks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ProviderResult:
    raw_output: str
    category: str | None
    scores: dict[str, float] | None


class ClauseClassifierProvider(Protocol):
    name: str
    model_id: str
    provider_type: str
    is_mock: bool

    def is_ready(self) -> tuple[bool, str | None]: ...

    async def classify(self, text: str) -> ProviderResult: ...

    async def close(self) -> None: ...
