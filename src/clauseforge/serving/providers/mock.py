"""Deterministic development stub; never represented as a trained model."""

from __future__ import annotations

from clauseforge.serving.providers.base import ProviderResult


class MockDevelopmentProvider:
    name = "mock-development"
    model_id = "deterministic-fixture-rules-v1"
    provider_type = "mock"
    is_mock = True

    def __init__(self, taxonomy: tuple[str, ...]) -> None:
        self._taxonomy = taxonomy
        self._by_name = {
            name: next(label for label in taxonomy if f'"{name}"' in label)
            for name in ("Governing Law", "Expiration Date", "Parties")
        }

    def is_ready(self) -> tuple[bool, str | None]:
        return True, None

    async def classify(self, text: str) -> ProviderResult:
        lowered = text.casefold()
        category = self._by_name["Parties"]
        if "governed by" in lowered or "governing law" in lowered:
            category = self._by_name["Governing Law"]
        elif "expire" in lowered or "expiration" in lowered:
            category = self._by_name["Expiration Date"]
        return ProviderResult(category, category, None)

    async def close(self) -> None:
        return None
