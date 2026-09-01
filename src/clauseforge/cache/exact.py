"""Version-isolated exact cache with deterministic keys and honest metrics."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol

from clauseforge.serving.providers.base import ProviderResult


def cache_key(
    text: str,
    *,
    provider: str,
    model: str,
    model_revision: str,
    adapter: str | None,
    taxonomy_version: str,
    prompt_version: str,
    inference: dict[str, object],
) -> str:
    value = {
        "text": " ".join(text.split()),
        "provider": provider,
        "model": model,
        "model_revision": model_revision,
        "adapter": adapter,
        "taxonomy_version": taxonomy_version,
        "prompt_version": prompt_version,
        "inference": inference,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class CacheMetrics:
    hits: int
    misses: int
    entries: int
    evictions: int

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class ExactMemoryCache:
    def __init__(self, capacity: int = 1024) -> None:
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        self._capacity = capacity
        self._values: OrderedDict[str, ProviderResult] = OrderedDict()
        self._hits = self._misses = self._evictions = 0

    def get(self, key: str) -> ProviderResult | None:
        value = self._values.get(key)
        if value is None:
            self._misses += 1
            return None
        self._hits += 1
        self._values.move_to_end(key)
        return value

    def put(self, key: str, value: ProviderResult) -> None:
        self._values[key] = value
        self._values.move_to_end(key)
        if len(self._values) > self._capacity:
            self._values.popitem(last=False)
            self._evictions += 1

    def metrics(self) -> CacheMetrics:
        return CacheMetrics(
            self._hits, self._misses, len(self._values), self._evictions
        )


class SemanticCache(Protocol):
    """Future opt-in interface; implementations must provide real embeddings."""

    async def lookup(self, text: str) -> ProviderResult | None: ...
    async def store(self, text: str, result: ProviderResult) -> None: ...
