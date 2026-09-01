from __future__ import annotations

import json
from pathlib import Path

from clauseforge.cache.exact import ExactMemoryCache, cache_key
from clauseforge.config import Settings
from clauseforge.serving.benchmark import Observation, percentile, summarize
from clauseforge.serving.dependencies import build_provider
from clauseforge.serving.providers.base import ProviderResult
from clauseforge.serving.providers.llamacpp import LlamaCppProvider
from clauseforge.serving.providers.vllm import VllmProvider


def test_backend_selection_and_missing_services(tmp_path: Path) -> None:
    vllm = build_provider(Settings(model_provider="vllm"), ("label",))
    llama = build_provider(
        Settings(model_provider="llamacpp", gguf_path=tmp_path / "missing.gguf"),
        ("label",),
    )
    assert isinstance(vllm, VllmProvider) and not vllm.is_ready()[0]
    assert isinstance(llama, LlamaCppProvider) and not llama.is_ready()[0]
    assert vllm.metadata()["scores_available"] is False


def test_cache_determinism_invalidation_and_metrics() -> None:
    def key_for(*, revision: str = "a", taxonomy: str = "v1") -> str:
        return cache_key(
            "a clause",
            provider="vllm",
            model="qwen",
            model_revision=revision,
            adapter="x",
            taxonomy_version=taxonomy,
            prompt_version="p1",
            inference={"temperature": 0},
        )

    key = cache_key(
        "a  clause",
        provider="vllm",
        model="qwen",
        model_revision="a",
        adapter="x",
        taxonomy_version="v1",
        prompt_version="p1",
        inference={"temperature": 0},
    )
    assert key == key_for()
    assert key != key_for(revision="b")
    assert key != key_for(taxonomy="v2")
    cache = ExactMemoryCache(1)
    assert cache.get(key) is None
    cache.put(key, ProviderResult("x", "x", None))
    assert cache.get(key) is not None
    cache.put("other", ProviderResult("y", "y", None))
    assert cache.metrics().evictions == 1 and cache.metrics().hit_rate == 0.5


def test_benchmark_math_and_serialization() -> None:
    assert percentile([1, 2, 3, 4, 100], 0.5) == 3
    result = summarize(
        [Observation(10, True), Observation(20, False, timeout=True)],
        1.0,
        provider="mock",
        model="dev",
        backend="mock",
        is_mock=True,
        concurrency=2,
    )
    assert result["timeouts"] == 1 and result["mean_latency_ms"] == 15
    assert "NOT MODEL" in str(result["label"])
    json.dumps(result)
