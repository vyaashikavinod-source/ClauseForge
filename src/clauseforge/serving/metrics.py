"""Dependency-free process-local service metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class ServiceMetrics:
    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    timeout_count: int = 0
    provider_unavailable_count: int = 0
    taxonomy_invalid_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    latency_ms: list[float] = field(default_factory=list)

    def observe(self, status: int, latency_ms: float) -> None:
        self.request_count += 1
        self.success_count += int(status < 400)
        self.failure_count += int(status >= 400)
        self.latency_ms.append(latency_ms)

    def snapshot(self) -> dict[str, object]:
        value = asdict(self)
        observations = value.pop("latency_ms")
        assert isinstance(observations, list)
        value["latency_observation_count"] = len(observations)
        value["mean_latency_ms"] = (
            sum(observations) / len(observations) if observations else 0.0
        )
        return value
