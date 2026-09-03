"""Offline-testable metrics and lightweight async HTTP load generation."""

from __future__ import annotations

import asyncio
import json
import math
import platform
import time
import urllib.request
from dataclasses import dataclass


def percentile(values: list[float], fraction: float) -> float:
    if not values or not 0 <= fraction <= 1:
        raise ValueError("percentile requires values and fraction in [0,1]")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


@dataclass(frozen=True, slots=True)
class Observation:
    latency_ms: float
    success: bool
    timeout: bool = False
    taxonomy_invalid: bool = False


def summarize(
    observations: list[Observation],
    runtime: float,
    *,
    provider: str,
    model: str,
    backend: str,
    is_mock: bool,
    concurrency: int,
    artifact_id: str | None = None,
    quantization: str | None = None,
    hardware: str | None = None,
) -> dict[str, object]:
    if not observations or runtime <= 0:
        raise ValueError("benchmark needs observations and positive runtime")
    latencies = [item.latency_ms for item in observations]
    count = len(observations)
    return {
        "label": "DEVELOPMENT BACKEND — NOT MODEL SERVING PERFORMANCE"
        if is_mock
        else "SERVING BENCHMARK",
        "provider": provider,
        "model": model,
        "backend": backend,
        "artifact_id": artifact_id,
        "quantization": quantization,
        "hardware": hardware,
        "is_mock": is_mock,
        "request_count": count,
        "concurrency": concurrency,
        "successful_responses": sum(x.success for x in observations),
        "failed_responses": sum(not x.success for x in observations),
        "timeouts": sum(x.timeout for x in observations),
        "taxonomy_invalid_outputs": sum(x.taxonomy_invalid for x in observations),
        "taxonomy_invalid_rate": sum(x.taxonomy_invalid for x in observations) / count,
        "failure_rate": sum(not x.success for x in observations) / count,
        "p50_latency_ms": percentile(latencies, 0.5),
        "p95_latency_ms": percentile(latencies, 0.95),
        "p99_latency_ms": percentile(latencies, 0.99),
        "mean_latency_ms": sum(latencies) / count,
        "throughput_requests_per_second": count / runtime,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.system(),
        },
    }


def _post(url: str, timeout: float) -> Observation:
    started = time.monotonic()
    request = urllib.request.Request(
        url + "/v1/classify",
        data=b'{"text":"This agreement is governed by Delaware law."}',
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
        success = response.status == 200 and isinstance(
            body.get("predicted_category"), str
        )
        return Observation(
            (time.monotonic() - started) * 1000,
            success,
            taxonomy_invalid=response.status == 200 and not success,
        )
    except TimeoutError:
        return Observation((time.monotonic() - started) * 1000, False, timeout=True)
    except (OSError, json.JSONDecodeError):
        return Observation((time.monotonic() - started) * 1000, False)


async def load_test(
    base_url: str, requests: int, concurrency: int, timeout: float, warmup: int
) -> list[Observation]:
    if min(requests, concurrency) <= 0 or warmup < 0:
        raise ValueError("invalid load configuration")
    for _ in range(warmup):
        await asyncio.to_thread(_post, base_url, timeout)
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one() -> Observation:
        async with semaphore:
            return await asyncio.to_thread(_post, base_url, timeout)

    return list(await asyncio.gather(*(run_one() for _ in range(requests))))
