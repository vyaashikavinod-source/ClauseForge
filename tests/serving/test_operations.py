from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi.testclient import TestClient

from clauseforge.config import Settings
from clauseforge.logging import JsonFormatter
from clauseforge.release.readiness import assess_readiness
from clauseforge.serving.app import create_app
from clauseforge.serving.metrics import ServiceMetrics
from clauseforge.serving.rate_limit import MemoryRateLimiter


def test_readiness_is_factually_blocked() -> None:
    report = assess_readiness(Settings(environment="test"))
    assert report.overall_status == "blocked"
    assert all(item.status == "ready" for item in report.infrastructure)
    blockers = {item.name for item in report.model_artifact if item.status == "blocked"}
    assert {
        "final_adapter_selected",
        "model_artifact",
        "quantization_executed",
    } <= blockers


def test_backend_environment_validation() -> None:
    assert Settings(model_provider="mock").backend_configuration_issues() == ()
    assert (
        len(Settings(model_provider="transformer").backend_configuration_issues()) == 2
    )
    assert len(Settings(model_provider="vllm").backend_configuration_issues()) == 2
    assert len(Settings(model_provider="llamacpp").backend_configuration_issues()) == 2


def test_request_ids_security_headers_version_and_metrics() -> None:
    settings = Settings(environment="test", metrics_enabled=True)
    with TestClient(create_app(settings)) as client:
        response = client.get("/version", headers={"X-Request-ID": "caller.safe-1"})
        bad = client.get("/health", headers={"X-Request-ID": "bad\nvalue"})
        client.get("/ready")
        metrics = client.get("/metrics")
    assert response.headers["X-Request-ID"] == "caller.safe-1"
    assert bad.headers["X-Request-ID"] != "bad\nvalue"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Cache-Control"] == "no-store"
    assert response.json()["taxonomy_version"]
    assert response.json()["prompt_version"]
    assert metrics.json()["request_count"] >= 3


def test_metrics_disabled_by_default() -> None:
    with TestClient(create_app(Settings(environment="test"))) as client:
        assert client.get("/metrics").status_code == 404


def test_exact_cache_is_initialized_and_counted() -> None:
    settings = Settings(
        environment="test", metrics_enabled=True, exact_cache_capacity=4
    )
    payload = {"text": "This agreement is governed by Delaware law."}
    with TestClient(create_app(settings)) as client:
        first = client.post("/v1/classify", json=payload)
        second = client.post("/v1/classify", json=payload)
        metrics = client.get("/metrics").json()
    assert first.json()["processing"]["cache_hit"] is False
    assert second.json()["processing"]["cache_hit"] is True
    assert metrics["cache_misses"] == 1 and metrics["cache_hits"] == 1


def test_metrics_and_rate_limit_are_deterministic() -> None:
    metrics = ServiceMetrics()
    metrics.observe(200, 10.0)
    metrics.observe(503, 30.0)
    assert metrics.snapshot()["mean_latency_ms"] == 20.0
    limiter = MemoryRateLimiter(2, 60)
    assert limiter.allow("client", 1.0)
    assert limiter.allow("client", 2.0)
    assert not limiter.allow("client", 3.0)
    assert limiter.allow("client", 62.0)


def test_provider_closed_once_on_shutdown() -> None:
    class Provider:
        name = "test"
        model_id = "test"
        provider_type = "mock"
        is_mock = True
        closes = 0

        def is_ready(self) -> tuple[bool, str | None]:
            return True, None

        async def classify(self, text: str):  # type: ignore[no-untyped-def]
            raise AssertionError(text)

        async def close(self) -> None:
            self.closes += 1

    provider = Provider()
    with TestClient(create_app(Settings(environment="test"), provider=provider)):
        pass
    assert provider.closes == 1


def test_structured_logging_redacts_sensitive_extras() -> None:
    record = logging.makeLogRecord(
        {
            "msg": "event",
            "levelno": logging.INFO,
            "levelname": "INFO",
            "token": "secret-value",
        }
    )
    value = json.loads(JsonFormatter().format(record))
    assert value["context"]["token"] == "[REDACTED]"
    assert "secret-value" not in json.dumps(value)


def test_container_files_exclude_sensitive_artifacts() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    ignored = Path(".dockerignore").read_text(encoding="utf-8")
    assert "USER clauseforge" in dockerfile and "HEALTHCHECK" in dockerfile
    for value in ("data", "checkpoints", "artifacts", ".env*", "*.gguf"):
        assert value in ignored


def test_strict_input_edge_cases() -> None:
    with TestClient(create_app(Settings(environment="test"))) as client:
        malformed = client.post(
            "/v1/classify", content="{", headers={"Content-Type": "application/json"}
        )
        extra = client.post(
            "/v1/classify", json={"text": "valid clause", "unexpected": True}
        )
        unicode_response = client.post(
            "/v1/classify", json={"text": "履行条款 governed by law"}
        )
    assert malformed.status_code == 422
    assert extra.status_code == 422
    assert unicode_response.status_code == 200
