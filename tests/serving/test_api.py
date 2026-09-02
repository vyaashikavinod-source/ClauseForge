from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clauseforge.config import Settings
from clauseforge.serving.app import create_app
from clauseforge.serving.constants import CUAD_TAXONOMY, DISCLAIMER
from clauseforge.serving.providers.base import ProviderResult
from clauseforge.serving.providers.local_transformer import LocalTransformerProvider
from clauseforge.serving.providers.mock import MockDevelopmentProvider


def _settings(**changes: object) -> Settings:
    values: dict[str, object] = {
        "environment": "test",
        "max_input_characters": 100,
        "inference_timeout_seconds": 0.05,
    }
    values.update(changes)
    return Settings(**values)  # type: ignore[arg-type]


def _client(provider: object | None = None) -> TestClient:
    return TestClient(
        create_app(_settings(), provider=provider),  # type: ignore[arg-type]
        raise_server_exceptions=False,
    )


def test_health_and_mock_readiness() -> None:
    with _client() as client:
        health = client.get("/health")
        ready = client.get("/ready")
    assert health.status_code == 200
    assert health.json() == {"status": "alive"}
    assert ready.status_code == 200
    assert ready.json()["provider"] == "mock-development"


def test_valid_classification_is_deterministic_and_honest() -> None:
    payload = {"text": "This agreement is governed by the laws of Delaware."}
    with _client() as client:
        first = client.post("/v1/classify", json=payload)
        second = client.post("/v1/classify", json=payload)
    assert first.status_code == 200
    body = first.json()
    assert body["predicted_category"] == second.json()["predicted_category"]
    assert '"Governing Law"' in body["predicted_category"]
    assert body["request_id"]
    assert body["disclaimer"] == DISCLAIMER
    assert body["processing"]["is_mock"] is True
    assert body["processing"]["score_availability"] == "unavailable"
    assert "scores" not in body


@pytest.mark.parametrize("text", ["", "  "])
def test_empty_text_rejected_with_structured_error(text: str) -> None:
    with _client() as client:
        response = client.post("/v1/classify", json={"text": text})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request_validation_error"
    assert response.json()["error"]["request_id"]


def test_oversized_text_is_not_truncated() -> None:
    with _client() as client:
        response = client.post("/v1/classify", json={"text": "x" * 101})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "input_too_large"


class UnavailableProvider:
    name = "unavailable"
    model_id = "missing"
    provider_type = "transformer"
    is_mock = False

    def is_ready(self) -> tuple[bool, str | None]:
        return False, "not provisioned"

    async def classify(self, text: str) -> ProviderResult:
        raise AssertionError("classify must not be called")

    async def close(self) -> None:
        return None


class InvalidProvider(UnavailableProvider):
    name = "invalid-output"

    def is_ready(self) -> tuple[bool, str | None]:
        return True, None

    async def classify(self, text: str) -> ProviderResult:
        return ProviderResult("hallucinated label", "hallucinated label", None)


class SlowProvider(InvalidProvider):
    name = "slow"

    async def classify(self, text: str) -> ProviderResult:
        await asyncio.sleep(0.2)
        return ProviderResult(CUAD_TAXONOMY[0], CUAD_TAXONOMY[0], None)


class BrokenProvider(InvalidProvider):
    name = "broken"

    async def classify(self, text: str) -> ProviderResult:
        raise RuntimeError("secret path C:/private/model")


class CategoryIdProvider(InvalidProvider):
    target_representation = "category_id"
    target_representation_version = "cuad-category-id-v1"

    def __init__(self, output: str) -> None:
        self.output = output

    async def classify(self, text: str) -> ProviderResult:
        return ProviderResult(self.output, self.output, None)


def test_unavailable_provider_readiness_and_classification() -> None:
    with _client(UnavailableProvider()) as client:
        ready = client.get("/ready")
        classified = client.post("/v1/classify", json={"text": "valid clause"})
    assert ready.status_code == 503
    assert ready.json()["status"] == "unavailable"
    assert classified.status_code == 503
    assert classified.json()["error"]["code"] == "provider_unavailable"


def test_unknown_model_label_is_never_success() -> None:
    with _client(InvalidProvider()) as client:
        response = client.post("/v1/classify", json={"text": "valid clause"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_model_output"
    assert "hallucinated" not in response.text


def test_category_id_provider_resolves_to_authoritative_taxonomy() -> None:
    with _client(CategoryIdProvider("renewal_term")) as client:
        response = client.post("/v1/classify", json={"text": "valid clause"})
    assert response.status_code == 200
    assert '"Renewal Term"' in response.json()["predicted_category"]
    assert response.json()["processing"]["target_representation"] == "category_id"


@pytest.mark.parametrize(
    "output",
    [
        "unknown_category",
        "Renewal Term",
        "renewal term",
        "renewal_ term",
        "The category is renewal_term",
    ],
)
def test_category_id_provider_rejects_non_exact_outputs(output: str) -> None:
    with _client(CategoryIdProvider(output)) as client:
        response = client.post("/v1/classify", json={"text": "valid clause"})
    assert response.status_code == 422


def test_category_id_provider_allows_surrounding_whitespace_only() -> None:
    with _client(CategoryIdProvider("renewal_term ")) as client:
        response = client.post("/v1/classify", json={"text": "valid clause"})
    assert response.status_code == 200


def test_timeout_is_structured() -> None:
    with _client(SlowProvider()) as client:
        response = client.post("/v1/classify", json={"text": "valid clause"})
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "inference_timeout"


def test_internal_exception_is_sanitized() -> None:
    with _client(BrokenProvider()) as client:
        response = client.post("/v1/classify", json={"text": "valid clause"})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "private" not in response.text
    assert "traceback" not in response.text.casefold()


def test_request_text_is_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    secret_clause = "governed by laws UNIQUE_CLAUSE_CONTENT"
    caplog.set_level(logging.INFO)
    with _client() as client:
        response = client.post("/v1/classify", json={"text": secret_clause})
    assert response.status_code == 200
    assert secret_clause not in caplog.text
    assert "UNIQUE_CLAUSE_CONTENT" not in caplog.text


def test_local_transformer_missing_weights_is_not_ready(tmp_path: Path) -> None:
    provider = LocalTransformerProvider(None, tmp_path, None, "cpu", 1024)
    ready, detail = provider.is_ready()
    assert ready is False
    assert detail == "required local model artifacts are not configured"


def test_mock_categories_are_authoritative() -> None:
    provider = MockDevelopmentProvider(CUAD_TAXONOMY)
    assert len(CUAD_TAXONOMY) == 41
    assert all(value in CUAD_TAXONOMY for value in provider._by_name.values())
