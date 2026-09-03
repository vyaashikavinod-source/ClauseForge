from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clauseforge.config import Settings
from clauseforge.serving.app import create_app


def test_serving_configuration_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUSEFORGE_PORT", "9000")
    monkeypatch.setenv("CLAUSEFORGE_MODEL_PROVIDER", "transformer")
    monkeypatch.setenv("CLAUSEFORGE_MAX_INPUT_CHARACTERS", "2048")
    monkeypatch.setenv("CLAUSEFORGE_INFERENCE_TIMEOUT_SECONDS", "2.5")
    settings = Settings.from_env()
    assert settings.port == 9000
    assert settings.model_provider == "transformer"
    assert settings.max_input_characters == 2048
    assert settings.inference_timeout_seconds == 2.5


def test_invalid_provider_configuration_rejected() -> None:
    with pytest.raises(ValueError, match="MODEL_PROVIDER"):
        Settings(model_provider="automatic").validate()


def test_invalid_real_manifest_starts_unready_without_mock_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CLAUSEFORGE_MODEL_BACKEND", "real")
    monkeypatch.setenv("MODEL_ARTIFACT_MANIFEST", str(tmp_path / "missing.json"))
    settings = Settings.from_env()
    assert settings.artifact_validation_error is not None
    with TestClient(create_app(settings)) as client:
        ready = client.get("/ready")
    assert ready.status_code == 503
    assert ready.json()["model_artifact_configured"] is True
    assert ready.json()["model_artifact_valid"] is False
    assert ready.json()["provider"] != "mock-development"
