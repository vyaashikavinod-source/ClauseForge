from __future__ import annotations

import pytest

from clauseforge.config import Settings


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
