"""Smoke tests for the Phase 0 package foundation."""

from clauseforge import Settings


def test_package_import_and_secretless_configuration() -> None:
    settings = Settings.from_env()

    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert str(settings.data_dir) == "data"
    assert str(settings.model_dir) == "models"
