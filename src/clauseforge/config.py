"""Environment-based application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_ALLOWED_ENVIRONMENTS = frozenset({"development", "test", "staging", "production"})


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated settings that require no secrets to initialize."""

    environment: str = "development"
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    host: str = "127.0.0.1"
    port: int = 8000
    model_provider: str = "mock"
    model_path: Path | None = None
    adapter_path: Path | None = None
    tokenizer_path: Path | None = None
    device: str = "cpu"
    max_sequence_length: int = 1024
    max_input_characters: int = 10_000
    inference_timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from `CLAUSEFORGE_*` environment variables."""
        settings = cls(
            environment=os.getenv("CLAUSEFORGE_ENVIRONMENT", "development").lower(),
            log_level=os.getenv("CLAUSEFORGE_LOG_LEVEL", "INFO").upper(),
            data_dir=Path(os.getenv("CLAUSEFORGE_DATA_DIR", "data")),
            model_dir=Path(os.getenv("CLAUSEFORGE_MODEL_DIR", "models")),
            host=os.getenv("CLAUSEFORGE_HOST", "127.0.0.1"),
            port=int(os.getenv("CLAUSEFORGE_PORT", "8000")),
            model_provider=os.getenv("CLAUSEFORGE_MODEL_PROVIDER", "mock").lower(),
            model_path=_optional_path("CLAUSEFORGE_MODEL_PATH"),
            adapter_path=_optional_path("CLAUSEFORGE_ADAPTER_PATH"),
            tokenizer_path=_optional_path("CLAUSEFORGE_TOKENIZER_PATH"),
            device=os.getenv("CLAUSEFORGE_DEVICE", "cpu").lower(),
            max_sequence_length=int(
                os.getenv("CLAUSEFORGE_MAX_SEQUENCE_LENGTH", "1024")
            ),
            max_input_characters=int(
                os.getenv("CLAUSEFORGE_MAX_INPUT_CHARACTERS", "10000")
            ),
            inference_timeout_seconds=float(
                os.getenv("CLAUSEFORGE_INFERENCE_TIMEOUT_SECONDS", "30")
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Raise `ValueError` when a setting is unsupported."""
        if self.environment not in _ALLOWED_ENVIRONMENTS:
            allowed = ", ".join(sorted(_ALLOWED_ENVIRONMENTS))
            raise ValueError(f"CLAUSEFORGE_ENVIRONMENT must be one of: {allowed}")

        valid_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        if self.log_level not in valid_levels:
            allowed = ", ".join(sorted(valid_levels))
            raise ValueError(f"CLAUSEFORGE_LOG_LEVEL must be one of: {allowed}")
        if not 1 <= self.port <= 65535:
            raise ValueError("CLAUSEFORGE_PORT must be between 1 and 65535")
        if self.model_provider not in {"mock", "transformer"}:
            raise ValueError("CLAUSEFORGE_MODEL_PROVIDER must be mock or transformer")
        if self.max_input_characters < 3:
            raise ValueError("CLAUSEFORGE_MAX_INPUT_CHARACTERS must be at least 3")
        if self.max_sequence_length < 32:
            raise ValueError("CLAUSEFORGE_MAX_SEQUENCE_LENGTH must be at least 32")
        if self.inference_timeout_seconds <= 0:
            raise ValueError("CLAUSEFORGE_INFERENCE_TIMEOUT_SECONDS must be positive")


def _optional_path(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value) if value else None
