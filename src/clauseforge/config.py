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
    gguf_path: Path | None = None
    vllm_base_url: str | None = None
    llamacpp_base_url: str | None = None
    device: str = "cpu"
    max_sequence_length: int = 1024
    max_input_characters: int = 10_000
    inference_timeout_seconds: float = 30.0
    max_new_tokens: int = 160
    temperature: float = 0.0

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
            model_provider=os.getenv(
                "CLAUSEFORGE_MODEL_BACKEND",
                os.getenv("CLAUSEFORGE_MODEL_PROVIDER", "mock"),
            ).lower(),
            model_path=_optional_path("CLAUSEFORGE_MODEL_PATH"),
            adapter_path=_optional_path("CLAUSEFORGE_ADAPTER_PATH"),
            tokenizer_path=_optional_path("CLAUSEFORGE_TOKENIZER_PATH"),
            gguf_path=_optional_path("CLAUSEFORGE_GGUF_PATH"),
            vllm_base_url=os.getenv("CLAUSEFORGE_VLLM_BASE_URL") or None,
            llamacpp_base_url=os.getenv("CLAUSEFORGE_LLAMACPP_BASE_URL") or None,
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
            max_new_tokens=int(os.getenv("CLAUSEFORGE_MAX_NEW_TOKENS", "160")),
            temperature=float(os.getenv("CLAUSEFORGE_TEMPERATURE", "0")),
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
        if self.model_provider not in {"mock", "transformer", "vllm", "llamacpp"}:
            raise ValueError("CLAUSEFORGE_MODEL_PROVIDER/MODEL_BACKEND is invalid")
        if self.max_input_characters < 3:
            raise ValueError("CLAUSEFORGE_MAX_INPUT_CHARACTERS must be at least 3")
        if self.max_sequence_length < 32:
            raise ValueError("CLAUSEFORGE_MAX_SEQUENCE_LENGTH must be at least 32")
        if self.inference_timeout_seconds <= 0:
            raise ValueError("CLAUSEFORGE_INFERENCE_TIMEOUT_SECONDS must be positive")
        if self.max_new_tokens <= 0 or not 0.0 <= self.temperature <= 2.0:
            raise ValueError("generation settings are invalid")


def _optional_path(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value) if value else None
