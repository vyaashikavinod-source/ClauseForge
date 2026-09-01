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

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from `CLAUSEFORGE_*` environment variables."""
        settings = cls(
            environment=os.getenv("CLAUSEFORGE_ENVIRONMENT", "development").lower(),
            log_level=os.getenv("CLAUSEFORGE_LOG_LEVEL", "INFO").upper(),
            data_dir=Path(os.getenv("CLAUSEFORGE_DATA_DIR", "data")),
            model_dir=Path(os.getenv("CLAUSEFORGE_MODEL_DIR", "models")),
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
