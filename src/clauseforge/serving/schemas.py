"""Versioned, strict HTTP schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=3, max_length=100_000)

    @field_validator("text")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain non-whitespace characters")
        return value


class ProcessingMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider_type: Literal["mock", "transformer"]
    is_mock: bool
    character_count: int
    latency_ms: float
    score_availability: Literal["available", "unavailable"]


class ClassificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_version: Literal["v1"] = "v1"
    request_id: str
    predicted_category: str
    status: Literal["success"] = "success"
    provider: str
    model_id: str
    taxonomy_version: str
    processing: ProcessingMetadata
    disclaimer: str
    scores: dict[str, float] | None = None


class HealthResponse(BaseModel):
    status: Literal["alive"] = "alive"


class ReadyResponse(BaseModel):
    status: Literal["ready", "unavailable"]
    provider: str
    model_id: str
    detail: str | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
