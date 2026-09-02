"""ClauseForge FastAPI application factory and default development app."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from clauseforge.cache.exact import ExactMemoryCache, cache_key
from clauseforge.config import Settings
from clauseforge.logging import configure_logging
from clauseforge.serving.constants import (
    CUAD_TAXONOMY,
    DISCLAIMER,
    TAXONOMY_VERSION,
)
from clauseforge.serving.dependencies import build_provider
from clauseforge.serving.errors import (
    InferenceTimeoutError,
    InputTooLargeError,
    InvalidModelOutputError,
    ProviderUnavailableError,
    ServingError,
)
from clauseforge.serving.metadata import build_metadata
from clauseforge.serving.metrics import ServiceMetrics
from clauseforge.serving.middleware import request_context_middleware
from clauseforge.serving.providers.base import ClauseClassifierProvider
from clauseforge.serving.rate_limit import MemoryRateLimiter
from clauseforge.serving.schemas import (
    ClassificationRequest,
    ClassificationResponse,
    ErrorResponse,
    HealthResponse,
    ProcessingMetadata,
    ReadyResponse,
)
from clauseforge.training.templates import PROMPT_TEMPLATE_VERSION

LOGGER = logging.getLogger("clauseforge.serving")


def _request_id(request: Request) -> str:
    return cast(str, getattr(request.state, "request_id", "unavailable"))


def _error_response(
    request: Request, status_code: int, code: str, message: str
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": _request_id(request),
            }
        },
    )


def create_app(
    settings: Settings | None = None,
    provider: ClauseClassifierProvider | None = None,
    taxonomy: tuple[str, ...] = CUAD_TAXONOMY,
) -> FastAPI:
    active_settings = settings or Settings.from_env()
    active_settings.validate()
    configure_logging(active_settings.log_level)
    active_provider = provider or build_provider(active_settings, taxonomy)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):  # type: ignore[no-untyped-def]
        yield
        await active_provider.close()

    application = FastAPI(
        title="ClauseForge API",
        version="1.0.0",
        description=(
            "Contract-clause classification infrastructure. The default provider "
            "is a deterministic development stub, not a trained legal model."
        ),
        lifespan=lifespan,
    )
    application.middleware("http")(request_context_middleware)
    application.state.settings = active_settings
    application.state.provider = active_provider
    application.state.taxonomy = taxonomy
    application.state.metrics = ServiceMetrics()
    application.state.rate_limiter = MemoryRateLimiter(
        active_settings.rate_limit_per_minute
    )
    application.state.exact_cache = ExactMemoryCache(
        max(1, active_settings.exact_cache_capacity)
    )

    @application.exception_handler(RequestValidationError)
    async def validation_handler(
        request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(
            request, 422, "request_validation_error", "The request is invalid."
        )

    @application.exception_handler(ServingError)
    async def serving_error_handler(
        request: Request, exc: ServingError
    ) -> JSONResponse:
        if isinstance(exc, InferenceTimeoutError):
            application.state.metrics.timeout_count += 1
        elif isinstance(exc, ProviderUnavailableError):
            application.state.metrics.provider_unavailable_count += 1
        elif isinstance(exc, InvalidModelOutputError):
            application.state.metrics.taxonomy_invalid_count += 1
        LOGGER.warning(
            "controlled_error",
            extra={
                "request_id": _request_id(request),
                "route": request.url.path,
                "provider": active_provider.name,
                "error_class": type(exc).__name__,
            },
        )
        return _error_response(request, exc.status_code, exc.code, exc.public_message)

    @application.exception_handler(Exception)
    async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception(
            "internal_error",
            extra={
                "request_id": _request_id(request),
                "route": request.url.path,
                "provider": active_provider.name,
                "error_class": type(exc).__name__,
            },
        )
        return _error_response(
            request, 500, "internal_error", "An internal processing error occurred."
        )

    @application.get(
        "/health",
        response_model=HealthResponse,
        summary="Liveness check",
        description="Reports whether the API process is alive.",
    )
    async def health() -> HealthResponse:
        return HealthResponse()

    @application.get("/version", summary="Safe application and build identity")
    async def version() -> dict[str, object]:
        return {
            **build_metadata(),
            "provider": active_provider.name,
            "backend": active_provider.provider_type,
        }

    @application.get("/metrics", summary="Internal process-local metrics")
    async def metrics() -> JSONResponse:
        if not active_settings.metrics_enabled:
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        return JSONResponse(content=application.state.metrics.snapshot())

    @application.get(
        "/ready",
        response_model=ReadyResponse,
        response_model_exclude_none=True,
        summary="Readiness check",
        description="Reports whether the configured classification provider is ready.",
    )
    async def ready() -> ReadyResponse | JSONResponse:
        is_ready, detail = active_provider.is_ready()
        response = ReadyResponse(
            status="ready" if is_ready else "unavailable",
            provider=active_provider.name,
            model_id=active_provider.model_id,
            detail=detail,
        )
        if is_ready:
            return response
        return JSONResponse(
            status_code=503, content=response.model_dump(exclude_none=True)
        )

    @application.post(
        "/v1/classify",
        response_model=ClassificationResponse,
        response_model_exclude_none=True,
        responses={
            413: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            504: {"model": ErrorResponse},
        },
        summary="Classify one contract clause",
        description="Returns one exact authoritative CUAD taxonomy category.",
    )
    async def classify(
        payload: ClassificationRequest, request: Request
    ) -> ClassificationResponse:
        if len(payload.text) > active_settings.max_input_characters:
            raise InputTooLargeError
        is_ready, _detail = active_provider.is_ready()
        if not is_ready:
            raise ProviderUnavailableError
        started = time.perf_counter()
        key = cache_key(
            payload.text,
            provider=active_provider.name,
            model=active_provider.model_id,
            model_revision="deployment-configured",
            adapter=(
                active_settings.adapter_path.name
                if active_settings.adapter_path is not None
                else None
            ),
            taxonomy_version=TAXONOMY_VERSION,
            prompt_version=PROMPT_TEMPLATE_VERSION,
            inference={
                "max_new_tokens": active_settings.max_new_tokens,
                "temperature": active_settings.temperature,
            },
        )
        cache_enabled = active_settings.exact_cache_capacity > 0
        result = application.state.exact_cache.get(key) if cache_enabled else None
        cache_hit = result is not None
        request.state.cache_hit = cache_hit
        if cache_hit:
            application.state.metrics.cache_hits += 1
        else:
            if cache_enabled:
                application.state.metrics.cache_misses += 1
            try:
                result = await asyncio.wait_for(
                    active_provider.classify(payload.text),
                    timeout=active_settings.inference_timeout_seconds,
                )
            except TimeoutError as exc:
                raise InferenceTimeoutError from exc
            if cache_enabled:
                application.state.exact_cache.put(key, result)
        assert result is not None
        if result.category is None or result.category not in taxonomy:
            raise InvalidModelOutputError
        latency_ms = (time.perf_counter() - started) * 1000
        return ClassificationResponse(
            request_id=_request_id(request),
            predicted_category=result.category,
            provider=active_provider.name,
            model_id=active_provider.model_id,
            taxonomy_version=TAXONOMY_VERSION,
            processing=ProcessingMetadata(
                provider_type=cast(Any, active_provider.provider_type),
                is_mock=active_provider.is_mock,
                character_count=len(payload.text),
                latency_ms=round(latency_ms, 3),
                score_availability=(
                    "available" if result.scores is not None else "unavailable"
                ),
                cache_hit=cache_hit,
            ),
            disclaimer=DISCLAIMER,
            scores=result.scores,
        )

    return application


app = create_app()
