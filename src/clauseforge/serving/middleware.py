"""Request identity and privacy-preserving structured access logs."""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from clauseforge.serving.metrics import ServiceMetrics
from clauseforge.serving.rate_limit import MemoryRateLimiter

LOGGER = logging.getLogger("clauseforge.serving")
REQUEST_ID_HEADER = "X-Request-ID"
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


async def request_context_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    supplied = request.headers.get(REQUEST_ID_HEADER, "")
    request_id = supplied if _SAFE_REQUEST_ID.fullmatch(supplied) else str(uuid.uuid4())
    request.state.request_id = request_id
    started = time.perf_counter()
    limiter = getattr(request.app.state, "rate_limiter", None)
    response: Response
    if isinstance(limiter, MemoryRateLimiter) and not limiter.allow(
        request.client.host if request.client else "unknown"
    ):
        response = JSONResponse(
            status_code=429,
            content={
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": "The request rate limit was exceeded.",
                    "request_id": request_id,
                }
            },
        )
    else:
        response = await call_next(request)
    latency_ms = (time.perf_counter() - started) * 1000
    response.headers[REQUEST_ID_HEADER] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    metrics = getattr(request.app.state, "metrics", None)
    if isinstance(metrics, ServiceMetrics):
        metrics.observe(response.status_code, latency_ms)
    LOGGER.info(
        "request_complete",
        extra={
            "request_id": request_id,
            "route": request.url.path,
            "method": request.method,
            "status": response.status_code,
            "latency_ms": round(latency_ms, 3),
            "content_length": request.headers.get("content-length"),
            "cache_hit": getattr(request.state, "cache_hit", None),
        },
    )
    return response
