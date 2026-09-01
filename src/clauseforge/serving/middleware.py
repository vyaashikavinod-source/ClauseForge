"""Request identity and privacy-preserving structured access logs."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response

LOGGER = logging.getLogger("clauseforge.serving")
REQUEST_ID_HEADER = "X-Request-ID"


async def request_context_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - started) * 1000
    response.headers[REQUEST_ID_HEADER] = request_id
    LOGGER.info(
        "request_complete",
        extra={
            "request_id": request_id,
            "route": request.url.path,
            "method": request.method,
            "status": response.status_code,
            "latency_ms": round(latency_ms, 3),
            "content_length": request.headers.get("content-length"),
        },
    )
    return response
