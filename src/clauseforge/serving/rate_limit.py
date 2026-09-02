"""Small process-local rate-limit boundary for development deployments."""

from __future__ import annotations

import time
from collections import defaultdict, deque


class MemoryRateLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, now: float | None = None) -> bool:
        if self.limit <= 0:
            return True
        current = time.monotonic() if now is None else now
        events = self._events[key]
        while events and current - events[0] >= self.window_seconds:
            events.popleft()
        if len(events) >= self.limit:
            return False
        events.append(current)
        return True
