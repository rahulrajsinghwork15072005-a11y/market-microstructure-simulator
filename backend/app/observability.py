"""Production observability: metrics, request context, rate limiting.

Dependency-free: exposes Prometheus text-format metrics and lightweight ASGI
middleware for request IDs, timing, security headers, and rate limiting.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict, deque

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("app")

START_TIME = time.time()


class Metrics:
    """Minimal thread-safe Prometheus-style registry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: dict[tuple[str, str, str], int] = defaultdict(int)
        self._dur_sum: dict[tuple[str, str], float] = defaultdict(float)
        self._dur_count: dict[tuple[str, str], int] = defaultdict(int)

    def observe(self, method: str, path: str, status: int, seconds: float) -> None:
        with self._lock:
            self._requests[(method, path, str(status))] += 1
            self._dur_sum[(method, path)] += seconds
            self._dur_count[(method, path)] += 1

    def render(self) -> str:
        lines = [
            "# HELP http_requests_total Total HTTP requests",
            "# TYPE http_requests_total counter",
        ]
        with self._lock:
            for (method, path, status), count in sorted(self._requests.items()):
                lines.append(
                    f'http_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}'
                )
            lines.append("# HELP http_request_duration_seconds Request duration summary")
            lines.append("# TYPE http_request_duration_seconds summary")
            for (method, path), total in sorted(self._dur_sum.items()):
                count = self._dur_count[(method, path)]
                lines.append(
                    f'http_request_duration_seconds_sum{{method="{method}",path="{path}"}} {total:.6f}'
                )
                lines.append(
                    f'http_request_duration_seconds_count{{method="{method}",path="{path}"}} {count}'
                )
        lines.append(f"process_uptime_seconds {time.time() - START_TIME:.1f}")
        return "\n".join(lines) + "\n"


metrics = Metrics()


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Adds request IDs, timing, security headers, structured logs, and metrics."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed = time.perf_counter() - start
            metrics.observe(request.method, request.url.path, 500, elapsed)
            logger.exception("unhandled error rid=%s path=%s", request_id, request.url.path)
            raise

        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        metrics.observe(request.method, path, response.status_code, elapsed)

        response.headers["x-request-id"] = request_id
        response.headers["x-response-time-ms"] = f"{elapsed * 1000:.2f}"
        response.headers.setdefault("x-content-type-options", "nosniff")
        response.headers.setdefault("x-frame-options", "DENY")
        response.headers.setdefault("referrer-policy", "no-referrer")
        logger.info(
            "%s %s -> %d %.1fms rid=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed * 1000,
            request_id,
        )
        return response


class RateLimiter:
    """Fixed-window in-memory rate limiter keyed by client."""

    def __init__(self, limit: int, window: float = 1.0) -> None:
        self.limit = limit
        self.window = window
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now = time.monotonic()
        cutoff = now - self.window
        with self._lock:
            bucket = self._hits[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False
            bucket.append(now)
            return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limiter: RateLimiter) -> None:
        super().__init__(app)
        self.limiter = limiter

    async def dispatch(self, request: Request, call_next):
        key = request.client.host if request.client else "anonymous"
        if not self.limiter.allow(key):
            return JSONResponse(
                status_code=429,
                content={"error": "rate_limited", "detail": "too many requests"},
            )
        return await call_next(request)


def uptime_seconds() -> float:
    return time.time() - START_TIME
