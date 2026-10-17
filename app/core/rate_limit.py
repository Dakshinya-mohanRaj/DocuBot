"""Minimal in-memory, per-IP token-bucket rate limiting.

Deliberately dependency-free and process-local: it protects a single demo
instance from casual abuse (scripted floods) and resets on restart. It is NOT a
distributed limiter — behind multiple instances each holds its own counters.

Limits are configurable; a value of 0 disables that limiter. Clients are keyed
on the left-most X-Forwarded-For entry when present (Render terminates TLS in
front of the app), falling back to the peer address.
"""

import logging
import os
import time
from collections import OrderedDict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.logging_config import log_event

logger = logging.getLogger("docubot.rate_limit")

_EXEMPT_PREFIXES = ("/health", "/ready", "/docs", "/redoc", "/openapi.json", "/static")


class TokenBucket:
    def __init__(self, per_minute: int, now: float | None = None) -> None:
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self.tokens = float(per_minute)
        self.updated = now if now is not None else time.monotonic()

    def allow(self, now: float) -> bool:
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class RateLimiter:
    """A per-key token bucket with bounded memory (LRU eviction)."""

    def __init__(self, per_minute: int, max_keys: int = 4096) -> None:
        self.per_minute = per_minute
        self.max_keys = max_keys
        self._buckets: OrderedDict[str, TokenBucket] = OrderedDict()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = TokenBucket(self.per_minute, now=now)
            self._buckets[key] = bucket
            if len(self._buckets) > self.max_keys:
                self._buckets.popitem(last=False)
        else:
            self._buckets.move_to_end(key)
        return bucket.allow(now)


def _client_ip(request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip() or "unknown"
    return request.client.host if request.client else "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rules: dict[str, RateLimiter]) -> None:
        super().__init__(app)
        self.rules = rules

    async def dispatch(self, request, call_next):
        path = request.url.path
        if any(path.startswith(prefix) for prefix in _EXEMPT_PREFIXES):
            return await call_next(request)

        limiter = next(
            (lim for prefix, lim in self.rules.items() if path.startswith(prefix)),
            None,
        )
        if limiter is None:
            return await call_next(request)

        key = _client_ip(request)
        if not limiter.allow(key):
            log_event(logger, "rate_limited", level=logging.WARNING, path=path)
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please slow down and try again shortly."},
                headers={"Retry-After": "60"},
            )
        return await call_next(request)


def build_rules() -> dict[str, RateLimiter]:
    """Create per-path limiters from env config; empty dict when all disabled."""
    chat = int(os.environ.get("DOCUBOT_RATE_LIMIT_CHAT", "20"))
    ingest = int(os.environ.get("DOCUBOT_RATE_LIMIT_INGEST", "10"))
    # Auth endpoints get their own, tighter limiter so credential-stuffing
    # attempts are throttled independently of normal chat traffic.
    auth = int(os.environ.get("DOCUBOT_RATE_LIMIT_AUTH", "10"))
    rules: dict[str, RateLimiter] = {}
    if chat > 0:
        rules["/chat"] = RateLimiter(chat)
    if ingest > 0:
        rules["/ingest"] = RateLimiter(ingest)
    if auth > 0:
        rules["/auth"] = RateLimiter(auth)
    return rules
