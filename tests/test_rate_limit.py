"""Rate limiting: token bucket behaviour and middleware wiring."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.rate_limit import RateLimiter, TokenBucket, build_rules


def test_token_bucket_allows_capacity_then_denies():
    bucket = TokenBucket(per_minute=2, now=0.0)
    assert bucket.allow(now=0.0) is True
    assert bucket.allow(now=0.0) is True
    assert bucket.allow(now=0.0) is False
    # After enough time, a token refills (2/min => 1 token per 30s).
    assert bucket.allow(now=31.0) is True


def test_rate_limiter_keys_are_independent():
    limiter = RateLimiter(per_minute=1)
    assert limiter.allow("a") is True
    assert limiter.allow("b") is True
    assert limiter.allow("a") is False


def test_rate_limiter_bounds_memory():
    limiter = RateLimiter(per_minute=5, max_keys=3)
    for i in range(10):
        limiter.allow(f"ip-{i}")
    assert len(limiter._buckets) <= 3


def test_build_rules_respects_disabled_limits(monkeypatch):
    monkeypatch.setenv("DOCUBOT_RATE_LIMIT_CHAT", "0")
    monkeypatch.setenv("DOCUBOT_RATE_LIMIT_INGEST", "0")
    assert build_rules() == {}

    monkeypatch.setenv("DOCUBOT_RATE_LIMIT_CHAT", "7")
    monkeypatch.setenv("DOCUBOT_RATE_LIMIT_INGEST", "3")
    rules = build_rules()
    assert rules["/chat"].per_minute == 7
    assert rules["/ingest"].per_minute == 3


def test_middleware_returns_429_and_exempts_health():
    from app.core.rate_limit import RateLimitMiddleware

    app = FastAPI()

    @app.get("/chat")
    def chat():
        return {"ok": True}

    @app.get("/health")
    def health():
        return {"ok": True}

    app.add_middleware(RateLimitMiddleware, rules={"/chat": RateLimiter(1)})

    with TestClient(app) as client:
        assert client.get("/chat").status_code == 200
        limited = client.get("/chat")
        assert limited.status_code == 429
        assert limited.headers.get("retry-after") == "60"
        # Health stays open even though the limiter is exhausted.
        for _ in range(5):
            assert client.get("/health").status_code == 200
