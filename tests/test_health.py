"""Health and readiness endpoint behaviour.

These must stay cheap: /health is the Render liveness probe and must never load
the embedding model or call the LLM; /ready inspects dependencies without any
network LLM request and never leaks secrets.
"""

import json
import os

from app.core import llm as llm_module


def test_health_is_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_never_touches_the_llm(client, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("/health must not resolve or call the LLM")

    monkeypatch.setattr(llm_module, "resolve_model", explode)
    monkeypatch.setattr(llm_module, "get_client", explode)
    assert client.get("/health").status_code == 200


def test_ready_reports_dependencies_without_secrets(client):
    response = client.get("/ready")
    assert response.status_code in (200, 503)
    body = response.json()
    assert "ready" in body
    checks = body["checks"]

    assert checks["vector_store"]["ready"] is True
    assert checks["embeddings"]["ready"] is True
    assert checks["llm"]["status"] in ("configured", "ok", "unconfigured", "unavailable")

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if api_key:
        assert api_key not in json.dumps(body)


def test_ready_reports_unconfigured_llm(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["llm"]["status"] == "unconfigured"
    assert response.json()["ready"] is False
