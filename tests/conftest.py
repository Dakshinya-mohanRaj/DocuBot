import os
import tempfile

import pytest

# --- Environment isolation -------------------------------------------------
# Point the database at a throwaway SQLite file and the vector store at a
# throwaway directory. These must be set BEFORE anything under app.* is imported
# (the engine and the Chroma collection are created at import time).
os.environ.setdefault(
    "DOCUBOT_DATABASE_URL",
    f"sqlite:///{tempfile.mkdtemp(prefix='docubot_db_')}/docubot.db",
)
os.environ.setdefault(
    "DOCUBOT_CHROMA_PATH", tempfile.mkdtemp(prefix="docubot_tests_")
)
os.environ.setdefault("DOCUBOT_COLLECTION", "docubot_test_chunks")
# Never rate-limit the test suite (one client IP would trip the per-IP buckets).
os.environ["DOCUBOT_RATE_LIMIT_CHAT"] = "0"
os.environ["DOCUBOT_RATE_LIMIT_INGEST"] = "0"
os.environ["DOCUBOT_RATE_LIMIT_AUTH"] = "0"
# TestClient talks plain http, so cookies must not be marked Secure.
os.environ["DOCUBOT_COOKIE_SECURE"] = "false"

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import delete  # noqa: E402

from app.core import security  # noqa: E402
from app.core import vector_store  # noqa: E402
from app.core.db import SessionLocal, init_db  # noqa: E402
from app.core.models import AuthSession, Conversation, Message, User  # noqa: E402
from app.main import app  # noqa: E402

# Create tables once so the autouse cleanup can run before any request.
init_db()

DEFAULT_EMAIL = "user@example.com"
DEFAULT_PASSWORD = "correct-horse-battery"


def _register_and_login(email: str, password: str, display_name: str | None = None) -> TestClient:
    """Sign up (or fall back to login) and return a CSRF-enabled client."""
    client = TestClient(app)
    payload = {"email": email, "password": password}
    if display_name:
        payload["display_name"] = display_name
    response = client.post("/auth/signup", json=payload)
    if response.status_code != 201:
        response = client.post("/auth/login", json={"email": email, "password": password})
        assert response.status_code == 200, response.text
    # Double-submit CSRF: send the readable cookie's value back as a header.
    client.headers["X-CSRF-Token"] = client.cookies.get(security.CSRF_COOKIE, "")
    return client


@pytest.fixture(autouse=True)
def _clean_state():
    """Every test starts and ends with empty accounts and an empty store."""

    def wipe():
        with SessionLocal() as db:
            db.execute(delete(Message))
            db.execute(delete(Conversation))
            db.execute(delete(AuthSession))
            db.execute(delete(User))
            db.commit()
        vector_store.reset_collection()

    wipe()
    yield
    wipe()


@pytest.fixture(autouse=True)
def _no_live_llm(monkeypatch):
    """Default every test to a stubbed LLM so no test spends a paid API call.

    Tests that need different behaviour (error paths, capturing arguments,
    asserting the LLM is never called) override this with their own
    monkeypatch, which runs after this fixture.
    """
    monkeypatch.setattr(
        "app.routes.chat.generate_answer",
        lambda question, hits, history: "Stub answer [doc]",
    )
    # Analysis has its own reference to the LLM. By default it reports
    # "insufficient" so the deterministic path runs without any model call;
    # analysis tests inject their own records fixture.
    monkeypatch.setattr(
        "app.core.analysis.extract_records",
        lambda question, chunks: {"records": [], "insufficient": True, "notes": ""},
    )
    monkeypatch.setattr(
        "app.core.analysis.generate_answer",
        lambda question, chunks, history: "Stub analysis fallback [doc]",
    )


@pytest.fixture()
def client():
    """An authenticated TestClient for the default user."""
    return _register_and_login(DEFAULT_EMAIL, DEFAULT_PASSWORD)


@pytest.fixture()
def anon_client():
    """An unauthenticated TestClient, for exercising auth guards."""
    return TestClient(app)


@pytest.fixture()
def make_user():
    """Factory for additional authenticated accounts (isolation tests)."""
    counter = {"n": 0}

    def _make(display_name: str | None = None) -> TestClient:
        counter["n"] += 1
        email = f"user{counter['n']}@example.com"
        return _register_and_login(email, DEFAULT_PASSWORD, display_name=display_name)

    return _make


@pytest.fixture()
def store():
    return vector_store
