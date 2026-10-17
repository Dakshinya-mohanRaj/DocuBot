import os
import tempfile

import pytest

# The vector store is created at import time, so the isolation path has to be
# set before anything under app.core.vector_store is imported. Doing this at
# module scope keeps every test file pointed at the same throwaway directory.
os.environ.setdefault(
    "DOCUBOT_CHROMA_PATH", tempfile.mkdtemp(prefix="docubot_tests_")
)
os.environ.setdefault("DOCUBOT_COLLECTION", "docubot_test_chunks")
# Never rate-limit the test suite (one client IP would trip the per-IP buckets).
os.environ["DOCUBOT_RATE_LIMIT_CHAT"] = "0"
os.environ["DOCUBOT_RATE_LIMIT_INGEST"] = "0"

from fastapi.testclient import TestClient  # noqa: E402

from app.core import vector_store  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_store():
    """Every test starts and ends with an empty vector store."""
    def wipe():
        for document in vector_store.list_documents():
            vector_store.delete_document(document["doc_id"])

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


@pytest.fixture()
def client():
    """A TestClient over an emptied vector store."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def store():
    return vector_store
