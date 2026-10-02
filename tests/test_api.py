"""API contract tests for the ingest and chat routes."""

import pytest

from scripts.fixtures import DOCUMENTS


def _upload(client, name, text):
    response = client.post(
        "/ingest/file", files={"file": (name, text.encode(), "text/plain")}
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_upload_reports_chunks_and_store_size(client):
    info = _upload(client, "a.txt", "Alpha beta gamma delta.")
    assert info["chunks_added"] >= 1
    assert info["total_chunks_in_store"] == info["chunks_added"]
    assert info["replaced"] is False


def test_reuploading_same_name_replaces_and_flags(client):
    _upload(client, "a.txt", "The rate limit is 100 requests per minute.")
    second = _upload(client, "a.txt", "The rate limit is now 250 requests per minute.")

    assert second["replaced"] is True
    documents = client.get("/ingest/documents").json()["documents"]
    assert len(documents) == 1, "same filename must not create a second document"


def test_list_documents_endpoint(client):
    _upload(client, "handbook.txt", DOCUMENTS["handbook.txt"])
    _upload(client, "spec.txt", DOCUMENTS["spec.txt"])

    body = client.get("/ingest/documents").json()
    assert body["document_count"] == 2
    assert {d["filename"] for d in body["documents"]} == {"handbook.txt", "spec.txt"}
    assert all(d["chunks"] >= 1 for d in body["documents"])


def test_delete_removes_document(client):
    _upload(client, "a.txt", "Some content here.")
    response = client.delete("/ingest/a.txt")
    assert response.status_code == 200
    assert response.json()["chunks_deleted"] >= 1
    assert client.get("/ingest/documents").json()["document_count"] == 0


def test_empty_text_is_rejected(client):
    response = client.post("/ingest/text", json={"text": "   "})
    assert response.status_code == 400


def test_unsupported_extension_is_rejected(client):
    response = client.post(
        "/ingest/file", files={"file": ("evil.exe", b"MZ", "application/octet-stream")}
    )
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]


def test_chat_without_documents_is_rejected(client):
    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 400
    assert "No documents" in response.json()["detail"]


@pytest.mark.parametrize("top_k", [0, -1, 9, 1000])
def test_top_k_outside_bounds_is_rejected(client, top_k):
    _upload(client, "a.txt", "Some content about limits.")
    response = client.post("/chat", json={"message": "limits?", "top_k": top_k})
    assert response.status_code == 422


@pytest.mark.parametrize("top_k", [1, 4, 8])
def test_top_k_inside_bounds_is_accepted(client, top_k):
    _upload(client, "a.txt", "Some content about limits.")
    response = client.post("/chat", json={"message": "limits?", "top_k": top_k})
    assert response.status_code == 200


def test_empty_message_is_rejected(client):
    _upload(client, "a.txt", "Some content.")
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422


def test_abstention_cites_nothing(client):
    """A question no document covers must return no sources at all."""
    _upload(client, "handbook.txt", DOCUMENTS["handbook.txt"])
    response = client.post("/chat", json={"message": "What is the office wifi password?"})
    body = response.json()
    assert response.status_code == 200
    assert body["abstained"] is True
    assert body["sources"] == []
    assert "couldn't find" in body["answer"].lower()


def test_sources_carry_filename_and_similarity(client, monkeypatch):
    _upload(client, "spec.txt", DOCUMENTS["spec.txt"])
    # Stub the LLM so this test never spends an API call.
    monkeypatch.setattr(
        "app.routes.chat.generate_answer",
        lambda question, hits, history: "Stubbed answer [spec.txt]",
    )
    body = client.post(
        "/chat", json={"message": "What is the API rate limit per key?"}
    ).json()

    assert body["abstained"] is False
    assert body["answer"] == "Stubbed answer [spec.txt]"
    assert body["sources"], "a relevant question must cite a source"
    for source in body["sources"]:
        assert source["filename"]
        assert 0.0 <= source["similarity"] <= 1.0


def test_history_is_capped(client, monkeypatch):
    from app.routes.chat import MAX_HISTORY_TURNS
    from app.core.session_store import append_turn

    _upload(client, "spec.txt", DOCUMENTS["spec.txt"])

    seen = {}

    def capture(question, hits, history):
        seen["len"] = len(history)
        return "ok"

    monkeypatch.setattr("app.routes.chat.generate_answer", capture)

    for i in range(20):
        append_turn("s1", "user", f"q{i}")
        append_turn("s1", "assistant", f"a{i}")

    client.post("/chat", json={"message": "rate limit?", "session_id": "s1"})
    assert seen["len"] <= MAX_HISTORY_TURNS


def test_chat_does_not_send_an_llm_call_when_abstaining(client, monkeypatch):
    _upload(client, "handbook.txt", DOCUMENTS["handbook.txt"])

    def explode(*args, **kwargs):
        raise AssertionError("LLM must not be called when nothing is relevant")

    monkeypatch.setattr("app.routes.chat.generate_answer", explode)
    response = client.post("/chat", json={"message": "What is the office wifi password?"})
    assert response.status_code == 200
    assert response.json()["abstained"] is True