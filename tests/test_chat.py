"""Chat, session memory, source attribution and LLM failure handling."""

from app.core.llm import LLMUnavailable

DOC = "The API enforces a rate limit of 100 requests per minute for each API key."


def _seed(client, name="spec.txt", text=DOC):
    response = client.post("/ingest/file", files={"file": (name, text.encode(), "text/plain")})
    assert response.status_code == 200, response.text


def test_llm_unavailable_returns_503_without_leaking_details(client, monkeypatch):
    _seed(client)

    def boom(question, hits, history):
        raise LLMUnavailable("The answer service is temporarily unavailable.")

    monkeypatch.setattr("app.routes.chat.generate_answer", boom)
    response = client.post("/chat", json={"message": "What is the API rate limit?"})
    assert response.status_code == 503
    assert response.json()["detail"] == "The answer service is temporarily unavailable."
    assert "Traceback" not in response.text


def test_unexpected_llm_failure_returns_sanitized_502(client, monkeypatch):
    _seed(client)

    def boom(question, hits, history):
        raise ValueError("internal secret detail")

    monkeypatch.setattr("app.routes.chat.generate_answer", boom)
    response = client.post("/chat", json={"message": "What is the API rate limit?"})
    assert response.status_code == 502
    assert "secret" not in response.text


def test_session_id_is_issued_and_reused(client):
    _seed(client)
    first = client.post("/chat", json={"message": "What is the rate limit?"}).json()
    session_id = first["session_id"]
    assert session_id

    second = client.post(
        "/chat", json={"message": "And per key?", "session_id": session_id}
    ).json()
    assert second["session_id"] == session_id

    history = client.get(f"/sessions/{session_id}/history")
    assert history.status_code == 200
    roles = [t["role"] for t in history.json()["history"]]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_follow_up_questions_see_prior_history(client, monkeypatch):
    _seed(client)
    seen = {}

    def capture(question, hits, history):
        seen.setdefault("calls", []).append(list(history))
        return "Answer " + str(len(seen["calls"]))

    monkeypatch.setattr("app.routes.chat.generate_answer", capture)
    first = client.post("/chat", json={"message": "What is the rate limit?"}).json()
    client.post(
        "/chat", json={"message": "And per key?", "session_id": first["session_id"]}
    )

    # The second call must have received the first exchange as history.
    assert seen["calls"][0] == []
    second_history = seen["calls"][1]
    assert any(t["content"] == "What is the rate limit?" for t in second_history)


def test_sessions_do_not_mix(client, monkeypatch):
    _seed(client)
    seen = {}

    def capture(question, hits, history):
        seen[question] = list(history)
        return "ok"

    monkeypatch.setattr("app.routes.chat.generate_answer", capture)
    client.post("/chat", json={"message": "What is the rate limit for session a?", "session_id": "aaa"})
    client.post("/chat", json={"message": "What is the rate limit for session b?", "session_id": "bbb"})

    # b's history must never contain a's turn.
    b_history = seen["What is the rate limit for session b?"]
    assert all("session a" not in t["content"] for t in b_history)


def test_session_id_too_long_is_rejected(client):
    _seed(client)
    response = client.post("/chat", json={"message": "hi", "session_id": "x" * 65})
    assert response.status_code == 422


def test_oversized_message_is_rejected(client):
    _seed(client)
    response = client.post("/chat", json={"message": "x" * 8001})
    assert response.status_code == 422


def test_sources_include_relevance_and_similarity(client):
    _seed(client)
    body = client.post("/chat", json={"message": "What is the API rate limit?"}).json()
    assert body["sources"], "a relevant question must cite a source"
    for source in body["sources"]:
        assert source["filename"]
        assert 0.0 <= source["similarity"] <= 1.0
        assert "relevance" in source


def test_abstention_records_history_and_cites_nothing(client, monkeypatch):
    _seed(client, "handbook.txt", "Employees accrue 1.5 days of PTO each full month.")
    monkeypatch.setattr(
        "app.routes.chat.generate_answer",
        lambda q, hits, history: "The uploaded documents do not contain information about that.",
    )
    body = client.post(
        "/chat", json={"message": "What is the office wifi password?"}
    ).json()
    assert body["abstained"] is True
    assert body["sources"] == []

    history = client.get(f"/sessions/{body['session_id']}/history")
    assert history.status_code == 200
    assert len(history.json()["history"]) == 2


def test_session_delete_is_idempotent_then_404(client):
    _seed(client)
    session_id = client.post("/chat", json={"message": "rate limit?"}).json()["session_id"]
    assert client.delete(f"/sessions/{session_id}").status_code == 200
    assert client.delete(f"/sessions/{session_id}").status_code == 404


def test_unknown_session_history_is_404(client):
    assert client.get("/sessions/nope/history").status_code == 404


def test_chat_response_schema_matches_frontend(client):
    """The UI reads exactly these fields — guard against breaking it."""
    _seed(client)
    body = client.post("/chat", json={"message": "What is the rate limit?"}).json()
    assert set(body) >= {"session_id", "answer", "sources", "abstained"}
    for source in body["sources"]:
        assert set(source) >= {"doc_id", "filename", "similarity", "chunk_index"}


# --- conversational intents (Features 1 & 4) -------------------------------

def test_greeting_skips_retrieval_and_llm(client, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("greeting must not touch retrieval or the LLM")

    monkeypatch.setattr("app.routes.chat.query", explode)
    monkeypatch.setattr("app.routes.chat.generate_answer", explode)

    body = client.post("/chat", json={"message": "Hi"}).json()
    assert "DocuBot" in body["answer"]
    assert body["sources"] == []
    assert body["abstained"] is False
    assert body["intent"] == "greeting"


def test_greeting_works_with_empty_knowledge_base(client):
    # No documents ingested, yet a greeting still succeeds.
    response = client.post("/chat", json={"message": "hello there"})
    assert response.status_code == 200
    assert response.json()["intent"] == "greeting"


def test_how_are_you_and_thanks_are_conversational(client):
    assert "doing well" in client.post("/chat", json={"message": "How are you?"}).json()["answer"]
    assert "welcome" in client.post("/chat", json={"message": "thanks!"}).json()["answer"].lower()


def test_capabilities_question_explains_features(client):
    answer = client.post("/chat", json={"message": "What can you do?"}).json()["answer"]
    assert "Document Q&A" in answer
    assert "Summarization" in answer
    assert "Analysis" in answer


def test_greeting_turns_are_recorded_in_session(client):
    body = client.post("/chat", json={"message": "hi"}).json()
    history = client.get(f"/sessions/{body['session_id']}/history").json()["history"]
    assert [t["role"] for t in history] == ["user", "assistant"]


def test_greeting_then_question_uses_document_pipeline(client, monkeypatch):
    _seed(client, "handbook.txt", "Employees receive 20 days of annual leave.")
    body = client.post(
        "/chat", json={"message": "hi, how much annual leave do employees get?"}
    ).json()
    assert body["intent"] == "document"
    assert body["sources"], "a greeting followed by a real question must retrieve evidence"


# --- document analysis through the API (Feature 3) -------------------------

def test_analysis_question_uses_deterministic_records(client, monkeypatch):
    _seed(client, "resume.txt", "Alice. Gender: female. Bob. Gender: male.")

    def fake_extract(question, chunks):
        return {
            "records": [
                {"name": "Alice", "gender": "female", "source": "resume.txt", "attributes": {}},
                {"name": "Bob", "gender": "male", "source": "resume.txt", "attributes": {}},
                {"name": "Carol", "gender": "unknown", "source": "resume.txt", "attributes": {}},
            ],
            "insufficient": False,
            "notes": "",
        }

    monkeypatch.setattr("app.core.analysis.extract_records", fake_extract)
    body = client.post(
        "/chat", json={"message": "How many candidates are male and female?"}
    ).json()

    assert body["intent"] == "analysis"
    assert "| Male | 1 |" in body["answer"]
    assert "| Female | 1 |" in body["answer"]
    assert "| Unknown / not stated | 1 |" in body["answer"]
    assert body["sources"], "analysis must cite the documents it read"

