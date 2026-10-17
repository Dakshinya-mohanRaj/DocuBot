"""Multi-tenancy: one account must never see or touch another's data.

Covers document isolation, conversation isolation, and per-user vector
retrieval scoping.
"""

from app.core import vector_store

DOC_A = "The rate limit for account A is 111 requests per minute."
DOC_B = "The PTO policy for account B is 44 days of paid leave."


def _uid(client) -> str:
    return client.get("/auth/me").json()["user"]["id"]


def test_documents_are_isolated_per_user(client, make_user):
    bob = make_user()
    client.post("/ingest/text", json={"doc_id": "shared.txt", "text": DOC_A})
    bob.post("/ingest/text", json={"doc_id": "shared.txt", "text": DOC_B})

    a_docs = client.get("/ingest/documents").json()["documents"]
    b_docs = bob.get("/ingest/documents").json()["documents"]
    assert [d["doc_id"] for d in a_docs] == ["shared.txt"]
    assert [d["doc_id"] for d in b_docs] == ["shared.txt"]

    # Same document name, but each user's stored content is their own.
    a_chunk = vector_store.get_chunks_for_documents(["shared.txt"], user_id=_uid(client))[0]
    b_chunk = vector_store.get_chunks_for_documents(["shared.txt"], user_id=_uid(bob))[0]
    assert "111" in a_chunk["text"] and "44" not in a_chunk["text"]
    assert "44" in b_chunk["text"] and "111" not in b_chunk["text"]


def test_deleting_a_document_does_not_affect_the_other_user(client, make_user):
    bob = make_user()
    client.post("/ingest/text", json={"doc_id": "shared.txt", "text": DOC_A})
    bob.post("/ingest/text", json={"doc_id": "shared.txt", "text": DOC_B})

    assert client.delete("/ingest/shared.txt").status_code == 200
    assert client.get("/ingest/documents").json()["document_count"] == 0
    # Bob still owns his copy, untouched.
    assert bob.get("/ingest/documents").json()["document_count"] == 1
    assert bob.delete("/ingest/shared.txt").status_code == 200


def test_vector_query_is_scoped_by_user_id(store):
    store.add_chunks("d.txt", [DOC_A], filename="d.txt", user_id="user-a")
    store.add_chunks("d.txt", [DOC_B], filename="d.txt", user_id="user-b")

    a_hits = store.query("rate limit per minute", n_results=4, min_relevance=0.0, user_id="user-a")
    assert a_hits and all(h["doc_id"] == "d.txt" for h in a_hits)

    # User-a's view contains only their document; user-b's only theirs.
    a_text = store.get_chunks_for_documents(["d.txt"], user_id="user-a")[0]["text"]
    b_text = store.get_chunks_for_documents(["d.txt"], user_id="user-b")[0]["text"]
    assert "111" in a_text and "111" not in b_text
    assert "44" in b_text and "44" not in a_text


def test_retrieval_with_no_documents_of_your_own_is_empty(client, make_user):
    bob = make_user()  # uploads nothing
    client.post("/ingest/text", json={"doc_id": "a.txt", "text": DOC_A})

    # Bob has no documents, so a substantive question is rejected with 400.
    response = bob.post("/chat", json={"message": "What is the rate limit?"})
    assert response.status_code == 400
    assert bob.get("/ingest/documents").json()["document_count"] == 0
    # Alice is unaffected.
    assert client.post("/chat", json={"message": "What is the rate limit?"}).status_code == 200


def test_conversations_are_isolated(client, make_user):
    bob = make_user()
    conversation_id = client.post("/conversations").json()["conversation"]["id"]

    # Bob cannot read, rename, or delete Alice's conversation.
    assert bob.get(f"/conversations/{conversation_id}").status_code == 404
    assert bob.patch(
        f"/conversations/{conversation_id}", json={"title": "hijack"}
    ).status_code == 404
    assert bob.delete(f"/conversations/{conversation_id}").status_code == 404
    assert bob.get("/sessions/" + conversation_id + "/history").status_code == 404

    # Alice can.
    assert client.get(f"/conversations/{conversation_id}").status_code == 200

    # And Bob's own list never contains Alice's conversation.
    bob_ids = [c["id"] for c in bob.get("/conversations").json()["conversations"]]
    assert conversation_id not in bob_ids


def test_chat_session_history_is_isolated(client, make_user):
    bob = make_user()
    response = client.post("/chat", json={"message": "hi"})
    assert response.status_code == 200
    session_id = response.json()["session_id"]

    assert client.get(f"/sessions/{session_id}/history").status_code == 200
    assert bob.get(f"/sessions/{session_id}/history").status_code == 404


def test_continuing_a_foreign_conversation_is_404(client, make_user):
    """Passing another user's conversation_id to /chat must fail, not leak."""
    bob = make_user()
    conversation_id = client.post("/conversations").json()["conversation"]["id"]

    response = bob.post("/chat", json={"message": "hi", "conversation_id": conversation_id})
    assert response.status_code == 404
    # Alice's thread is still empty.
    assert client.get(f"/conversations/{conversation_id}").json()["messages"] == []


def test_renaming_and_deleting_own_conversation_works(client):
    conversation_id = client.post("/conversations").json()["conversation"]["id"]
    assert client.patch(
        f"/conversations/{conversation_id}", json={"title": "My thread"}
    ).status_code == 200
    assert client.get(
        f"/conversations/{conversation_id}"
    ).json()["conversation"]["title"] == "My thread"
    assert client.delete(f"/conversations/{conversation_id}").status_code == 200
    assert client.get(f"/conversations/{conversation_id}").status_code == 404
