"""Vector store behaviour: honest upserts, filtering, and diversification."""

from app.core.vector_store import (
    _content_tokens,
    _lexical_overlap,
    query,
)


def test_add_then_count(store):
    added = store.add_chunks("a.txt", ["alpha", "beta", "gamma"], filename="a.txt")
    assert added == 3
    assert store.collection_count() == 3


def test_reupload_replaces_previous_content(store):
    """The bug this guards: re-adding the same ids silently kept stale text."""
    store.add_chunks("a.txt", ["original version one"], filename="a.txt")
    store.add_chunks("a.txt", ["original version one"], filename="a.txt")

    store.add_chunks("a.txt", ["updated version two"], filename="a.txt")

    stored = store._collection.get(where={"doc_id": "a.txt"})
    assert stored["documents"] == ["updated version two"]


def test_reupload_reports_true_write_count(store):
    store.add_chunks("a.txt", ["one", "two", "three", "four"], filename="a.txt")
    # Fewer chunks this time: the count must reflect reality, not intent.
    assert store.add_chunks("a.txt", ["only one now"], filename="a.txt") == 1


def test_shorter_replacement_does_not_leave_orphan_chunks(store):
    store.add_chunks("a.txt", ["one", "two", "three", "four", "five"], filename="a.txt")
    store.add_chunks("a.txt", ["tiny"], filename="a.txt")
    assert store.collection_count() == 1


def test_empty_chunk_list_writes_nothing(store):
    assert store.add_chunks("empty.txt", [], filename="empty.txt") == 0


def test_documents_with_the_same_name_collide(store):
    """Same filename means the same logical document, so re-upload replaces."""
    store.add_chunks("report.pdf", ["version one"], filename="report.pdf")
    store.add_chunks("report.pdf", ["version two"], filename="report.pdf")
    documents = store.list_documents()
    assert len(documents) == 1
    assert documents[0]["chunks"] == 1


def test_list_documents_reports_chunk_counts(store):
    store.add_chunks("one.txt", ["a", "b"], filename="one.txt")
    store.add_chunks("two.txt", ["c"], filename="two.txt")
    listed = {doc["doc_id"]: doc for doc in store.list_documents()}
    assert listed["one.txt"]["chunks"] == 2
    assert listed["two.txt"]["chunks"] == 1


def test_delete_removes_only_the_named_document(store):
    store.add_chunks("keep.txt", ["stays"], filename="keep.txt")
    store.add_chunks("drop.txt", ["goes"], filename="drop.txt")
    assert store.delete_document("drop.txt") == 1
    assert [d["doc_id"] for d in store.list_documents()] == ["keep.txt"]


def test_delete_of_unknown_document_is_zero(store):
    assert store.delete_document("nope.txt") == 0


def test_query_on_empty_store_returns_nothing(store):
    assert query("anything", min_relevance=-1.0) == []


def test_query_filters_below_the_relevance_floor(store):
    store.add_chunks("d.txt", ["The rate limit is 100 requests per minute."],
                     filename="d.txt")
    assert query("wifi password", n_results=4, min_relevance=0.30) == []


def test_query_returns_relevant_chunk_with_similarity(store):
    store.add_chunks("d.txt", ["The rate limit is 100 requests per minute."],
                     filename="d.txt")
    hits = query("what is the rate limit", n_results=4, min_relevance=0.30)
    assert len(hits) == 1
    assert 0.0 <= hits[0]["similarity"] <= 1.0
    assert hits[0]["relevance"] >= 0.30


def test_per_document_cap_limits_dominance(store):
    store.add_chunks("noisy.txt", ["limit limit limit"] * 6, filename="noisy.txt")
    store.add_chunks("quiet.txt", ["the limit is 100 per minute"], filename="quiet.txt")

    hits = query("limit", n_results=4, min_relevance=0.0, per_doc_cap=1)
    noisy = [h for h in hits if h["doc_id"] == "noisy.txt"]
    assert len(noisy) == 1, "one document must not fill every context slot"


def test_n_results_is_bounded(store):
    store.add_chunks("d.txt", [f"chunk number {i}" for i in range(10)],
                     filename="d.txt")
    assert len(query("chunk", n_results=99, min_relevance=-1.0)) <= 8


# ---- lexical helpers -------------------------------------------------------

def test_content_tokens_drop_stopwords_and_fold_plurals():
    tokens = _content_tokens("What are the policies for the laptops?")
    assert "the" not in tokens and "what" not in tokens
    assert "policy" in tokens
    assert "laptop" in tokens


def test_content_tokens_do_not_stem_words_ending_in_ss():
    assert "class" in _content_tokens("classes")


def test_lexical_overlap_is_proportional():
    assert _lexical_overlap({"a"}, {"a", "b", "c"}) == 1.0
    assert _lexical_overlap({"a"}, {"b"}) == 0.0
    assert _lexical_overlap(set(), {"a"}) == 0.0


def test_lexical_overlap_matches_across_word_forms():
    overlap = _lexical_overlap(_content_tokens("laptops refreshed"),
                               _content_tokens("The laptop refresh cycle"))
    assert overlap > 0.0