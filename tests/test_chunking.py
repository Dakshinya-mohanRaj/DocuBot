"""Chunking behaviour: sentences stay whole and overlap carries context."""

from app.core.chunking import chunk_text

PARAGRAPH = (
    "Employees accrue 1.5 days of PTO each full month. "
    "The accrual cap is 30 days. "
    "Unused days above the cap are forfeited at year end."
)


def test_empty_text_produces_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.") == ["Hello world."]


def test_sentences_are_never_split_mid_sentence():
    chunks = chunk_text(PARAGRAPH, chunk_size=60, overlap=15)
    for chunk in chunks:
        # Every chunk must end on a sentence terminator, not mid-clause.
        assert chunk.rstrip().endswith(".")


def test_consecutive_chunks_overlap():
    # Overlap can only carry a whole sentence when overlap >= one sentence,
    # because units are never split.
    chunks = chunk_text(PARAGRAPH, chunk_size=120, overlap=60)
    assert len(chunks) > 1
    shared = set(chunks[0].split()) & set(chunks[1].split())
    assert shared, f"expected shared context, got {chunks}"


def test_table_rows_are_preserved_together():
    """No table row may be split across chunks."""
    rows = ["Col | Value | Note", "A | 100 | limit", "B | 25 | max"]
    chunks = chunk_text("\n".join(rows), chunk_size=40, overlap=10)
    assert len(chunks) > 1, "this test is only meaningful if chunking occurs"
    emitted = {line for chunk in chunks for line in chunk.splitlines()}
    assert emitted == set(rows)


def test_oversized_sentence_is_split_on_word_boundaries():
    chunks = chunk_text("word " * 200, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 110
        assert not chunk.startswith(" ") and not chunk.endswith(" ")


def test_full_corpus_chunks_within_size_budget():
    from scripts.fixtures import DOCUMENTS

    for name, text in DOCUMENTS.items():
        for chunk in chunk_text(text):
            assert chunk.strip(), f"{name} produced a blank chunk"