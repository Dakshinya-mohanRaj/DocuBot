"""ChromaDB vector store.

Collection space is cosine (an earlier build used L2, which made distances
incomparable across documents and impossible to threshold meaningfully), and
retrieval applies a relevance floor plus a per-document cap so that one verbose
document cannot consume every slot in the context window.
"""

import os
import re

import chromadb
from chromadb.utils import embedding_functions

CHROMA_PATH = os.environ.get("DOCUBOT_CHROMA_PATH", "./data/chroma")
COLLECTION_NAME = os.environ.get("DOCUBOT_COLLECTION", "docubot_chunks_v2")

# Chunks whose combined relevance falls below this are treated as unrelated, so
# the caller can abstain instead of handing weak context to the model.
#
# A cosine-similarity floor alone cannot separate signal from noise here:
# "laptop refresh cycle" scored 0.244 while the unanswerable "refund policy"
# scored 0.228 -- a 0.016 gap no threshold could split reliably. Adding a
# lexical term-overlap term widens that gap to 0.135, which does separate.
#
# 0.30 sits near the midpoint of the calibrated gap. On the evaluation corpus
# the lowest-scoring genuinely answerable question reached 0.362 and the
# highest-scoring unanswerable one reached 0.228. Override with
# DOCUBOT_MIN_RELEVANCE if your corpus behaves differently.
MIN_RELEVANCE = float(os.environ.get("DOCUBOT_MIN_RELEVANCE", "0.30"))

# Weight of the lexical term-overlap term relative to cosine similarity.
LEXICAL_WEIGHT = float(os.environ.get("DOCUBOT_LEXICAL_WEIGHT", "0.5"))

# Most chunks any single document may contribute to one context window.
PER_DOC_CAP = int(os.environ.get("DOCUBOT_PER_DOC_CAP", "2"))

# Over-fetch before filtering, so the cap does not starve other documents.
OVERFETCH_FACTOR = 4

_STOPWORDS = {
    "a", "an", "and", "any", "are", "as", "at", "be", "been", "but", "by",
    "can", "did", "do", "does", "for", "from", "get", "had", "has", "have",
    "how", "i", "if", "in", "into", "is", "it", "its", "me", "much", "my",
    "of", "on", "or", "our", "so", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "was", "we", "were",
    "what", "when", "where", "which", "who", "whom", "why", "will", "with",
    "you", "your",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Minimum shared prefix length before two words are treated as the same stem.
_STEM_MIN = 4

_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Local, free embedding model — no API key needed for this part.
_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=_embedding_fn,
    metadata={"hnsw:space": "cosine"},
)

# Warm the embedding model at startup so the first ingest/chat request
# never triggers a slow model load (which can time out the Render proxy).
_embedding_fn(["docubot-warmup"])


def add_chunks(doc_id: str, chunks: list[str], filename: str | None = None) -> int:
    """Replace all chunks for doc_id and return how many were actually stored.

    Chunk ids are derived from doc_id, and ChromaDB's ``add`` silently skips ids
    that already exist (it only logs "Insert of existing embedding ID"). That
    made re-uploading an edited file a no-op while still reporting success, so
    the previous version is always deleted first.
    """
    delete_document(doc_id)
    if not chunks:
        return 0

    display = filename or doc_id
    ids = [f"{doc_id}::{i}" for i in range(len(chunks))]
    metadatas = [
        {"doc_id": doc_id, "chunk_index": i, "filename": display}
        for i in range(len(chunks))
    ]
    _collection.add(ids=ids, documents=chunks, metadatas=metadatas)

    # Report what is genuinely in the store, not what we intended to write.
    stored = _collection.get(ids=ids).get("ids") or []
    return len(stored)


# Backwards-compatible alias.
upsert_document = add_chunks


def _stem(token: str) -> str:
    """Light plural folding: policies -> policy, laptops -> laptop, classes -> class."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("sses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _content_tokens(text: str) -> set[str]:
    """Significant words, with light plural folding."""
    tokens: set[str] = set()
    for token in _TOKEN_RE.findall(text.lower()):
        if token in _STOPWORDS or len(token) < 2:
            continue
        tokens.add(_stem(token))
    return tokens


def _lexical_overlap(query_tokens: set[str], chunk_tokens: set[str]) -> float:
    """Fraction of query content words present in a chunk, 0.0-1.0.

    Matching is prefix-based so "laptop" matches "laptops" and "policy" matches
    "policies" without needing a full stemmer.
    """
    if not query_tokens:
        return 0.0
    matched = 0
    for query_token in query_tokens:
        for chunk_token in chunk_tokens:
            if query_token == chunk_token:
                matched += 1
                break
            if (
                len(query_token) >= _STEM_MIN
                and len(chunk_token) >= _STEM_MIN
                and (query_token.startswith(chunk_token) or chunk_token.startswith(query_token))
            ):
                matched += 1
                break
    return matched / len(query_tokens)


def query(
    question: str,
    n_results: int = 4,
    min_relevance: float | None = None,
    per_doc_cap: int | None = None,
) -> list[dict]:
    """Return the most relevant chunks, filtered and diversified.

    An empty list means nothing in the store cleared the relevance floor, which
    callers should treat as "no answer available" rather than passing weak
    chunks to the model.
    """
    total = collection_count()
    if total == 0:
        return []

    floor = MIN_RELEVANCE if min_relevance is None else min_relevance
    cap = PER_DOC_CAP if per_doc_cap is None else per_doc_cap
    n_results = max(1, min(n_results, 8))

    fetch = min(total, n_results * OVERFETCH_FACTOR)
    results = _collection.query(query_texts=[question], n_results=fetch)
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    query_tokens = _content_tokens(question)

    scored: list[dict] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        doc_id = metadata.get("doc_id", "unknown")
        # Chroma returns cosine distance, so similarity is 1 - distance.
        similarity = 1.0 - float(distance)
        overlap = _lexical_overlap(query_tokens, _content_tokens(document))
        scored.append(
            {
                "text": document,
                "doc_id": doc_id,
                "chunk_index": metadata.get("chunk_index", 0),
                "filename": metadata.get("filename") or doc_id,
                "similarity": round(similarity, 4),
                "lexical_overlap": round(overlap, 4),
                "relevance": round(similarity + LEXICAL_WEIGHT * overlap, 4),
            }
        )

    relevant = [hit for hit in scored if hit["relevance"] >= floor]
    relevant.sort(key=lambda hit: hit["relevance"], reverse=True)

    selected: list[dict] = []
    per_doc: dict[str, int] = {}
    for hit in relevant:
        used = per_doc.get(hit["doc_id"], 0)
        if used >= cap:
            continue
        per_doc[hit["doc_id"]] = used + 1
        selected.append(hit)
        if len(selected) >= n_results:
            break

    return selected


def collection_count() -> int:
    return _collection.count()


def list_documents() -> list[dict]:
    """Summarise every document currently in the store, with chunk counts."""
    fetched = _collection.get(include=["metadatas"])
    summary: dict[str, dict] = {}
    for metadata in fetched.get("metadatas") or []:
        if not metadata:
            continue
        doc_id = metadata.get("doc_id", "unknown")
        entry = summary.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "filename": metadata.get("filename") or doc_id,
                "chunks": 0,
            },
        )
        entry["chunks"] += 1
    return sorted(summary.values(), key=lambda item: item["filename"])


def delete_document(doc_id: str) -> int:
    """Delete all chunks belonging to doc_id. Returns the number removed."""
    existing = _collection.get(where={"doc_id": doc_id})
    ids_to_delete = (existing or {}).get("ids") or []
    if ids_to_delete:
        _collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)