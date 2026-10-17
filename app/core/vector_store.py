"""ChromaDB vector store.

Collection space is cosine (an earlier build used L2, which made distances
incomparable across documents and impossible to threshold meaningfully), and
retrieval applies a relevance floor plus a per-document cap so that one verbose
document cannot consume every slot in the context window.
"""

import logging
import os
import re
import uuid

import chromadb
from chromadb.utils import embedding_functions

from app.core.logging_config import log_event

logger = logging.getLogger("docubot.vector_store")

CHROMA_PATH = os.environ.get("DOCUBOT_CHROMA_PATH", "./data/chroma")
COLLECTION_NAME = os.environ.get("DOCUBOT_COLLECTION", "docubot_chunks_v2")

# Chunks whose combined relevance falls below this are treated as unrelated, so
# the caller can abstain instead of handing weak context to the model.
#
# 0.30 is calibrated for all-MiniLM-L6-v2 on the evaluation corpus: the lowest
# genuinely answerable question scored 0.362 and the highest unanswerable one
# scored 0.228. Override with DOCUBOT_MIN_RELEVANCE if your corpus behaves
# differently.
MIN_RELEVANCE = float(os.environ.get("DOCUBOT_MIN_RELEVANCE", "0.30"))

# Weight of the lexical term-overlap term relative to cosine similarity.
LEXICAL_WEIGHT = float(os.environ.get("DOCUBOT_LEXICAL_WEIGHT", "0.5"))

# Most chunks any single document may contribute to one context window. Kept low
# so one verbose document cannot crowd out the rest and to bound prompt tokens.
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
# all-MiniLM-L6-v2 (384-dim, 22M params) is ~5x smaller and much faster to encode
# than all-mpnet-base-v2 (768-dim, 110M params), for a modest retrieval-quality
# drop — keeping memory, startup time, and CPU low. Bake the model you choose
# into the image (see Dockerfile) so it is never downloaded at runtime.
# NOTE: After changing the model you MUST clear the Chroma database and
# re-ingest all documents because embeddings from different models are
# incompatible. Delete ./data/chroma and restart the server.
_EMBEDDING_MODEL = os.environ.get("DOCUBOT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=_EMBEDDING_MODEL
)

_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=_embedding_fn,
    metadata={"hnsw:space": "cosine"},
)

# Warm the embedding model at startup so the first ingest/chat request
# never triggers a slow model load (which can time out the Render proxy).
_embedding_fn(["docubot-warmup"])


def add_chunks(
    doc_id: str,
    chunks: list[str],
    filename: str | None = None,
    metadatas: list[dict] | None = None,
) -> int:
    """Write chunks for doc_id, replacing any previously stored version.

    Writes are made "generation safe": new chunks are written under fresh ids
    first and only then are the old ids removed. A crash or embedding failure
    mid-write therefore leaves the previous good version intact instead of
    deleting it first (the old delete-then-add order could silently drop a
    document when the new write failed).

    ``metadatas`` may supply per-chunk metadata (e.g. ``{"page": 3}``); it is
    merged with the fixed keys, which always win.

    Returns the number of chunks genuinely stored (verified by read-back), not
    the number intended.
    """
    old_ids = (_collection.get(where={"doc_id": doc_id}).get("ids")) or []
    if not chunks:
        # An empty re-upload is still a replacement: drop the old version.
        if old_ids:
            _collection.delete(ids=old_ids)
        return 0

    display = filename or doc_id
    generation = uuid.uuid4().hex[:8]
    ids = [f"{doc_id}::{generation}::{i}" for i in range(len(chunks))]

    base_meta = []
    for i in range(len(chunks)):
        extra = (metadatas[i] if metadatas and i < len(metadatas) else None) or {}
        meta = {k: v for k, v in extra.items() if v is not None}
        meta.update({"doc_id": doc_id, "chunk_index": i, "filename": display})
        base_meta.append(meta)

    try:
        _collection.add(ids=ids, documents=chunks, metadatas=base_meta)
    except Exception:
        # Roll back any partially written new chunks so the store does not end
        # up with a half-indexed document, and leave the old version in place.
        try:
            _collection.delete(ids=ids)
        except Exception:
            log_event(logger, "rollback_failed", level=logging.ERROR, doc_id=doc_id)
        log_event(logger, "add_chunks_failed", level=logging.ERROR, doc_id=doc_id)
        raise

    if old_ids:
        try:
            _collection.delete(ids=old_ids)
        except Exception:
            # New content is safely stored; the stale generation is dead weight
            # that list_documents would double-count. Surface it loudly.
            log_event(
                logger,
                "stale_generation_not_deleted",
                level=logging.WARNING,
                doc_id=doc_id,
                stale=len(old_ids),
            )

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
                "page": metadata.get("page"),
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


def document_exists(doc_id: str) -> bool:
    """True if any chunk for doc_id is stored (non-destructive)."""
    found = _collection.get(where={"doc_id": doc_id}, limit=1)
    return bool((found or {}).get("ids"))


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