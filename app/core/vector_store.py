import chromadb
from chromadb.utils import embedding_functions

_client = chromadb.PersistentClient(path="./data/chroma")

# Local, free embedding model — no API key needed for this part.
_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

_collection = _client.get_or_create_collection(
    name="docubot_chunks",
    embedding_function=_embedding_fn,
)


def add_chunks(doc_id: str, chunks: list[str]) -> int:
    ids = [f"{doc_id}::{i}" for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]
    _collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    return len(chunks)


def query(question: str, n_results: int = 4) -> list[dict]:
    results = _collection.query(query_texts=[question], n_results=n_results)
    if not results["documents"] or not results["documents"][0]:
        return []

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"text": doc, "doc_id": meta["doc_id"], "score": dist})
    return hits


def collection_count() -> int:
    return _collection.count()
