import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.vector_store import query, collection_count
from app.core.llm import generate_answer
from app.core.session_store import append_turn, as_llm_messages

router = APIRouter()

# Prior turns are passed back to the model as trusted conversation. Capping the
# window stops an early hallucinated answer hardening into "fact" later on,
# and keeps the prompt inside the model's context budget.
MAX_HISTORY_TURNS = 6

NO_ANSWER = (
    "I couldn't find anything in your documents that answers that question. "
    "Try rephrasing it, or upload a document that covers the topic."
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None
    top_k: int = Field(default=4, ge=1, le=8)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[dict]
    abstained: bool = False


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    if collection_count() == 0:
        raise HTTPException(status_code=400, detail="No documents ingested yet — call /ingest/text or /ingest/file first")

    session_id = payload.session_id or str(uuid.uuid4())[:8]

    try:
        hits = query(payload.message, n_results=payload.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    # Nothing cleared the relevance floor. Answer honestly without spending an
    # LLM call, and cite nothing rather than attaching unrelated chunks.
    if not hits:
        return ChatResponse(
            session_id=session_id, answer=NO_ANSWER, sources=[], abstained=True
        )

    history = as_llm_messages(session_id)[-MAX_HISTORY_TURNS:]

    try:
        answer = generate_answer(payload.message, hits, history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    append_turn(session_id, "user", payload.message)
    append_turn(session_id, "assistant", answer)

    sources = [
        {
            "doc_id": hit["doc_id"],
            "filename": hit["filename"],
            "similarity": hit["similarity"],
            "chunk_index": hit["chunk_index"],
        }
        for hit in hits
    ]

    return ChatResponse(
        session_id=session_id, answer=answer, sources=sources, abstained=False
    )