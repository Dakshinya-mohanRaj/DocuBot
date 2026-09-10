import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.vector_store import query, collection_count
from app.core.llm import generate_answer
from app.core.session_store import append_turn, as_llm_messages

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    top_k: int = 4


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[dict]


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    if collection_count() == 0:
        raise HTTPException(status_code=400, detail="No documents ingested yet — call /ingest/text or /ingest/file first")

    session_id = payload.session_id or str(uuid.uuid4())[:8]

    hits = query(payload.message, n_results=payload.top_k)
    history = as_llm_messages(session_id)

    answer = generate_answer(payload.message, hits, history)

    append_turn(session_id, "user", payload.message)
    append_turn(session_id, "assistant", answer)

    sources = [{"doc_id": h["doc_id"], "score": round(h["score"], 4)} for h in hits]

    return ChatResponse(session_id=session_id, answer=answer, sources=sources)
