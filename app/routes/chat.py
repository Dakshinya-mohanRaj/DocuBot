import logging
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.logging_config import log_event
from app.core.llm import LLMUnavailable, generate_answer
from app.core.session_store import append_turn, as_llm_messages
from app.core.vector_store import collection_count, query

logger = logging.getLogger("docubot.chat")

router = APIRouter()

# Prior turns are passed back to the model as trusted conversation. Capping the
# window stops an early hallucinated answer hardening into "fact" later on,
# and keeps the prompt inside the model's context budget.
MAX_HISTORY_TURNS = 6

# Session ids are opaque tokens we issue; bound what we will echo back into the
# LLM prompt and reject anything that cannot plausibly be one.
MAX_SESSION_ID_LEN = 64

NO_ANSWER = (
    "I couldn't find anything in your documents that answers that question. "
    "Try rephrasing it, or upload a document that covers the topic."
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
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

    if payload.session_id is not None:
        sid = payload.session_id.strip()
        if not (1 <= len(sid) <= MAX_SESSION_ID_LEN):
            raise HTTPException(
                status_code=422,
                detail=f"session_id must be 1-{MAX_SESSION_ID_LEN} characters",
            )
        session_id = sid
    else:
        session_id = str(uuid.uuid4())[:8]

    try:
        hits = query(payload.message, n_results=payload.top_k)
    except Exception as exc:
        log_event(logger, "retrieval_failed", level=logging.ERROR, error=type(exc).__name__)
        raise HTTPException(status_code=500, detail="Retrieval is temporarily unavailable.")

    history = as_llm_messages(session_id)[-MAX_HISTORY_TURNS:]

    # Nothing cleared the relevance floor. Answer honestly without spending an
    # LLM call, and cite nothing rather than attaching unrelated chunks.
    if not hits:
        append_turn(session_id, "user", payload.message)
        append_turn(session_id, "assistant", NO_ANSWER)
        return ChatResponse(
            session_id=session_id, answer=NO_ANSWER, sources=[], abstained=True
        )

    try:
        answer = generate_answer(payload.message, hits, history)
    except LLMUnavailable as exc:
        # Message is pre-sanitized by the llm module; details stay in our logs.
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        log_event(logger, "answer_generation_failed", level=logging.ERROR, error=type(exc).__name__)
        raise HTTPException(status_code=502, detail="The answer service failed to respond. Please try again.")

    append_turn(session_id, "user", payload.message)
    append_turn(session_id, "assistant", answer)

    sources = [
        {
            "doc_id": hit["doc_id"],
            "filename": hit["filename"],
            "similarity": hit["similarity"],
            "relevance": hit["relevance"],
            "chunk_index": hit["chunk_index"],
            "page": hit.get("page"),
        }
        for hit in hits
    ]

    return ChatResponse(
        session_id=session_id, answer=answer, sources=sources, abstained=False
    )
