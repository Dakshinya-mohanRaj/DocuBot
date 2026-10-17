import logging
import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core import analysis, conversations, intents
from app.core.db import get_db
from app.core.dependencies import CurrentUser, require_csrf
from app.core.logging_config import log_event
from app.core.llm import LLMUnavailable, generate_answer
from app.core.vector_store import collection_count, query

logger = logging.getLogger("docubot.chat")

router = APIRouter(dependencies=[Depends(require_csrf)])

# Prior turns are passed back to the model as trusted conversation. Capping the
# window stops an early hallucinated answer hardening into "fact" later on,
# and keeps the prompt inside the model's context budget.
MAX_HISTORY_TURNS = 6

# Conversation ids are opaque tokens we issue, but a client may supply one to
# continue an existing thread. Bound what we will echo back and store.
MAX_CONVERSATION_ID_LEN = 64

NO_ANSWER = (
    "I couldn't find anything in your documents that answers that question. "
    "Try rephrasing it, or upload a document that covers the topic."
)

# When the confident relevance floor finds nothing, the grounded model decides
# whether the closest chunks actually answer the question. These phrases mark an
# answer as an abstention so its (unused) retrieved chunks are not shown as
# sources. Only consulted on the low-confidence path.
_ABSTENTION_RE = re.compile(
    r"(do not|don't|does not|doesn't|did not|didn't|cannot|can't|could not|couldn't|unable to)"
    r"[^.\n]{0,60}\b(contain|include|find|locate|provide|answer|determine|state|mention)\b"
    r"|\bno (relevant |sufficient |enough )?information\b"
    r"|\bnot (mentioned|found|stated|provided|available|covered|listed)\b",
    re.IGNORECASE,
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    # ``conversation_id`` is the current field; ``session_id`` is kept as an
    # accepted alias so existing clients keep working.
    conversation_id: str | None = None
    session_id: str | None = None
    top_k: int = Field(default=4, ge=1, le=8)


class ChatResponse(BaseModel):
    session_id: str
    conversation_id: str
    answer: str
    sources: list[dict]
    abstained: bool = False
    intent: str = "document"


def _requested_conversation_id(payload: ChatRequest) -> str | None:
    raw = payload.conversation_id if payload.conversation_id is not None else payload.session_id
    if raw is None:
        return None
    cid = raw.strip()
    if not (1 <= len(cid) <= MAX_CONVERSATION_ID_LEN):
        raise HTTPException(
            status_code=422,
            detail=f"conversation_id must be 1-{MAX_CONVERSATION_ID_LEN} characters",
        )
    return cid


def _record(
    db: Session,
    user,
    conversation_id: str,
    question: str,
    answer: str,
    sources: list[dict],
    abstained: bool,
    intent: str,
) -> None:
    conversations.append_message(db, user.id, conversation_id, "user", question)
    conversations.append_message(
        db,
        user.id,
        conversation_id,
        "assistant",
        answer,
        sources=sources,
        abstained=abstained,
        intent=intent,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, user: CurrentUser, db: Session = Depends(get_db)):
    # Resolve/create the conversation first; the id is validated even for a
    # greeting, and an id owned by another user is a 404 (never a data leak).
    requested = _requested_conversation_id(payload)
    conversation = conversations.ensure_conversation(db, user.id, requested)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    conversation_id = conversation.id

    # Name a fresh thread after its first message.
    if not conversations.as_llm_messages(db, user.id, conversation_id, 1):
        conversations.rename_conversation(
            db, user.id, conversation_id, conversations.derive_title(payload.message)
        )

    # Conversational messages (greetings, small talk, identity, help) are
    # answered directly: no retrieval, no LLM, and they work with an empty
    # knowledge base.
    intent = intents.detect(payload.message)
    if intent.is_conversational:
        answer = intent.response or ""
        _record(db, user, conversation_id, payload.message, answer, [], False, intent.kind)
        return ChatResponse(
            session_id=conversation_id,
            conversation_id=conversation_id,
            answer=answer,
            sources=[],
            abstained=False,
            intent=intent.kind,
        )

    if collection_count(user.id) == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet — call /ingest/text or /ingest/file first",
        )

    history = conversations.as_llm_messages(db, user.id, conversation_id, MAX_HISTORY_TURNS)

    # Document analysis (counting, demographic breakdowns, bulk extraction) uses
    # a deterministic record pipeline instead of free-form generation.
    if analysis.is_analysis_request(payload.message):
        evidence = analysis.gather_evidence(payload.message, user_id=user.id)
        try:
            answer = analysis.answer_from_evidence(payload.message, evidence, history)
        except LLMUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception as exc:
            log_event(logger, "analysis_failed", level=logging.ERROR, error=type(exc).__name__)
            raise HTTPException(status_code=502, detail="The answer service failed to respond. Please try again.")

        sources = _sources(evidence)
        _record(
            db, user, conversation_id, payload.message, answer,
            sources, not evidence, "analysis",
        )
        return ChatResponse(
            session_id=conversation_id,
            conversation_id=conversation_id,
            answer=answer,
            sources=sources,
            abstained=not evidence,
            intent="analysis",
        )

    try:
        hits = query(payload.message, n_results=payload.top_k, user_id=user.id)
        confident = bool(hits)
        if not hits:
            # Nothing cleared the confident floor. A single cosine threshold
            # separates poorly across corpora (a legitimate question can score
            # below an unrelated one), and meta questions like "summarize this
            # PDF" score low against every chunk. So instead of abstaining on
            # the threshold alone, fall back to the closest chunks and let the
            # grounded model decide — its prompt tells it to say the documents
            # don't cover the question when they don't.
            hits = query(
                payload.message, n_results=payload.top_k, min_relevance=0.0, user_id=user.id
            )
    except Exception as exc:
        log_event(logger, "retrieval_failed", level=logging.ERROR, error=type(exc).__name__)
        raise HTTPException(status_code=500, detail="Retrieval is temporarily unavailable.")

    # Even the soft pass found nothing usable (e.g. an empty store): answer
    # honestly without spending an LLM call, and cite nothing.
    if not hits:
        _record(db, user, conversation_id, payload.message, NO_ANSWER, [], True, "document")
        return ChatResponse(
            session_id=conversation_id,
            conversation_id=conversation_id,
            answer=NO_ANSWER,
            sources=[],
            abstained=True,
        )

    try:
        answer = generate_answer(payload.message, hits, history)
    except LLMUnavailable as exc:
        # Message is pre-sanitized by the llm module; details stay in our logs.
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        log_event(logger, "answer_generation_failed", level=logging.ERROR, error=type(exc).__name__)
        raise HTTPException(status_code=502, detail="The answer service failed to respond. Please try again.")

    # Only second-guess the model when retrieval was low-confidence; a confident
    # hit is allowed to answer even if it mentions something it couldn't find.
    abstained = (not confident) and _looks_like_abstention(answer)
    sources = [] if abstained else _sources(hits)

    _record(
        db, user, conversation_id, payload.message, answer, sources, abstained, "document"
    )
    return ChatResponse(
        session_id=conversation_id,
        conversation_id=conversation_id,
        answer=answer,
        sources=sources,
        abstained=abstained,
    )


def _looks_like_abstention(answer: str) -> bool:
    """True if the grounded answer says it could not find the information."""
    return bool(_ABSTENTION_RE.search(answer or ""))


def _sources(hits: list[dict]) -> list[dict]:
    """Project retrieval/analysis chunks onto the public source schema."""
    return [
        {
            "doc_id": hit["doc_id"],
            "filename": hit["filename"],
            "similarity": hit.get("similarity"),
            "relevance": hit.get("relevance"),
            "chunk_index": hit.get("chunk_index"),
            "page": hit.get("page"),
        }
        for hit in hits
    ]
