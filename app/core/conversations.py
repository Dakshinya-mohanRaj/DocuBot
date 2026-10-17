"""Conversation and message persistence (the durable replacement for the old
in-memory session store).

Every function is scoped by ``user_id`` so one account can never read, append
to, rename, or delete another account's conversation. Callers pass the id from
the authenticated session, never from the request body alone.
"""

import os

from sqlalchemy import delete, desc, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.models import Conversation, Message, new_id, utcnow

# Keep only the most recent messages per conversation. The LLM only sees a short
# window (chat.MAX_HISTORY_TURNS) but the transcript is retained for the UI, so
# this bounds unbounded growth on a long-lived thread.
MAX_STORED_MESSAGES = int(os.environ.get("DOCUBOT_MAX_STORED_MESSAGES", "200"))

DEFAULT_TITLE = "New chat"
TITLE_MAX = 60


def derive_title(text: str) -> str:
    title = " ".join((text or "").split())
    if not title:
        return DEFAULT_TITLE
    if len(title) > TITLE_MAX:
        title = title[:TITLE_MAX].rstrip() + "…"
    return title


def create_conversation(
    db: Session, user_id: str, conversation_id: str | None = None, title: str | None = None
) -> Conversation:
    row = Conversation(
        id=conversation_id or new_id(),
        user_id=user_id,
        title=title or DEFAULT_TITLE,
    )
    db.add(row)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise ValueError("A conversation with that id already exists.")
    db.refresh(row)
    return row


def get_conversation(db: Session, user_id: str, conversation_id: str) -> Conversation | None:
    row = db.get(Conversation, conversation_id)
    if row is None or row.user_id != user_id:
        return None
    return row


def ensure_conversation(
    db: Session, user_id: str, conversation_id: str | None
) -> Conversation | None:
    """Return the user's conversation, creating it if the id is free.

    ``None`` means the id belongs to a different user (ownership violation) —
    the caller turns that into a 404 so we never reveal another account's data.
    """
    if conversation_id is None:
        return create_conversation(db, user_id)
    existing = db.get(Conversation, conversation_id)
    if existing is not None:
        return existing if existing.user_id == user_id else None
    return create_conversation(db, user_id, conversation_id=conversation_id)


def list_conversations(db: Session, user_id: str) -> list[Conversation]:
    return list(
        db.scalars(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(desc(Conversation.updated_at), desc(Conversation.created_at))
        )
    )


def rename_conversation(db: Session, user_id: str, conversation_id: str, title: str) -> bool:
    row = get_conversation(db, user_id, conversation_id)
    if row is None:
        return False
    row.title = derive_title(title)
    row.updated_at = utcnow()
    db.commit()
    return True


def delete_conversation(db: Session, user_id: str, conversation_id: str) -> bool:
    row = get_conversation(db, user_id, conversation_id)
    if row is None:
        return False
    # Delete messages explicitly so this works regardless of database FK-cascade
    # support (SQLite only cascades when the foreign_keys pragma is on).
    db.execute(delete(Message).where(Message.conversation_id == conversation_id))
    db.delete(row)
    db.commit()
    return True


def append_message(
    db: Session,
    user_id: str,
    conversation_id: str,
    role: str,
    content: str,
    sources: list | None = None,
    abstained: bool = False,
    intent: str | None = None,
) -> Message | None:
    """Append one turn, returning it, or ``None`` if the user does not own it."""
    conversation = get_conversation(db, user_id, conversation_id)
    if conversation is None:
        return None

    message = Message(
        conversation_id=conversation_id,
        user_id=user_id,
        role=role,
        content=content,
        sources=sources,
        abstained=abstained,
        intent=intent,
    )
    db.add(message)
    conversation.updated_at = utcnow()
    db.commit()
    db.refresh(message)
    _trim(db, conversation_id)
    return message


def _trim(db: Session, conversation_id: str) -> None:
    ids = list(
        db.scalars(
            select(Message.id)
            .where(Message.conversation_id == conversation_id)
            .order_by(desc(Message.id))
            .offset(MAX_STORED_MESSAGES)
        )
    )
    if ids:
        db.execute(delete(Message).where(Message.id.in_(ids)))
        db.commit()


def get_messages(
    db: Session, user_id: str, conversation_id: str, limit: int | None = None
) -> list[Message]:
    if get_conversation(db, user_id, conversation_id) is None:
        return []
    query = (
        select(Message)
        .where(Message.conversation_id == conversation_id, Message.user_id == user_id)
        .order_by(desc(Message.id))
    )
    if limit is not None:
        query = query.limit(limit)
    messages = list(db.scalars(query))
    messages.reverse()  # back to chronological order
    return messages


def as_llm_messages(
    db: Session, user_id: str, conversation_id: str, limit: int
) -> list[dict]:
    """The last ``limit`` turns as ``{"role", "content"}`` for the LLM prompt."""
    recent = get_messages(db, user_id, conversation_id, limit=limit)
    return [{"role": m.role, "content": m.content} for m in recent]
