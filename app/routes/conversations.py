"""Conversation (chat thread) management for the sidebar.

Every route is scoped to the authenticated user; the conversation id in the URL
is only ever honoured when it belongs to the caller.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core import conversations
from app.core.db import get_db
from app.core.dependencies import CurrentUser, require_csrf
from app.core.models import Conversation, Message

# CSRF is required on the unsafe methods; the dependency no-ops on GET.
router = APIRouter(prefix="/conversations", dependencies=[Depends(require_csrf)])


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


def _conversation_dict(row: Conversation, message_count: int | None = None) -> dict:
    data = {
        "id": row.id,
        "title": row.title,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }
    if message_count is not None:
        data["message_count"] = message_count
    return data


def _message_dict(row: Message) -> dict:
    return {
        "id": row.id,
        "role": row.role,
        "content": row.content,
        "sources": row.sources or [],
        "abstained": bool(row.abstained),
        "intent": row.intent,
        "timestamp": row.created_at.isoformat() if row.created_at else None,
    }


@router.get("")
def list_conversations(user: CurrentUser, db: Session = Depends(get_db)) -> dict:
    rows = conversations.list_conversations(db, user.id)
    return {"conversations": [_conversation_dict(row) for row in rows]}


@router.post("", status_code=status.HTTP_201_CREATED)
def create_conversation(user: CurrentUser, db: Session = Depends(get_db)) -> dict:
    row = conversations.create_conversation(db, user.id)
    return {"conversation": _conversation_dict(row, message_count=0)}


@router.get("/{conversation_id}")
def get_conversation(conversation_id: str, user: CurrentUser, db: Session = Depends(get_db)) -> dict:
    row = conversations.get_conversation(db, user.id, conversation_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    messages = conversations.get_messages(db, user.id, conversation_id)
    return {
        "conversation": _conversation_dict(row, message_count=len(messages)),
        "messages": [_message_dict(m) for m in messages],
    }


@router.patch("/{conversation_id}")
def rename_conversation(
    conversation_id: str, payload: RenameRequest, user: CurrentUser, db: Session = Depends(get_db)
) -> dict:
    if not conversations.rename_conversation(db, user.id, conversation_id, payload.title):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    row = conversations.get_conversation(db, user.id, conversation_id)
    return {"conversation": _conversation_dict(row)}


@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: str, user: CurrentUser, db: Session = Depends(get_db)) -> dict:
    if not conversations.delete_conversation(db, user.id, conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"id": conversation_id, "deleted": True}
