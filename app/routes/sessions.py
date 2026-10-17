"""Backwards-compatible session endpoints, now backed by user-owned
conversations.

``session_id`` and ``conversation_id`` refer to the same resource. These routes
exist so older clients keep working; the preferred surface is /conversations.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core import conversations
from app.core.db import get_db
from app.core.dependencies import CurrentUser, require_csrf

router = APIRouter()

MAX_SESSION_ID_LEN = 64


def _validate(session_id: str) -> str:
    sid = session_id.strip()
    if not (1 <= len(sid) <= MAX_SESSION_ID_LEN):
        raise HTTPException(
            status_code=422,
            detail=f"session_id must be 1-{MAX_SESSION_ID_LEN} characters",
        )
    return sid


@router.get("/sessions/{session_id}/history")
def history(session_id: str, user: CurrentUser, db: Session = Depends(get_db)):
    sid = _validate(session_id)
    if conversations.get_conversation(db, user.id, sid) is None:
        raise HTTPException(status_code=404, detail="No session found with that id")
    messages = conversations.get_messages(db, user.id, sid)
    turns = [{"role": m.role, "content": m.content} for m in messages]
    if not turns:
        raise HTTPException(status_code=404, detail="No session found with that id")
    return {"session_id": sid, "history": turns}


@router.delete("/sessions/{session_id}", dependencies=[Depends(require_csrf)])
def delete_session(session_id: str, user: CurrentUser, db: Session = Depends(get_db)):
    sid = _validate(session_id)
    deleted = conversations.delete_conversation(db, user.id, sid)
    if not deleted:
        raise HTTPException(status_code=404, detail="No session found with that id")
    return {"session_id": sid, "deleted": True}
