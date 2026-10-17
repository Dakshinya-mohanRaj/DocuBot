from fastapi import APIRouter, HTTPException

from app.core.session_store import clear_session, get_history

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
def history(session_id: str):
    sid = _validate(session_id)
    turns = get_history(sid)
    if not turns:
        raise HTTPException(status_code=404, detail="No session found with that id")
    return {"session_id": sid, "history": turns}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    sid = _validate(session_id)
    deleted = clear_session(sid)
    if not deleted:
        raise HTTPException(status_code=404, detail="No session found with that id")
    return {"session_id": sid, "deleted": True}
