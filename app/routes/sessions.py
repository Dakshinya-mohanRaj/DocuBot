from fastapi import APIRouter, HTTPException
from app.core.session_store import get_history, clear_session

router = APIRouter()


@router.get("/sessions/{session_id}/history")
def history(session_id: str):
    turns = get_history(session_id)
    if not turns:
        raise HTTPException(status_code=404, detail="No session found with that id")
    return {"session_id": session_id, "history": turns}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    deleted = clear_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="No session found with that id")
    return {"session_id": session_id, "deleted": True}
