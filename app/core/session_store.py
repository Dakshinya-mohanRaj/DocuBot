from datetime import datetime

# NOTE: in-memory only — resets when the server restarts.
# Fine for a demo project; swap for SQLite/Redis if you want persistence.
_sessions: dict[str, list[dict]] = {}


def get_history(session_id: str) -> list[dict]:
    return _sessions.get(session_id, [])


def append_turn(session_id: str, role: str, content: str) -> None:
    _sessions.setdefault(session_id, []).append(
        {"role": role, "content": content, "timestamp": datetime.utcnow().isoformat()}
    )


def clear_session(session_id: str) -> bool:
    return _sessions.pop(session_id, None) is not None


def as_llm_messages(session_id: str) -> list[dict]:
    """Strip timestamps for the LLM call — it only needs role + content."""
    return [{"role": t["role"], "content": t["content"]} for t in get_history(session_id)]
