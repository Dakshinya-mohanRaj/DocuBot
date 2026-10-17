"""In-memory conversation store.

Persistence: **none**. Sessions live in this process's memory and are lost when
the server restarts, redeploys, or scales to a second instance. This is
documented in the README. Swapping in SQLite/Redis means implementing the same
four functions below against a persistent backend.
"""

import os
from datetime import datetime, timezone

# NOTE: in-memory only — resets when the server restarts.
_sessions: dict[str, list[dict]] = {}

# Hard cap on stored turns per session. The LLM only ever sees the last few
# (see chat.MAX_HISTORY_TURNS), but without a cap a long-lived session would
# grow without bound. Oldest turns are dropped first.
MAX_STORED_TURNS = int(os.environ.get("DOCUBOT_MAX_STORED_TURNS", "40"))


def get_history(session_id: str) -> list[dict]:
    return _sessions.get(session_id, [])


def session_exists(session_id: str) -> bool:
    return bool(_sessions.get(session_id))


def append_turn(session_id: str, role: str, content: str) -> None:
    turns = _sessions.setdefault(session_id, [])
    turns.append(
        {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    if len(turns) > MAX_STORED_TURNS:
        del turns[: len(turns) - MAX_STORED_TURNS]


def clear_session(session_id: str) -> bool:
    return _sessions.pop(session_id, None) is not None


def as_llm_messages(session_id: str) -> list[dict]:
    """Strip timestamps for the LLM call — it only needs role + content."""
    return [{"role": t["role"], "content": t["content"]} for t in get_history(session_id)]
