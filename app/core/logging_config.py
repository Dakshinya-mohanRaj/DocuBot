"""Central logging setup.

Every log line is a single, greppable ``event`` with ``key=value`` context. The
helpers below never accept or emit secrets: callers pass only identifiers such
as doc_id, session_id, counts, and elapsed time — never document text, API keys,
or raw provider payloads.
"""

import logging
import os
import sys

_LEVEL = os.environ.get("DOCUBOT_LOG_LEVEL", "INFO").upper()

_configured = False


def setup_logging() -> None:
    """Install one stderr handler for the whole process (idempotent)."""
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(_LEVEL)

    # Uvicorn installs its own handlers; route them through ours so deploy logs
    # share one format and one destination.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers.clear()
        logging.getLogger(name).propagate = True

    _configured = True


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields) -> None:
    """Emit one structured line: ``event key=value key=value``.

    Values are coerced to short strings. Long values are truncated so an
    accidental document fragment can never flood the logs.
    """
    parts = [event]
    for key, value in fields.items():
        if value is None:
            continue
        text = str(value).replace("\n", " ")
        if len(text) > 120:
            text = text[:117] + "..."
        parts.append(f"{key}={text}")
    logger.log(level, " ".join(parts))
