import os

from dotenv import load_dotenv

load_dotenv()

import logging
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.logging_config import log_event, setup_logging
from app.core.rate_limit import RateLimitMiddleware, build_rules
from app.routes import ingest, chat, sessions

setup_logging()
logger = logging.getLogger("docubot.main")

app = FastAPI(
    title="DocuBot",
    description="A document-grounded RAG chatbot API. Ingest docs, then ask questions grounded in their content.",
    version="0.1.0",
)

# CORS: same-origin UI needs none. Cross-origin consumers must be allow-listed
# explicitly via DOCUBOT_CORS_ORIGINS (comma-separated). Default = no CORS
# headers, so no third-party site can call this API from a browser.
_cors_origins = [
    origin.strip()
    for origin in os.environ.get("DOCUBOT_CORS_ORIGINS", "").split(",")
    if origin.strip()
]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

_rate_rules = build_rules()
if _rate_rules:
    app.add_middleware(RateLimitMiddleware, rules=_rate_rules)

app.include_router(ingest.router, tags=["Ingest"])
app.include_router(chat.router, tags=["Chat"])
app.include_router(sessions.router, tags=["Sessions"])

static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
def read_root():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Welcome to DocuBot API! Visit /docs for Swagger UI."}


@app.get("/health", tags=["Health"])
def health():
    """Liveness probe — must never touch the embedding model or the LLM.

    Returns 200 as soon as the process can serve requests. Render uses this for
    its health check, so it stays intentionally trivial.
    """
    return {"status": "ok"}


@app.get("/ready", tags=["Health"])
def ready(response: Response):
    """Readiness probe — reports real dependency status without any secrets.

    Unlike /health this inspects the vector store and LLM configuration. It does
    NOT make a network LLM call: the embedding model is checked via a cheap
    in-process encode (already warm at startup) and the LLM is reported from its
    last known state. A component is only ever reported unavailable when it
    genuinely is. Returns 503 when a dependency is not usable.
    """
    from app.core import llm as llm_module
    from app.core import vector_store

    checks: dict[str, dict] = {}
    ok = True

    try:
        count = vector_store.collection_count()
        checks["vector_store"] = {"ready": True, "chunks": count}
    except Exception as exc:
        ok = False
        log_event(logger, "ready_vector_store_failed", level=logging.ERROR, error=type(exc).__name__)
        checks["vector_store"] = {"ready": False, "error": "unavailable"}

    try:
        # Cheap in-process probe: the model is loaded and warmed at import time.
        vector_store._embedding_fn(["ready-probe"])
        checks["embeddings"] = {"ready": True, "model": vector_store._EMBEDDING_MODEL}
    except Exception as exc:
        ok = False
        log_event(logger, "ready_embeddings_failed", level=logging.ERROR, error=type(exc).__name__)
        checks["embeddings"] = {"ready": False, "error": "unavailable"}

    status = llm_module.llm_status()
    checks["llm"] = status
    # "configured" means key present but not yet exercised — honest, not "ready".
    if status["status"] in ("unconfigured", "unavailable"):
        ok = False

    if not ok:
        response.status_code = 503
    return {"ready": ok, "checks": checks}
