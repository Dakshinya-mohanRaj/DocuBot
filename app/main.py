from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.routes import ingest, chat, sessions

app = FastAPI(
    title="DocuBot",
    description="A document-grounded RAG chatbot API. Ingest docs, then ask questions grounded in their content.",
    version="0.1.0",
)

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
    return {"message": "Welcome to DocuBot API! Visit /docs for Swagger UI UI."}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
