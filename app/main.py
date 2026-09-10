from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from app.routes import ingest, chat, sessions

app = FastAPI(
    title="DocuBot",
    description="A document-grounded RAG chatbot API. Ingest docs, then ask questions grounded in their content.",
    version="0.1.0",
)

app.include_router(ingest.router, tags=["Ingest"])
app.include_router(chat.router, tags=["Chat"])
app.include_router(sessions.router, tags=["Sessions"])


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
