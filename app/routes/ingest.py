import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.core.chunking import chunk_text
from app.core.vector_store import add_chunks, collection_count

router = APIRouter()


class IngestTextRequest(BaseModel):
    doc_id: str | None = None
    text: str


@router.post("/ingest/text")
def ingest_text(payload: IngestTextRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    doc_id = payload.doc_id or str(uuid.uuid4())[:8]
    chunks = chunk_text(payload.text)
    added = add_chunks(doc_id, chunks)

    return {"doc_id": doc_id, "chunks_added": added, "total_chunks_in_store": collection_count()}


@router.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    if not file.filename.endswith((".txt", ".md")):
        raise HTTPException(
            status_code=400, detail="Only .txt and .md supported in this starter — add PDF parsing yourself"
        )

    raw = await file.read()
    text = raw.decode("utf-8", errors="ignore")

    doc_id = file.filename
    chunks = chunk_text(text)
    added = add_chunks(doc_id, chunks)

    return {"doc_id": doc_id, "chunks_added": added, "total_chunks_in_store": collection_count()}
