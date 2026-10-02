import csv
import io
import uuid
from typing import Any

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.core.chunking import chunk_text
from app.core.vector_store import add_chunks, collection_count, delete_document

# Optional heavy dependencies — only required for their respective file types.
# Imported at module level so linters can resolve them; missing installs are
# caught at the point of use with a clear 400 error message.
_pypdf: Any = None
try:
    import pypdf as _pypdf
except ImportError:
    pass

_docx: Any = None
try:
    import docx as _docx
except ImportError:
    pass

_openpyxl: Any = None
try:
    import openpyxl as _openpyxl
except ImportError:
    pass


router = APIRouter()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


class IngestTextRequest(BaseModel):
    doc_id: str | None = None
    text: str


@router.post("/ingest/text")
def ingest_text(payload: IngestTextRequest) -> dict:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    doc_id = payload.doc_id or str(uuid.uuid4())[:8]
    chunks = chunk_text(payload.text)
    added = add_chunks(doc_id, chunks)

    return {"doc_id": doc_id, "chunks_added": added, "total_chunks_in_store": collection_count()}


def parse_file_content(filename: str, raw: bytes) -> str:
    lower = filename.lower()

    # 1. PDF Documents (.pdf)
    if lower.endswith(".pdf"):
        if _pypdf is None:
            raise HTTPException(status_code=400, detail="PDF parsing requires 'pypdf'. Run: pip install pypdf")
        try:
            reader = _pypdf.PdfReader(io.BytesIO(raw))
            extracted = [t for page in reader.pages if (t := page.extract_text())]
            return "\n\n".join(extracted)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF file: {exc}") from exc

    # 2. Word Documents (.docx)
    elif lower.endswith(".docx"):
        if _docx is None:
            raise HTTPException(status_code=400, detail="Word parsing requires 'python-docx'. Run: pip install python-docx")
        try:
            document = _docx.Document(io.BytesIO(raw))
            return "\n".join(p.text for p in document.paragraphs if p.text.strip())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse Word file: {exc}") from exc

    # 3. Excel Spreadsheets (.xlsx)
    elif lower.endswith(".xlsx"):
        if _openpyxl is None:
            raise HTTPException(status_code=400, detail="Excel parsing requires 'openpyxl'. Run: pip install openpyxl")
        try:
            wb = _openpyxl.load_workbook(io.BytesIO(raw), data_only=True)
            lines: list[str] = []
            for sheet in wb.worksheets:
                lines.append(f"Sheet: {sheet.title}")
                for row in sheet.iter_rows(values_only=True):
                    row_str = [str(cell) if cell is not None else "" for cell in row]
                    if any(row_str):
                        lines.append(" | ".join(row_str))
            return "\n".join(lines)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse Excel file: {exc}") from exc

    # 4. CSV Files (.csv)
    elif lower.endswith(".csv"):
        try:
            decoded = raw.decode("utf-8", errors="ignore")
            reader = csv.reader(io.StringIO(decoded))
            return "\n".join(" | ".join(row) for row in reader if any(row))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {exc}") from exc

    # 5. Text & Markdown (.txt, .md, .json, .log)
    elif lower.endswith((".txt", ".md", ".json", ".log")):
        return raw.decode("utf-8", errors="ignore")

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{filename}'. Supported: .pdf, .docx, .xlsx, .csv, .txt, .md",
        )


@router.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)) -> dict:
    try:
        raw = await file.read()

        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({len(raw) // (1024 * 1024)} MB). Maximum allowed size is 10 MB.",
            )

        filename: str = file.filename or "unknown"
        text = parse_file_content(filename, raw)

        if not text.strip():
            raise HTTPException(status_code=400, detail="Uploaded file contained no extractable text.")

        chunks = chunk_text(text)
        added = add_chunks(filename, chunks)

        return {"doc_id": filename, "chunks_added": added, "total_chunks_in_store": collection_count()}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to ingest file: {type(exc).__name__}: {exc}"
        ) from exc


@router.delete("/ingest/{doc_id:path}")
def delete_ingested_doc(doc_id: str) -> dict:
    deleted_count = delete_document(doc_id)
    return {
        "message": f"Successfully deleted document '{doc_id}'",
        "chunks_deleted": deleted_count,
        "total_chunks_in_store": collection_count(),
    }
