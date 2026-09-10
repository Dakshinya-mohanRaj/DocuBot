import csv
import io
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


def parse_file_content(filename: str, raw: bytes) -> str:
    lower = filename.lower()

    # 1. PDF Documents (.pdf)
    if lower.endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(raw))
            extracted = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    extracted.append(t)
            return "\n\n".join(extracted)
        except ImportError:
            raise HTTPException(
                status_code=400,
                detail="PDF parsing requires 'pypdf'. Please run: pip install pypdf"
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF file: {str(e)}")

    # 2. Word Documents (.docx)
    elif lower.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            raise HTTPException(
                status_code=400,
                detail="Word (.docx) parsing requires 'python-docx'. Please run: pip install python-docx"
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse Word file: {str(e)}")

    # 3. Excel Spreadsheets (.xlsx)
    elif lower.endswith(".xlsx"):
        try:
            import openpyxl
            wb = openpyxl.load_workbook(io.BytesIO(raw), data_only=True)
            lines = []
            for sheet in wb.worksheets:
                lines.append(f"Sheet: {sheet.title}")
                for row in sheet.iter_rows(values_only=True):
                    row_str = [str(cell) if cell is not None else "" for cell in row]
                    if any(row_str):
                        lines.append(" | ".join(row_str))
            return "\n".join(lines)
        except ImportError:
            raise HTTPException(
                status_code=400,
                detail="Excel (.xlsx) parsing requires 'openpyxl'. Please run: pip install openpyxl"
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse Excel file: {str(e)}")

    # 4. CSV Files (.csv)
    elif lower.endswith(".csv"):
        try:
            decoded = raw.decode("utf-8", errors="ignore")
            reader = csv.reader(io.StringIO(decoded))
            return "\n".join(" | ".join(row) for row in reader if any(row))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")

    # 5. Text & Markdown (.txt, .md)
    elif lower.endswith((".txt", ".md", ".json", ".log")):
        return raw.decode("utf-8", errors="ignore")

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{filename}'. Supported formats: .pdf, .docx, .xlsx, .csv, .txt, .md"
        )


@router.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    raw = await file.read()
    text = parse_file_content(file.filename, raw)

    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file contained no extractable text.")

    doc_id = file.filename
    chunks = chunk_text(text)
    added = add_chunks(doc_id, chunks)

    return {"doc_id": doc_id, "chunks_added": added, "total_chunks_in_store": collection_count()}
