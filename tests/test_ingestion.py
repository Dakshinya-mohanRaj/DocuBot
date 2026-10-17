"""File and raw-text ingestion behaviour."""

import io

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.core import vector_store


def make_pdf(text: str = "The secret code is 42.") -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject()
    font.update(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    resources = DictionaryObject()
    resources.update({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})})
    page[NameObject("/Resources")] = resources
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")] = writer._add_object(stream)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _upload(client, name, data: bytes, content_type="application/octet-stream"):
    return client.post("/ingest/file", files={"file": (name, data, content_type)})


def test_valid_pdf_is_ingested_with_page_metadata(client):
    response = _upload(client, "report.pdf", make_pdf())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["chunks_added"] >= 1
    assert body["replaced"] is False

    stored = vector_store._collection.get(where={"doc_id": "report.pdf"})
    pages = [m.get("page") for m in stored["metadatas"]]
    assert pages and all(p == 1 for p in pages)


def test_pdf_with_no_text_is_rejected(client):
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)

    response = _upload(client, "scan.pdf", buf.getvalue())
    assert response.status_code == 400
    assert "no extractable text" in response.json()["detail"].lower()


def test_malformed_pdf_is_rejected(client):
    response = _upload(client, "broken.pdf", b"%PDF-1.4 this is not a real pdf")
    assert response.status_code == 400


def test_markdown_and_txt_are_ingested(client):
    md = _upload(client, "notes.md", b"# Title\n\nThe budget is 500 dollars.")
    assert md.status_code == 200
    assert md.json()["chunks_added"] >= 1

    txt = _upload(client, "plain.txt", b"The budget is 500 dollars.")
    assert txt.status_code == 200


def test_empty_file_is_rejected(client):
    response = _upload(client, "empty.txt", b"")
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_unsupported_extension_is_rejected(client):
    response = _upload(client, "evil.exe", b"MZ")
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]


def test_oversized_file_is_rejected(client):
    payload = b"a" * (10 * 1024 * 1024 + 1)
    response = _upload(client, "big.txt", payload)
    assert response.status_code == 413


def test_filename_path_traversal_is_sanitized(client):
    response = _upload(client, "../../etc/passwd.txt", b"root:x:0:0")
    assert response.status_code == 200
    doc_id = response.json()["doc_id"]
    assert ".." not in doc_id and "/" not in doc_id
    assert doc_id == "passwd.txt"


def test_delete_unknown_document_returns_404(client):
    assert client.delete("/ingest/does-not-exist.txt").status_code == 404


def test_delete_known_document_then_404(client):
    _upload(client, "a.txt", b"some content")
    first = client.delete("/ingest/a.txt")
    assert first.status_code == 200
    assert first.json()["chunks_deleted"] >= 1
    assert client.delete("/ingest/a.txt").status_code == 404


def test_raw_text_ingest_is_retrievable(client):
    response = client.post(
        "/ingest/text",
        json={"doc_id": "snippet", "text": "The office wifi password is hunter2."},
    )
    assert response.status_code == 200
    assert response.json()["doc_id"] == "snippet"

    chat = client.post("/chat", json={"message": "What is the office wifi password?"})
    assert chat.status_code == 200
    body = chat.json()
    assert body["abstained"] is False
    assert any(s["doc_id"] == "snippet" for s in body["sources"])


def test_reupload_replaces_rather_than_duplicates(client):
    _upload(client, "a.txt", b"The rate limit is 100 per minute.")
    second = _upload(client, "a.txt", b"The rate limit is now 250 per minute.")
    assert second.json()["replaced"] is True
    documents = client.get("/ingest/documents").json()["documents"]
    assert len(documents) == 1
