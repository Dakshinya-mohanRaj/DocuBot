# DocuBot 🤖📄

A **document-grounded RAG chatbot**. Upload your own documents (PDF, Word, Excel, CSV, or plain text), then ask questions —
DocuBot answers **strictly from that content**, cites which document it pulled from, and keeps full multi-turn
conversation memory per session.

## 🚀 Live Demo

> **Try it now: [https://docubot-7oun.onrender.com](https://docubot-7oun.onrender.com)**

Hosted on **Render** with a modern single-page web UI (upload → chat in seconds). API docs are live at
[https://docubot-7oun.onrender.com/docs](https://docubot-7oun.onrender.com/docs).

> ⚠️ The free-tier instance sleeps after ~15 min of inactivity — the first request after waking can take ~30–60 s.

## ✨ Key Features

- 📥 **Multi-format ingestion** — PDF (`.pdf`), Word (`.docx`), Excel (`.xlsx`), CSV (`.csv`), and text/`.md` files
- 🔍 **Semantic search** — documents are chunked and embedded locally (no extra API key for embeddings)
- 💬 **Multi-turn chat** — per-session conversation history
- 📎 **Grounded answers with sources** — every answer cites the document chunks it used, so you can verify it
- 🚫 **No hallucination** — the model is instructed to say it doesn't know rather than guess when context is missing
- 🖥️ **Single-page web UI** served at the root route

## 🛠️ Technologies Used

| Layer          | Technology                                                        |
| -------------- | ----------------------------------------------------------------- |
| Backend API    | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn                |
| Vector store   | [ChromaDB](https://www.trychroma.com/)                            |
| Embeddings     | `sentence-transformers` (`all-MiniLM-L6-v2`, local, CPU)          |
| LLM (inference)| [Groq](https://groq.com) API (`qwen/qwen3.6-27b`)                 |
| Document parse | `pypdf`, `python-docx`, `openpyxl`                                |
| Frontend       | Vanilla HTML/CSS/JS (single page)                                 |
| Deploy         | [Render](https://render.com) (Docker, Python 3.12)                |

## 🚀 Quickstart (local)

```bash
git clone <your-repo-url>
cd docubot
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# add your GROQ_API_KEY to .env

uvicorn app.main:app --reload
```

Then open **http://localhost:8000** for the web UI, or **http://localhost:8000/docs** for interactive Swagger docs.

## 📚 API Walkthrough

**1. Upload a document**

```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@data/sample_docs/about_docubot.md"
```

**2. Ask a question**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How does DocuBot decide which chunks to use?"}'
```

Response:

```json
{
  "session_id": "a1b2c3d4",
  "answer": "DocuBot embeds your question and retrieves the top-k most similar chunks...",
  "sources": [
    {"doc_id": "about_docubot.md", "score": 0.31}
  ]
}
```

**3. Continue the conversation** (reuse the `session_id`)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "And what embedding model does it use?", "session_id": "a1b2c3d4"}'
```

**4. Review or clear session history**

```bash
curl http://localhost:8000/sessions/a1b2c3d4/history
curl -X DELETE http://localhost:8000/sessions/a1b2c3d4
```

## 🧠 Architecture

```
Browser (SPA) / curl
        │
        ├─ POST /ingest/file or /ingest/text
        │     └─ parse (pypdf/docx/openpyxl) → chunk_text() → embed (sentence-transformers)
        │           → store in ChromaDB
        │
        └─ POST /chat
              └─ embed question → retrieve top-k chunks from ChromaDB
                    → build prompt with context + session history
                          → Groq generates grounded answer
                                → append turn to session store → return answer + sources
```

## 📁 Project Structure

```
docubot/
├── app/
│   ├── main.py                # FastAPI app + router wiring
│   ├── core/
│   │   ├── chunking.py        # text splitting
│   │   ├── vector_store.py    # ChromaDB + sentence-transformers embeddings
│   │   ├── llm.py             # Groq API calls (with model fallbacks)
│   │   └── session_store.py   # in-memory conversation history
│   └── routes/
│       ├── ingest.py
│       ├── chat.py
│       └── sessions.py
├── static/index.html          # single-page web UI
├── data/sample_docs/          # sample doc to test with
├── Dockerfile                 # Render/Docker deploy config
├── render.yaml                # Render Blueprint
├── requirements.txt
└── .env.example
```

## ⚠️ Known Limitations

- Session history is in-memory only (resets on restart) — swap in SQLite/Redis for persistence
- On Render's free tier the filesystem is ephemeral: uploaded documents are lost when the instance sleeps/restarts, and the instance spins down after ~15 min idle
- No auth — add an API-key middleware before exposing to the public

## 🔭 Possible Extensions

- Persistent storage (SQLite/Redis for sessions, a hosted vector DB for documents)
- Streaming responses (SSE) for the `/chat` endpoint
- WebSocket-based chat for a snappier UI
- Authentication + per-user collections