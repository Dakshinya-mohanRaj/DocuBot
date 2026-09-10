# DocuBot 🤖📄

A **document-grounded RAG chatbot API**. Ingest your own documents, then ask questions —
DocuBot answers strictly from that content and cites which document it pulled from, with
full multi-turn conversation memory per session.

Built with **FastAPI**, **ChromaDB**, local **sentence-transformers** embeddings, and
**Claude** for generation.

## Why this exists

Most "AI chatbot" demos are just a thin wrapper around a chat API. DocuBot instead shows
a real retrieval pipeline: chunking → embedding → vector search → grounded generation →
session memory — the core pattern behind most production RAG systems.

## Features

- 📥 Ingest raw text or `.txt`/`.md` files — automatically chunked and embedded
- 🔍 Semantic search over ingested docs (local embeddings, no extra API key needed)
- 💬 Multi-turn chat with per-session conversation history
- 📎 Every answer includes its source chunks, so you can verify grounding
- 🚫 Model is instructed to say "I don't know" rather than hallucinate when context is missing

## Quickstart

```bash
git clone <your-repo-url>
cd docubot
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# add your ANTHROPIC_API_KEY to .env

uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive Swagger docs.

## API Walkthrough

**1. Ingest a document**

```bash
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "about_docubot", "text": "DocuBot is a RAG chatbot..."}'
```

Or upload the included sample file:

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

## Architecture

```
Client
  │
  ├─ POST /ingest/text or /ingest/file
  │     └─ chunk_text() → embed (sentence-transformers) → store in ChromaDB
  │
  └─ POST /chat
        └─ embed question → retrieve top-k chunks from ChromaDB
              → build prompt with context + session history
                    → Claude generates grounded answer
                          → append turn to session store → return answer + sources
```

## Project structure

```
docubot/
├── app/
│   ├── main.py                # FastAPI app + router wiring
│   ├── core/
│   │   ├── chunking.py        # text splitting
│   │   ├── vector_store.py    # ChromaDB + embeddings
│   │   ├── llm.py             # Claude API calls
│   │   └── session_store.py   # in-memory conversation history
│   └── routes/
│       ├── ingest.py
│       ├── chat.py
│       └── sessions.py
├── data/sample_docs/          # sample doc to test with
├── requirements.txt
└── .env.example
```

## Known limitations (intentional — this is a portfolio-scoped project)

- Session history is in-memory only (resets on restart) — swap in SQLite/Redis for persistence
- Only `.txt`/`.md` file ingestion out of the box — PDF support is a natural next step
- No auth — add an API key middleware before deploying publicly

## Possible extensions

- PDF ingestion via `pypdf`
- Streaming responses (SSE) for the `/chat` endpoint
- Swap ChromaDB for a hosted vector DB (Pinecone/Qdrant) for multi-user deployments
- Add a minimal frontend to demo it live
