# DocuBot 🤖📄

A **document-grounded RAG chatbot**. Upload your own documents (PDF, Word, Excel, CSV, or plain text), then ask
questions — DocuBot answers **strictly from that content**, cites which document it pulled from, and keeps
multi-turn conversation memory per session.

## 🚀 Live Demo

> **Try it: [https://docubot-5fy3.onrender.com](https://docubot-5fy3.onrender.com)
>
> API docs: [https://docubot-5fy3.onrender.com/docs](https://docubot-5fy3.onrender.com/docs)

> ⚠️ On Render's free tier the instance sleeps after ~15 min idle. The first request after waking can take
> ~30–60 s while the process starts and loads the local embedding model. The UI shows a "warming up" badge
> and disables upload/chat until `/health` responds.

> ⚠️ **Uploaded documents do not survive a restart on the free tier.** Render provides an ephemeral
> filesystem and `render.yaml` declares no persistent disk, so `data/chroma` is wiped on every deploy,
> spin-down, or crash. Re-upload after a restart, or enable a persistent disk (see Render deployment below).

## ✨ Key Features

- 📥 **Multi-format ingestion** — PDF (`.pdf`), Word (`.docx`), Excel (`.xlsx`), CSV (`.csv`), and text/`.md`
- 🔍 **Semantic search** — documents are chunked and embedded locally (no extra API key for embeddings)
- 💬 **Multi-turn chat** — per-session conversation history
- 📎 **Grounded answers with sources** — every answer cites the document chunks it used
- 🚫 **No hallucination** — the model abstains with an explicit message when the documents don't answer the question
- 🖥️ **Single-page web UI** served at the root route, responsive down to mobile

## 🛠️ Technologies Used

| Layer          | Technology                                                        |
| -------------- | ----------------------------------------------------------------- |
| Backend API    | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn                |
| Vector store   | [ChromaDB](https://www.trychroma.com/)                            |
| Embeddings     | `sentence-transformers` (`all-MiniLM-L6-v2`, local, CPU)          |
| LLM (inference)| [Groq](https://groq.com) API (`openai/gpt-oss-20b`, override with `GROQ_MODEL`) |
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
# edit .env and set GROQ_API_KEY

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** for the web UI, or **http://localhost:8000/docs** for Swagger.

## 📚 API Walkthrough

**1. Upload a document**

```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@data/sample_docs/about_docubot.md"
```

Response:

```json
{
  "doc_id": "about_docubot.md",
  "chunks_added": 4,
  "replaced": false,
  "total_chunks_in_store": 4
}
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
  "answer": "DocuBot embeds your question and retrieves the most similar chunks... [about_docubot.md]",
  "sources": [
    {
      "doc_id": "about_docubot.md",
      "filename": "about_docubot.md",
      "similarity": 0.62,
      "relevance": 0.81,
      "chunk_index": 1,
      "page": null
    }
  ],
  "abstained": false
}
```

**3. Continue the conversation** — pass the returned `session_id`

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "And what embedding model does it use?", "session_id": "a1b2c3d4"}'
```

**4. Manage the knowledge base and sessions**

```bash
curl http://localhost:8000/ingest/documents              # list documents
curl -X DELETE http://localhost:8000/ingest/about_docubot.md
curl http://localhost:8000/sessions/a1b2c3d4/history     # session turns
curl -X DELETE http://localhost:8000/sessions/a1b2c3d4   # clear session
```

### Endpoints

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `GET`  | `/` | Single-page web UI |
| `GET`  | `/health` | Liveness probe — always cheap, no model/LLM work |
| `GET`  | `/ready` | Readiness — real vector-store/embedding/LLM status, no secrets; 503 when a dependency is unusable |
| `POST` | `/ingest/file` | Upload `.pdf`/`.docx`/`.xlsx`/`.csv`/`.txt`/`.md` (≤ 10 MB) |
| `POST` | `/ingest/text` | Ingest raw text (`{"text": "...", "doc_id": "optional"}`) |
| `GET`  | `/ingest/documents` | List documents with chunk counts |
| `DELETE` | `/ingest/{doc_id}` | Delete a document (404 if absent) |
| `POST` | `/chat` | Ask a question (`{"message", "session_id"?, "top_k"?}`) |
| `GET`  | `/sessions/{id}/history` | Conversation turns for a session |
| `DELETE` | `/sessions/{id}` | Clear a session |

### About the `similarity` score

`similarity` is the **cosine similarity** between the question embedding and the chunk embedding, in `[0, 1]`.
It is a *relative closeness* signal, **not a probability or a percent-confidence**. The UI displays it as
`similarity 0.62`, not `62%`. Retrieval ranks by a combined score (`relevance`) that also weights lexical
term overlap; `DOCUBOT_MIN_RELEVANCE` is the floor below which chunks are discarded so the bot abstains
instead of citing an unrelated document.

## 🧠 Architecture

```
Browser (SPA) / curl
        │
        ├─ POST /ingest/file or /ingest/text
        │     └─ parse (pypdf/docx/openpyxl) → chunk_text() → embed (sentence-transformers)
        │           → write new chunk generation to ChromaDB → drop the old generation
        │
        └─ POST /chat
              └─ embed question → retrieve + filter top-k chunks from ChromaDB
                    → build prompt with context + capped session history
                          → Groq generates grounded answer
                                → append turn to session store → return answer + sources
```

## 📁 Project Structure

```
docubot/
├── app/
│   ├── main.py                # FastAPI app, /health, /ready, CORS, rate limiting
│   ├── core/
│   │   ├── chunking.py        # sentence-aware text splitting
│   │   ├── vector_store.py    # ChromaDB + sentence-transformers (generation-safe writes)
│   │   ├── llm.py             # Groq client, prompt, timeouts, sanitized errors
│   │   ├── session_store.py   # in-memory conversation history
│   │   ├── logging_config.py  # structured, secret-free logging
│   │   └── rate_limit.py      # in-memory per-IP token bucket
│   └── routes/
│       ├── ingest.py
│       ├── chat.py
│       └── sessions.py
├── static/index.html          # single-page web UI
├── scripts/
│   ├── fixtures.py            # three-document evaluation corpus
│   └── eval_rag.py            # retrieval + answer-quality evaluation
├── tests/                     # pytest suite (no live API calls)
├── data/sample_docs/          # sample doc to test with
├── Dockerfile / render.yaml   # deploy config
├── requirements.txt / requirements-dev.txt
└── .env.example
```

## 🧪 Testing

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q          # 80 tests, no network/API calls
```

The suite stubs the LLM by default (an autouse fixture) so **no test spends a paid API call**.

A separate, optional evaluation runs against the **live** LLM (costs Groq calls):

```bash
python scripts/eval_rag.py                                  # local, in-process
python scripts/eval_rag.py --base-url https://docubot-5fy3.onrender.com
```

It uploads a three-document corpus with deliberately non-overlapping numbers and scores answers across
`single_doc`, `cross_doc`, `unanswerable`, and `contamination` categories.

## ⚙️ Configuration

All variables are optional except `GROQ_API_KEY`. See `.env.example`.

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `GROQ_API_KEY` | — | **Required.** Groq API key. |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Chat model, validated against your account on first use. |
| `DOCUBOT_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model. Changing it **requires clearing `data/chroma` and re-ingesting**. |
| `DOCUBOT_MIN_RELEVANCE` | `0.30` | Combined relevance floor; below it `/chat` abstains. |
| `DOCUBOT_LEXICAL_WEIGHT` | `0.5` | Weight of lexical overlap vs cosine similarity. |
| `DOCUBOT_PER_DOC_CAP` | `2` | Max chunks one document may contribute per answer. |
| `DOCUBOT_CHROMA_PATH` | `./data/chroma` | Vector store location. |
| `DOCUBOT_COLLECTION` | `docubot_chunks_v2` | Chroma collection name. |
| `DOCUBOT_LLM_TIMEOUT` | `45` | Per-request LLM timeout (seconds). |
| `DOCUBOT_LLM_RETRIES` | `2` | LLM client retries on transient failures. |
| `DOCUBOT_MAX_STORED_TURNS` | `40` | Stored turns per session before oldest are dropped. |
| `DOCUBOT_RATE_LIMIT_CHAT` | `20` | Per-IP `/chat` requests per minute (`0` disables). |
| `DOCUBOT_RATE_LIMIT_INGEST` | `10` | Per-IP `/ingest` requests per minute (`0` disables). |
| `DOCUBOT_CORS_ORIGINS` | *(empty)* | Comma-separated allowed origins. Empty = same-origin only. |
| `DOCUBOT_LOG_LEVEL` | `INFO` | Root log level. |

## ☁️ Render Deployment

- **Runtime:** Docker (see `Dockerfile`), Python 3.12.
- **Start command** (in the Dockerfile, honors Render's port):
  `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}`
- **Health check path:** `/health`
- **Required env var:** `GROQ_API_KEY` (set as a secret in the Render dashboard; never committed).

### Persistence

| Environment | Vector store (`data/chroma`) | Sessions |
| ----------- | ---------------------------- | -------- |
| Local dev | Persists on disk | In-memory, lost on restart |
| Render **free** | **Ephemeral** — lost on deploy/spin-down/restart | In-memory, lost on restart |
| Render + **persistent disk** | Persists | Still in-memory unless you add a DB |

To keep the knowledge base on Render, upgrade to a paid instance and uncomment the `disk:` block plus the
`DOCUBOT_CHROMA_PATH` variable in `render.yaml` (mount path `/var/data`, store path `/var/data/chroma`).

## 🧯 Troubleshooting

**HTTP 503 immediately after deploy / on first request**
- Most common cause: **cold start**. The free instance is asleep; the first request waits while the process
  imports torch and loads the embedding model. `/health` returns as soon as the app is up; the UI waits for it.
- Check the Render deploy log for a completed build and `Uvicorn running on ...`.
- If the service **crash-loops**, suspect memory (`OOM`): the free plan has 512 MB, and torch +
  sentence-transformers + ChromaDB are close to that ceiling. The Dockerfile bakes the embedding model into
  the image to avoid a runtime download; if it still OOMs, upgrade the instance type.
- Confirm `GROQ_API_KEY` is set in **Environment**; without it `/ready` reports the LLM as `unconfigured`.

**Chat returns 503 / "answer service is temporarily unavailable"**
- Groq is unreachable, rate-limited, or the configured `GROQ_MODEL` isn't available to your account.
  Check `GET /ready` → `checks.llm.status`, and the Render logs (`llm_call_failed` / `llm_model_unavailable`).

**Upload fails**
- 413: file exceeds 10 MB. 400: unsupported type, empty file, or a scanned/image-only PDF (no extractable
  text — OCR is not supported). Check the message the UI shows.

**Documents disappeared**
- Expected on the Render free tier after any restart. Enable a persistent disk (above) to prevent it.

## ⚠️ Known Limitations

- Session history is **in-memory only** — it does not survive a restart and is not shared between instances.
- On Render's free tier the filesystem is **ephemeral** (see Persistence).
- Rate limiting is **per-instance, in-memory** — it resets on restart and does not coordinate across instances.
- No authentication. Put an API-key middleware in front before exposing it beyond a demo.
- Corpus-wide aggregation questions ("list every limit across all documents") are answered from whichever
  documents score above the relevance floor; the floor deliberately prefers abstaining over padding context
  with irrelevant passages.

## 🔭 Possible Extensions

- Persistent storage (SQLite/Redis for sessions, a hosted vector DB for documents)
- Streaming responses (SSE) for `/chat`
- Authentication + per-user collections
