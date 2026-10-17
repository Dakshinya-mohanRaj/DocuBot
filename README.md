# DocuBot 🤖📄

A **document-grounded RAG chatbot with real user accounts**. Sign up, upload your own documents (PDF, Word,
Excel, CSV, or plain text), then ask questions — DocuBot answers **strictly from your documents**, cites which
one it pulled from, and keeps per-user conversation history that survives a restart.

## 🚀 Live Demo

> **Try it: [https://docubot-5fy3.onrender.com](https://docubot-5fy3.onrender.com)
>
> API docs: [https://docubot-5fy3.onrender.com/docs](https://docubot-5fy3.onrender.com/docs)

> ⚠️ On Render's free tier the instance sleeps after ~15 min idle. The first request after waking can take
> ~30–60 s while the process starts and loads the local embedding model. The UI shows a "warming up" badge
> and disables upload/chat until `/health` responds.

> ⚠️ **On the free tier, uploaded documents do not survive a restart.** Accounts, conversations, and message
> history live in Postgres and *do* persist; only the ChromaDB vector index (`data/chroma`) is wiped because the
> filesystem is ephemeral. Re-upload after a restart, or enable a persistent disk (see Render deployment).

## ✨ Key Features

- 🔐 **Real accounts** — email + password signup/login with **Argon2id** hashing, opaque **HttpOnly session
  cookies**, and **double-submit CSRF** protection
- 👤 **Per-user isolation** — each account sees only its own documents and conversations; every read/write is
  ownership-checked server-side (never trusted from the client)
- 💬 **Multiple conversations** — a sidebar of recent chats you can open, rename, and delete; history is stored
  in the database and restored on reload
- 📥 **Multi-format ingestion** — PDF (`.pdf`), Word (`.docx`), Excel (`.xlsx`), CSV (`.csv`), and text/`.md`
- 🔍 **Semantic search** — documents are chunked and embedded locally (no extra API key for embeddings)
- 🗣️ **Conversational assistant** — greetings, small talk, "who are you?" and "what can you do?" are answered
  directly (no retrieval, no LLM cost, and they work even with an empty knowledge base)
- 🧭 **Intent routing** — a deterministic layer classifies each message as conversation, document Q&A, or
  document analysis before any expensive work happens
- 📊 **Document analysis** — counting and demographic questions ("how many male/female candidates?") are
  computed in **Python from LLM-extracted records**, so tallies and percentages are deterministic, with the
  denominator stated explicitly
- 💬 **Multi-turn chat** — per-conversation history, persisted
- 📎 **Grounded answers with sources** — every answer cites the document chunks it used
- 🚫 **No hallucination** — the model abstains with an explicit message when the documents don't answer the question
- 🖥️ **Redesigned web UI** — ChatGPT-style sidebar (recent chats, your documents, account) and a floating
  composer with a `+` upload menu, responsive down to mobile

## 🛠️ Technologies Used

| Layer          | Technology                                                        |
| -------------- | ----------------------------------------------------------------- |
| Backend API    | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn                |
| Database       | SQLAlchemy 2.0 over **Postgres** (prod) / SQLite (dev + tests), `psycopg` v3 |
| Passwords      | `argon2-cffi` (**Argon2id**)                                      |
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
# optional: set DOCUBOT_DATABASE_URL to a Postgres URL, or leave unset to use SQLite

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** for the web UI (sign up first), or **http://localhost:8000/docs** for Swagger.

## 🔐 Authentication (curl walkthrough)

Every chat/ingest/conversation call requires a logged-in user. The session is an HttpOnly cookie; unsafe
requests (POST/PATCH/DELETE) must also echo the readable `docubot_csrf` cookie back in the `X-CSRF-Token`
header (double-submit). The `demo.sh` pattern below uses a cookie jar.

**1. Sign up (or log in)**

```bash
curl -s -c jar -X POST http://localhost:8000/auth/signup \
  -H 'Content-Type: application/json' \
  -d '{"email":"me@example.com","password":"correct-horse-1","display_name":"Me"}'
# 201 {"user": {"id": "...", "email": "me@example.com", ...}}
```

Capture the CSRF token the browser would read from its cookie:

```bash
CSRF=$(grep docubot_csrf jar | awk '{print $NF}')
```

**2. Upload a document** (authenticated + CSRF)

```bash
curl -s -b jar -X POST http://localhost:8000/ingest/file \
  -H "X-CSRF-Token: $CSRF" \
  -F "file=@data/sample_docs/about_docubot.md"
```

**3. Ask a question**

```bash
curl -s -b jar -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' -H "X-CSRF-Token: $CSRF" \
  -d '{"message": "How does DocuBot decide which chunks to use?"}'
```

Response:

```json
{
  "session_id": "a1b2c3d4...",
  "conversation_id": "a1b2c3d4...",
  "answer": "DocuBot embeds your question and retrieves the most similar chunks... [about_docubot.md]",
  "sources": [
    { "doc_id": "about_docubot.md", "filename": "about_docubot.md", "similarity": 0.62,
      "relevance": 0.81, "chunk_index": 1, "page": null }
  ],
  "abstained": false,
  "intent": "document"
}
```

**4. Continue the conversation** — pass the returned `conversation_id` (or legacy `session_id`).

**5. Manage chats and log out**

```bash
curl -s -b jar http://localhost:8000/conversations              # recent chats
curl -s -b jar -X DELETE -H "X-CSRF-Token: $CSRF" http://localhost:8000/conversations/<id>
curl -s -b jar -X POST -H "X-CSRF-Token: $CSRF" http://localhost:8000/auth/logout
```

### Endpoints

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `GET`  | `/` | Single-page web UI |
| `GET`  | `/health` | Liveness probe — always cheap, no model/LLM work |
| `GET`  | `/ready` | Readiness — database, vector-store, embedding, and LLM status, no secrets |
| `POST` | `/auth/signup` | Create an account → sets session + CSRF cookies |
| `POST` | `/auth/login`  | Log in → sets session + CSRF cookies |
| `POST` | `/auth/logout` | Revoke the session and clear cookies (idempotent) |
| `GET`  | `/auth/me` | Current user (401 if not signed in) |
| `POST` | `/conversations` | Start a new chat thread |
| `GET`  | `/conversations` | List your chats (most recent first) |
| `GET`  | `/conversations/{id}` | One chat + its messages |
| `PATCH` | `/conversations/{id}` | Rename a chat |
| `DELETE` | `/conversations/{id}` | Delete a chat and its messages |
| `POST` | `/ingest/file` | Upload `.pdf`/`.docx`/`.xlsx`/`.csv`/`.txt`/`.md` (≤ 10 MB) |
| `POST` | `/ingest/text` | Ingest raw text (`{"text": "...", "doc_id": "optional"}`) |
| `GET`  | `/ingest/documents` | List **your** documents (`?recent=true` sorts newest-first) |
| `DELETE` | `/ingest/{doc_id}` | Delete a document (404 if absent) |
| `POST` | `/chat` | Ask (`{"message", "conversation_id"?, "top_k"?}`) |
| `GET`  | `/sessions/{id}/history` | Legacy alias for a conversation's turns |
| `DELETE` | `/sessions/{id}` | Legacy alias to clear a conversation |

### About the `similarity` score

`similarity` is the **cosine similarity** between the question embedding and the chunk embedding, in `[0, 1]`.
It is a *relative closeness* signal, **not a probability or a percent-confidence**. The UI displays it as
`similarity 0.62`, not `62%`. Retrieval ranks by a combined score (`relevance`) that also weights lexical
term overlap; `DOCUBOT_MIN_RELEVANCE` is the confident floor. Below it DocuBot still tries the closest chunks
and lets the grounded model decide whether they answer the question, so a paraphrase or a meta question
("summarize this") is not silently dropped by the threshold.

## 🧭 Intent Routing & Document Analysis

Every message is classified **before** any retrieval or LLM call:

| Intent | Example | Behaviour |
| ------ | ------- | --------- |
| `greeting` / `small_talk` / `identity` / `help` | "Hi", "How are you?", "Who are you?", "What can you do?" | Answered from a fixed reply. No retrieval, no LLM, works with an empty KB. |
| `document` | "What is the rate limit?", "Summarize this PDF." | Normal grounded RAG with source citations. |
| `analysis` | "How many male and female candidates?", "What percentage list Python?" | Deterministic record pipeline (below). |

**Retrieval is recall-first.** Chunks above `DOCUBOT_MIN_RELEVANCE` are used directly. If nothing clears that
floor, DocuBot falls back to the *closest* chunks and lets the grounded model decide — its prompt makes it say
the documents don't cover the question when they don't. This matters because a single cosine threshold
separates poorly: a legitimate paraphrase ("What is the PTO policy?" when the document says "paid time off")
can score below an unrelated question, and meta questions ("summarize this PDF") score low against every
chunk. Abstention is therefore enforced by the model, not only by the threshold; when it abstains, no sources
are shown.

Matching is deliberately conservative: only a whole message, or a message that is *only* a greeting (plus filler
like "there"), counts as conversational. `"hi, what is the PTO policy?"` starts with a greeting but carries a
real question, so it is routed to document Q&A. The response includes an `intent` field so clients can tell the
two apart; existing fields are unchanged.

**Analysis** works in two stages so numbers are not guessed:

1. The LLM extracts structured records (`name`, explicit `gender`, documented `attributes`, `source`) from the
   relevant documents — reading their **whole** content up to a token budget, not just the top passages.
2. Python computes counts, totals and percentages from those records, states how many records were examined,
   separates unknown/unspecified entries, de-duplicates entities that span multiple chunks, and flags
   contradictory statements.

**Gender is reported only when the document states it explicitly** — never inferred from names, pronouns,
photographs, or other indirect clues. Undocumented entries are listed as *unknown / not stated* and are
excluded from percentage denominators (which are stated in the answer).

## 🛡️ Security Model

- **Passwords** are hashed with Argon2id; nothing ever returns or logs a hash. Unknown-email logins still run a
  verify against a throwaway hash so response timing doesn't reveal which accounts exist.
- **Sessions** are server-side records keyed by the SHA-256 of an opaque cookie token, so a database leak does
  not hand out live sessions. Cookies are `HttpOnly`, `SameSite=Lax`, and `Secure` (configurable).
- **CSRF** uses the double-submit pattern: a readable `docubot_csrf` cookie must be echoed in the `X-CSRF-Token`
  header, and its hash is bound to the session server-side.
- **Authorization** is enforced on every route — the user identity comes from the verified session cookie, never
  from the request body, and document/conversation ids are ownership-checked (404 for another user's id, so we
  never reveal whether it exists).
- **Retrieval is tenant-scoped**: every vector query is filtered by the caller's `user_id` in ChromaDB metadata.
- **Rate limiting** is applied per IP to `/chat`, `/ingest`, and `/auth` (tighter for signup/login).

## 🧠 Architecture

```
Browser (SPA) — login/signup → session cookie
        │
        ├─ POST /auth/*          → accounts (Argon2id) → Postgres (users, auth_sessions)
        │
        ├─ POST /ingest/file or /ingest/text   [auth + CSRF]
        │     └─ parse → chunk_text() → embed → write to ChromaDB (tagged with user_id)
        │
        ├─ GET/POST/PATCH/DELETE /conversations [auth + CSRF] → Postgres (conversations, messages)
        │
        └─ POST /chat            [auth + CSRF]
              └─ resolve/create conversation (ownership-checked)
                    └─ intent detection
                          ├─ greeting/small talk/help → fixed reply (no retrieval, no LLM)
                          └─ document request
                                └─ embed question → retrieve + filter (user-scoped) from ChromaDB
                                      ├─ analysis intent → LLM extracts records → Python tallies
                                      └─ Q&A intent      → Groq generates grounded answer
                                            → persist both turns → return answer + sources
```

## 📁 Project Structure

```
docubot/
├── app/
│   ├── main.py                # FastAPI app, lifespan/init_db, /health, /ready, CORS, rate limiting
│   ├── core/
│   │   ├── db.py              # SQLAlchemy engine/session (Postgres or SQLite) + get_db
│   │   ├── models.py          # User, AuthSession, Conversation, Message
│   │   ├── security.py        # Argon2id hashing, opaque tokens, cookie config
│   │   ├── accounts.py        # user + auth-session persistence (hashing-agnostic)
│   │   ├── conversations.py   # durable per-user conversation/message store
│   │   ├── dependencies.py    # get_current_user / require_csrf FastAPI dependencies
│   │   ├── chunking.py        # sentence-aware text splitting
│   │   ├── vector_store.py    # ChromaDB + sentence-transformers (user-scoped, generation-safe writes)
│   │   ├── llm.py             # Groq client, prompts, timeouts, sanitized errors
│   │   ├── intents.py         # deterministic greeting/small-talk/help routing
│   │   ├── analysis.py        # record extraction + deterministic tallies (gender-safe)
│   │   ├── logging_config.py  # structured, secret-free logging
│   │   └── rate_limit.py      # in-memory per-IP token bucket (chat/ingest/auth)
│   └── routes/
│       ├── auth.py            # /auth/signup, /login, /logout, /me
│       ├── conversations.py   # /conversations CRUD
│       ├── ingest.py
│       ├── chat.py
│       └── sessions.py        # legacy /sessions aliases
├── static/index.html          # redesigned single-page UI (auth + sidebar + composer)
├── scripts/
│   ├── fixtures.py            # three-document evaluation corpus
│   └── eval_rag.py            # retrieval + answer-quality evaluation
├── tests/                     # pytest suite (no live API calls)
├── data/sample_docs/          # sample doc to test with
├── Dockerfile / render.yaml   # deploy config (+ managed Postgres)
├── requirements.txt / requirements-dev.txt
└── .env.example
```

## 🧪 Testing

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q          # 173 tests, no network/API calls
```

The suite stubs the LLM by default (an autouse fixture) so **no test spends a paid API call**. Every test runs
against a throwaway SQLite database and an empty vector store; the default `client` fixture is a signed-in user,
and dedicated suites cover authentication/CSRF (`test_auth.py`) and cross-account isolation
(`test_multi_tenancy.py`).

A separate, optional evaluation runs against the **live** LLM (costs Groq calls):

```bash
python scripts/eval_rag.py                                  # local, in-process
python scripts/eval_rag.py --base-url https://docubot-5fy3.onrender.com
```

It uploads a three-document corpus with deliberately non-overlapping numbers and scores answers across
`single_doc`, `cross_doc`, `unanswerable`, and `contamination` categories.

## ⚙️ Configuration

All variables are optional except `GROQ_API_KEY` (and, in production, `DOCUBOT_DATABASE_URL`). See `.env.example`.

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `GROQ_API_KEY` | — | **Required.** Groq API key. |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Chat model, validated against your account on first use. |
| `DOCUBOT_DATABASE_URL` | `sqlite:///./data/docubot.db` | Relational DB for accounts/conversations. Use a `postgres://` or `postgresql+psycopg://` URL in production; unset falls back to SQLite. |
| `DOCUBOT_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model. Changing it **requires clearing `data/chroma` and re-ingesting**. |
| `DOCUBOT_MIN_RELEVANCE` | `0.30` | Confident-relevance floor. Below it, DocuBot soft-falls-back to the closest chunks and lets the model decide (it abstains in-prompt). |
| `DOCUBOT_LEXICAL_WEIGHT` | `0.5` | Weight of lexical overlap vs cosine similarity. |
| `DOCUBOT_PER_DOC_CAP` | `2` | Max chunks one document may contribute per answer. |
| `DOCUBOT_ANALYSIS_MAX_CHUNKS` | `40` | Max chunks document analysis may read at once (bounds prompt tokens). |
| `DOCUBOT_CHROMA_PATH` | `./data/chroma` | Vector store location. |
| `DOCUBOT_COLLECTION` | `docubot_chunks_v2` | Chroma collection name. |
| `DOCUBOT_SESSION_TTL_DAYS` | `14` | Session cookie + server-side record lifetime. |
| `DOCUBOT_COOKIE_SECURE` | `true` | Mark cookies `Secure`. Set `false` only for local http / tests. |
| `DOCUBOT_MIN_PASSWORD_LENGTH` | `8` | Minimum signup password length. |
| `DOCUBOT_MAX_STORED_MESSAGES` | `200` | Messages retained per conversation before oldest are trimmed. |
| `DOCUBOT_LLM_TIMEOUT` | `45` | Per-request LLM timeout (seconds). |
| `DOCUBOT_LLM_RETRIES` | `2` | LLM client retries on transient failures. |
| `DOCUBOT_RATE_LIMIT_CHAT` | `20` | Per-IP `/chat` requests per minute (`0` disables). |
| `DOCUBOT_RATE_LIMIT_INGEST` | `10` | Per-IP `/ingest` requests per minute (`0` disables). |
| `DOCUBOT_RATE_LIMIT_AUTH` | `10` | Per-IP `/auth` requests per minute (`0` disables). |
| `DOCUBOT_CORS_ORIGINS` | *(empty)* | Comma-separated allowed origins (credentials allowed for these only). Empty = same-origin only. |
| `DOCUBOT_LOG_LEVEL` | `INFO` | Root log level. |

## ☁️ Render Deployment

- **Runtime:** Docker (see `Dockerfile`), Python 3.12.
- **Start command** (in the Dockerfile, honors Render's port):
  `uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}`
- **Health check path:** `/health`
- **Required env var:** `GROQ_API_KEY` (set as a secret in the Render dashboard; never committed).
- **Database:** point `DOCUBOT_DATABASE_URL` at a **Neon** Postgres (or any managed Postgres). It is a manually-set
  secret — `render.yaml` lists it with `sync: false` so the connection string stays out of git. Neon hands out a
  `postgresql://...?sslmode=require` URL; the app maps it to the psycopg3 driver automatically. Accounts and
  conversations then persist across deploys.

### Persistence

| Environment | Accounts / conversations / messages | Vector store (`data/chroma`) |
| ----------- | ----------------------------------- | ---------------------------- |
| Local dev | SQLite file (`./data/docubot.db`) | Persists on disk |
| Render **free** | **Neon Postgres — persists** | **Ephemeral** — lost on deploy/spin-down/restart |
| Render + **persistent disk** | Neon Postgres — persists | Persists |

On Render free the vector index is wiped on restart, so after a deploy users must re-upload documents (their
accounts and chat history remain). To keep the knowledge base too, upgrade to a paid instance and uncomment the
`disk:` block plus the `DOCUBOT_CHROMA_PATH` variable in `render.yaml` (mount `/var/data`, store
`/var/data/chroma`).

## 🧯 Troubleshooting

**HTTP 503 immediately after deploy / on first request**
- Most common cause: **cold start**. The free instance is asleep; the first request waits while the process
  imports torch and loads the embedding model. `/health` returns as soon as the app is up; the UI waits for it.
- Check the Render deploy log for a completed build and `Uvicorn running on ...`.
- If the service **crash-loops**, suspect memory (`OOM`): the free plan has 512 MB, and torch +
  sentence-transformers + ChromaDB are close to that ceiling. The Dockerfile bakes the embedding model into
  the image to avoid a runtime download; if it still OOMs, upgrade the instance type.
- Confirm `GROQ_API_KEY` is set in **Environment**; without it `/ready` reports the LLM as `unconfigured`.
- If `/ready` shows `database: {ready: false}`, the Postgres connection string is missing or unreachable.

**Chat returns 503 / "answer service is temporarily unavailable"**
- Groq is unreachable, rate-limited, or the configured `GROQ_MODEL` isn't available to your account.
  Check `GET /ready` → `checks.llm.status`, and the Render logs (`llm_call_failed` / `llm_model_unavailable`).

**Upload fails**
- 413: file exceeds 10 MB. 400: unsupported type, empty file, or a scanned/image-only PDF (no extractable
  text — OCR is not supported). Check the message the UI shows.

**401 / 403 on API calls**
- 401: not signed in (no/expired session cookie). 403: CSRF check failed — the `X-CSRF-Token` header is
  missing or doesn't match the `docubot_csrf` cookie. Sign in again and resend both.

**Documents disappeared**
- Expected on the Render free tier after any restart (accounts and history persist). Enable a persistent disk
  (above) to prevent it.

## ⚠️ Known Limitations

- On Render's free tier the **vector store is ephemeral** (accounts/conversations persist in Postgres, but
  uploaded documents must be re-added after a restart unless a disk is mounted).
- Rate limiting is **per-instance, in-memory** — it resets on restart and does not coordinate across instances.
- Corpus-wide aggregation questions ("list every limit across all documents") are answered from whichever
  documents score above the relevance floor; the floor deliberately prefers abstaining over padding context
  with irrelevant passages.
- **Document analysis is bounded by retrieved context.** Analysis reads the whole content of the *relevant*
  documents up to `DOCUBOT_ANALYSIS_MAX_CHUNKS` (default 40), so on a very large corpus a tally is "the most we
  read", not a guaranteed exhaustive scan.
- **Gender is reported only when explicitly documented.** Indirect signals (names, pronouns, photos) are never
  used; undocumented records are reported as unknown rather than guessed.
- Analysis relies on the model returning valid JSON; if it doesn't, DocuBot falls back to a normal grounded
  answer instead of failing.
- Free-text comparison questions ("compare the two documents") use the standard grounded Q&A path, not the
  deterministic record pipeline.
- Password reset / email verification are **not** implemented — signup is open.

## 🔭 Possible Extensions

- Password reset / email verification
- Streaming responses (SSE) for `/chat`
- Hosted vector DB for the knowledge base (so it survives free-tier restarts)
- Per-document sharing / team workspaces
