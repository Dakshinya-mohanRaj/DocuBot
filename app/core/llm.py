import os
import re
from functools import lru_cache

from groq import Groq

DEFAULT_MODEL = "qwen/qwen3.8-27b"

MAX_TOKENS = 1024
TEMPERATURE = 0.1

# Substrings that mark models unusable for chat: guard/classification models
# never answer, whisper is speech-to-text only, and the canopylabs orpheus
# models are gated behind org-level terms acceptance and fail with
# `model_terms_required` on a default account.
EXCLUDED_MODEL_MARKERS = (
    "whisper",
    "guard",
    "safeguard",
    "embed",
    "classification",
    "orpheus",
)

SYSTEM_PROMPT = """You are DocuBot, a helpful assistant that answers questions strictly \
based on the provided document context.

Rules:
- Answer ONLY from the numbered context excerpts below. Never use outside \
knowledge or assume facts that are not stated.
- The excerpts come from different documents. Use an excerpt only if it \
actually addresses the question, and ignore the rest.
- Comparisons and cross-document questions are expected. When the question \
asks you to compare, contrast or summarise across documents, use the \
relevant excerpt from each one and state each figure next to its own document \
name. Report what the documents say; do not manufacture a relationship, \
common metric or conclusion that the documents do not state.
- Never merge values from different documents into a single number, and never \
alter a figure. Copy numbers, dates and names exactly as written.
- Cite the document name(s) you used at the end, in the form [filename]. \
Cite every document the answer draws on.
- If the context genuinely does not contain the answer, say plainly that the \
documents do not cover it. Do not guess or fill gaps from general knowledge. \
Abstain only when the information is absent -- not merely because a question \
compares unrelated things.
- Your own earlier answers in this conversation may have been wrong. Prefer \
the current context over anything you said previously.
- Respond directly with the final answer. Do NOT output thinking, reasoning, or \
<think> tags.

Examples of the expected format:
The cap on accrued time off is 30 days. [handbook.txt]
The laptop refresh cycle is 36 months, while the API rate limit is 100 requests \
per minute. [handbook.txt] [spec.txt]
"""


def get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in environment or .env file.")
    return Groq(api_key=api_key)


@lru_cache(maxsize=1)
def resolve_model() -> str:
    """Return the single pinned chat model, validated once per process.

    Previously this iterated the live model list and picked whichever model
    came first out of a Python ``set``. Set ordering is randomized per
    process, so each container silently picked a different model and chat
    worked or failed depending on deployment luck. The model is now pinned
    explicitly and validated once, so every instance behaves identically.
    """
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

    try:
        active = {m.id for m in get_client().models.list().data}
    except Exception as exc:
        raise RuntimeError(
            f"Could not verify Groq model availability: {exc}"
        ) from exc

    if model not in active:
        usable = sorted(
            m
            for m in active
            if not any(bad in m.lower() for bad in EXCLUDED_MODEL_MARKERS)
        )
        raise RuntimeError(
            f"Configured model {model!r} is not available to this Groq account. "
            f"Set GROQ_MODEL to one of: {', '.join(usable) or '(none)'}"
        )

    return model


def _strip_thinking(content: str) -> str:
    return re.sub(r"<think>.*?(?:</think>|$)", "", content, flags=re.DOTALL).strip()


def generate_answer(question: str, context_chunks: list[dict], history: list[dict]) -> str:
    context_block = "\n\n".join(
        f"[{i}] Source: {c.get('filename') or c['doc_id']}\n{c['text']}"
        for i, c in enumerate(context_chunks, 1)
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context_block}\n\nQuestion: {question}",
        }
    )

    model_name = resolve_model()
    response = get_client().chat.completions.create(
        model=model_name,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=messages,
    )

    content = response.choices[0].message.content or ""
    answer = _strip_thinking(content)

    if not answer:
        raise RuntimeError(
            f"Model {model_name!r} returned an empty answer "
            f"(raw content: {content[:200]!r})"
        )

    return answer