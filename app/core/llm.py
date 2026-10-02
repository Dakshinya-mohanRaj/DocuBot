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
based on the provided document context. Rules:
- Only use information found in the context below.
- If the answer isn't in the context, say you don't have enough information — do not guess.
- Keep answers concise and cite which source chunk(s) you used by their doc_id.
- Respond directly with the final answer. Do NOT output thinking, reasoning, or <think> tags.
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
        f"[Source: {c['doc_id']}]\n{c['text']}" for c in context_chunks
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