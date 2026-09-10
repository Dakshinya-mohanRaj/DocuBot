import os
from groq import Groq

SYSTEM_PROMPT = """You are DocuBot, a helpful assistant that answers questions strictly \
based on the provided document context. Rules:
- Only use information found in the context below.
- If the answer isn't in the context, say you don't have enough information — do not guess.
- Keep answers concise and cite which source chunk(s) you used by their doc_id.
"""



def get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in environment or .env file.")
    return Groq(api_key=api_key)


CANDIDATE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]


def get_best_model(client: Groq) -> str:
    try:
        models_response = client.models.list()
        active_ids = {m.id for m in models_response.data}
        for cand in CANDIDATE_MODELS:
            if cand in active_ids:
                return cand
        for m_id in active_ids:
            if "whisper" not in m_id.lower() and "safeguard" not in m_id.lower():
                return m_id
    except Exception:
        pass
    return "llama-3.1-8b-instant"


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

    client = get_client()
    selected_model = get_best_model(client)

    # Try selected model first, with fallbacks if model_not_found occurs
    models_to_try = [selected_model] + [m for m in CANDIDATE_MODELS if m != selected_model]
    last_exception = None

    for model_name in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model_name,
                max_tokens=500,
                messages=messages,
            )
            content = response.choices[0].message.content or ""
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
            return content

        except Exception as e:
            last_exception = e
            if "model_not_found" in str(e) or "404" in str(e):
                continue
            break

    raise RuntimeError(f"Groq API Error: {str(last_exception)}") from last_exception