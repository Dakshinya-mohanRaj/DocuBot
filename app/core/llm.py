import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Free, fast Groq-hosted model. See https://console.groq.com/docs/models for current options.
MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are DocuBot, a helpful assistant that answers questions strictly \
based on the provided document context. Rules:
- Only use information found in the context below.
- If the answer isn't in the context, say you don't have enough information — do not guess.
- Keep answers concise and cite which source chunk(s) you used by their doc_id.
"""


def generate_answer(question: str, context_chunks: list[dict], history: list[dict]) -> str:
    context_block = "\n\n".join(
        f"[Source: {c['doc_id']}]\n{c['text']}" for c in context_chunks
    )

    # Groq uses the OpenAI-style chat format: a "system" message plus a rolling
    # list of {"role": "user"/"assistant", "content": "..."} turns.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context_block}\n\nQuestion: {question}",
        }
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=800,
        messages=messages,
    )

    return response.choices[0].message.content