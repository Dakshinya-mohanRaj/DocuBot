def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Split text into overlapping chunks by character count.
    Simple and dependency-free — good enough for a first RAG pass.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start = end - overlap  # step forward with overlap

    return chunks
