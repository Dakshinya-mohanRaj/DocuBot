"""Sentence-aware chunking.

The original implementation split on a fixed character window, which cut
mid-word and mid-sentence and shredded tabular content. This version splits on
paragraph and sentence boundaries, keeps table rows whole, and only falls back
to a hard character split for a single sentence that is longer than the target
chunk size.
"""

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def _looks_like_table(block: str) -> bool:
    lines = [ln for ln in block.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    return all(ln.count("|") >= 2 for ln in lines)


def _split_oversized(text: str, chunk_size: int) -> list[str]:
    """Break a single over-long sentence on word boundaries."""
    pieces: list[str] = []
    current = ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > chunk_size:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        pieces.append(current)
    return pieces


def _units(text: str, chunk_size: int) -> list[str]:
    """Break text into indivisible units that will never be split further."""
    units: list[str] = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if _looks_like_table(paragraph):
            units.extend(ln.strip() for ln in paragraph.splitlines() if ln.strip())
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > chunk_size:
                units.extend(_split_oversized(sentence, chunk_size))
            else:
                units.append(sentence)
    return units


def _joined_len(parts: list[str]) -> int:
    return sum(len(p) for p in parts) + max(0, len(parts) - 1)


def _join(parts: list[str]) -> str:
    """Reassemble units, keeping table rows on separate lines."""
    if parts and all("|" in part for part in parts):
        return "\n".join(parts)
    return " ".join(parts)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks on sentence boundaries.

    Chunks are packed up to ``chunk_size`` characters and each chunk repeats up
    to ``overlap`` characters of trailing context from the previous one.
    """
    text = text.strip()
    if not text:
        return []

    units = _units(text, chunk_size)
    chunks: list[str] = []
    current: list[str] = []

    for unit in units:
        if current and _joined_len(current) + 1 + len(unit) > chunk_size:
            chunks.append(_join(current))
            tail: list[str] = []
            for previous in reversed(current):
                trial = [previous] + tail
                if _joined_len(trial) > overlap:
                    break
                tail = trial
            current = tail
        current.append(unit)

    if current:
        chunks.append(_join(current))

    return [chunk for chunk in (c.strip() for c in chunks) if chunk]