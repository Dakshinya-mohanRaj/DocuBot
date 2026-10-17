"""Deterministic document analysis.

Quantitative and demographic questions ("how many", "what percentage", "list
all the names", "extract the skills") are answered in two stages:

1. The LLM extracts *structured records* from the document excerpts
   (``extract_records``). It never does arithmetic.
2. Python computes the counts, totals and percentages from those records
   (``format_analysis``), so numbers are deterministic and reproducible.

Privacy guardrail: gender is reported only when the document states it
explicitly. It is never inferred from names, pronouns, photographs or other
indirect clues; anything undocumented is reported separately as unknown.
"""

import json
import re

from app.core import intents, vector_store
from app.core.llm import generate_answer, extract_json

# Cues that a question wants counting / aggregation / demographic breakdown /
# bulk extraction — the cases where a deterministic tally beats free-form prose.
_ANALYSIS_PATTERNS = [
    r"\bhow many\b", r"\bcount\b", r"\bnumber of\b", r"\btotal number\b",
    r"\bsum of\b", r"\bpercentage\b", r"\bpercent\b", r"%", r"\baverage\b",
    r"\bmean\b", r"\bmedian\b", r"\btally\b", r"\bbreakdown\b",
    r"\bgender\b", r"\bmale\b", r"\bfemale\b", r"\bmen\b", r"\bwomen\b",
    r"\bsex\b", r"\bdemographic", r"\blist all\b", r"\blist the\b",
    r"\benumerate\b", r"\bextract\b", r"\ball the names\b", r"\bnames of\b",
]
_ANALYSIS_RE = re.compile("|".join(_ANALYSIS_PATTERNS))

_GENDER_RE = re.compile(r"\bgender\b|\bmale\b|\bfemale\b|\bmen\b|\bwomen\b|\bsex\b|\bdemographic")
_PERCENT_RE = re.compile(r"\bpercentage\b|\bpercent\b|%|\bproportion\b|\bratio\b")

INSUFFICIENT_ANSWER = (
    "The uploaded documents don't provide the information needed to answer that "
    "as a count or breakdown. I only report what is explicitly documented and I "
    "won't estimate or infer missing values."
)

# Explicit gender labels only — anything else is treated as undocumented.
_GENDER_MAP = {
    "male": "male", "m": "male", "man": "male", "boy": "male",
    "female": "female", "f": "female", "woman": "female", "girl": "female",
    "other": "other", "nonbinary": "other", "non binary": "other",
    "non-binary": "other", "nb": "other", "enby": "other",
    "not_stated": "unknown", "not stated": "unknown", "unspecified": "unknown",
    "not specified": "unknown", "prefer not to say": "unknown",
    "unknown": "unknown", "none": "unknown", "": "unknown", "n/a": "unknown",
    "na": "unknown", "-": "unknown",
}


def is_analysis_request(message: str) -> bool:
    """True when the message should use the deterministic record pipeline."""
    return bool(_ANALYSIS_RE.search(intents.normalize(message)))


def _normalize_gender(value) -> str:
    """Map an explicit gender label to male/female/other/unknown.

    Only labels the document states explicitly are considered; the caller is
    responsible for never extracting gender from a name or pronoun.
    """
    if not isinstance(value, str):
        return "unknown"
    return _GENDER_MAP.get(value.strip().lower(), "unknown")


def _parse_object(raw: str) -> dict | None:
    """Parse the model's JSON, tolerating code fences and surrounding prose."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    if isinstance(data, list):
        data = {"records": data}
    if not isinstance(data, dict):
        return None

    records = data.get("records")
    if not isinstance(records, list):
        records = []
    records = [r for r in records if isinstance(r, dict)]
    return {
        "records": records,
        "insufficient": bool(data.get("insufficient")),
        "notes": str(data.get("notes") or ""),
    }


def extract_records(question: str, chunks: list[dict]) -> dict | None:
    """Ask the LLM for structured records; ``None`` means unusable output.

    Kept as a thin, overridable seam so tests can inject fixtures without
    spending an LLM call.
    """
    return _parse_object(extract_json(question, chunks))


def gather_evidence(question: str) -> list[dict]:
    """Collect the chunks analysis should read.

    Analysis needs whole documents, not just the top-ranked passages, so we
    identify the relevant documents by retrieval and then pull *all* of their
    chunks up to the store's analysis budget. Corpus-wide wording reads every
    document.
    """
    norm = intents.normalize(question)
    corpus_wide = any(
        phrase in norm
        for phrase in (
            "all document", "across document", "each document", "every document",
            "both document", "all the document", "all files", "across files",
        )
    )
    if corpus_wide:
        return vector_store.get_chunks_for_documents(None)

    hits = vector_store.query(question, n_results=8, per_doc_cap=100)
    doc_ids: list[str] = []
    for hit in hits:
        if hit["doc_id"] not in doc_ids:
            doc_ids.append(hit["doc_id"])
    if not doc_ids:
        return vector_store.get_chunks_for_documents(None)
    return vector_store.get_chunks_for_documents(doc_ids)


def _dedupe(records: list[dict]) -> list[dict]:
    """Merge records that describe the same entity across chunks.

    Keyed by name + source so a candidate split across chunks is counted once.
    Unnamed records are kept separate (they cannot be safely merged).
    """
    merged: dict[tuple, dict] = {}
    unnamed: list[dict] = []
    for record in records:
        name = record.get("name")
        key = intents.normalize(name) if isinstance(name, str) and name.strip() else None
        if key is None:
            unnamed.append(record)
            continue
        source = record.get("source") or ""
        composite = (key, intents.normalize(str(source)))
        if composite not in merged:
            merged[composite] = {
                "name": name,
                "source": record.get("source"),
                "genders": [],
                "attributes": {},
            }
        entry = merged[composite]
        if record.get("gender"):
            entry["genders"].append(record["gender"])
        for k, v in (record.get("attributes") or {}).items():
            entry["attributes"].setdefault(k, v)
    return list(merged.values()) + unnamed


def _candidate_gender(record: dict) -> tuple[str, bool]:
    """Return (gender, conflict) using only explicitly stated labels."""
    labels = {_normalize_gender(g) for g in record.get("genders", []) if g}
    explicit = {lbl for lbl in labels if lbl in {"male", "female", "other"}}
    if len(explicit) > 1:
        return "conflict", True
    if explicit:
        return next(iter(explicit)), False
    return "unknown", False


def _format_gender(question: str, records: list[dict]) -> str:
    candidates = _dedupe(records)
    counts = {"male": 0, "female": 0, "other": 0, "unknown": 0, "conflict": 0}
    for candidate in candidates:
        gender, _ = _candidate_gender(candidate)
        counts[gender] += 1

    examined = len(candidates)
    explicit = counts["male"] + counts["female"] + counts["other"]
    unknown = counts["unknown"]
    wants_percent = bool(_PERCENT_RE.search(intents.normalize(question)))

    header = "| Gender (explicitly documented) | Count |" + (" Percentage |" if wants_percent else "")
    divider = "| --- | --- |" + (" --- |" if wants_percent else "")
    label = "Percentage of explicit records" if wants_percent else ""

    def row(name: str, value: int) -> str:
        if not wants_percent:
            return f"| {name} | {value} |"
        pct = (value / explicit * 100) if explicit else 0.0
        return f"| {name} | {value} | {pct:.1f}% |"

    lines = [header, divider]
    lines.append(row("Male", counts["male"]))
    lines.append(row("Female", counts["female"]))
    lines.append(row("Other", counts["other"]))
    # Unknown / conflict rows never contribute to the percentage denominator.
    lines.append(f"| Unknown / not stated | {unknown} |" + (" — |" if wants_percent else ""))
    if counts["conflict"]:
        lines.append(
            f"| Conflicting statements | {counts['conflict']} |"
            + (" — |" if wants_percent else "")
        )
    lines.append(f"| **Total records examined** | **{examined}** |" + ("  |" if wants_percent else ""))

    notes = [
        f"I examined {examined} candidate record(s). "
        f"{explicit} explicitly documented a gender; {unknown} did not."
    ]
    if wants_percent and explicit:
        notes.append(
            f"Percentages use the denominator of {explicit} record(s) with an "
            "explicitly documented gender — records listed as unknown / not "
            "stated are excluded from that denominator."
        )
    elif wants_percent:
        notes.append("No records documented a gender, so no percentages are reported.")
    notes.append(
        "Gender is reported only when the document states it explicitly; it is "
        "never inferred from names, pronouns or other indirect clues."
    )

    return "\n".join(lines) + "\n\n" + " ".join(notes)


def _format_records(question: str, records: list[dict]) -> str:
    records = _dedupe(records)
    if not records:
        return INSUFFICIENT_ANSWER

    attr_keys: list[str] = []
    for record in records:
        for key in (record.get("attributes") or {}):
            if key not in attr_keys:
                attr_keys.append(key)
    attr_keys = attr_keys[:4]

    if not attr_keys:
        lines = [f"- {r.get('name') or 'Record'}"
                 + (f" — [{r['source']}]" if r.get("source") else "")
                 for r in records[:25]]
        return "\n".join(lines)

    header = "| Name | " + " | ".join(attr_keys) + " | Source |"
    divider = "| --- |" + " --- |" * len(attr_keys) + " --- |"
    rows = [header, divider]
    for record in records[:25]:
        attrs = record.get("attributes") or {}
        cells = [str(attrs.get(key, "—")) for key in attr_keys]
        rows.append(
            "| " + (record.get("name") or "—") + " | "
            + " | ".join(cells) + " | "
            + (record.get("source") or "—") + " |"
        )
    return "\n".join(rows)


def format_analysis(question: str, parsed: dict) -> str:
    """Turn extracted records into a deterministic, cited answer."""
    records = parsed.get("records") or []
    if parsed.get("insufficient") or not records:
        return INSUFFICIENT_ANSWER
    if _GENDER_RE.search(intents.normalize(question)):
        return _format_gender(question, records)
    return _format_records(question, records)


def answer_from_evidence(question: str, chunks: list[dict], history: list[dict]) -> str:
    """Analyse already-gathered chunks (extract records, then compute)."""
    if not chunks:
        return INSUFFICIENT_ANSWER

    parsed = extract_records(question, chunks)
    if parsed is None:
        # Unusable structured output — fall back to a normal grounded answer
        # rather than failing the request.
        return generate_answer(question, chunks, history)
    return format_analysis(question, parsed)


def answer_analysis(question: str, history: list[dict]) -> str:
    """Full analysis path: gather evidence, then analyse it."""
    return answer_from_evidence(question, gather_evidence(question), history)
