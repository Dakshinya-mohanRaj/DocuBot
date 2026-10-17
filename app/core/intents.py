"""Deterministic intent routing for conversational chat.

Greetings, small talk, identity and capability questions are answered directly
from a fixed set of replies so they never touch the embedding model or the LLM
(no retrieval cost, no API spend, and they still work with an empty knowledge
base). Everything else is treated as a document request and flows through the
normal RAG pipeline.

The matching is intentionally conservative: only a whole-message phrase, or a
message that is *only* a greeting (plus filler such as "there"/"docubot"),
counts as conversational. "hi, what is the PTO policy?" starts with a greeting
but carries a real question, so it is routed to document Q&A. This avoids the
classic bug where a substring like "hi" inside a substantive question is
misread as a greeting.
"""

from dataclasses import dataclass

# Intent kinds.
GREETING = "greeting"
SMALL_TALK = "small_talk"
IDENTITY = "identity"
HELP = "help"
DOCUMENT = "document"

# Kinds that are answered conversationally (no retrieval, no LLM).
CONVERSATIONAL_KINDS = frozenset({GREETING, SMALL_TALK, IDENTITY, HELP})

GREETING_REPLY = (
    "Hi! 👋 I'm DocuBot. You can upload a document and ask questions, summarize "
    "its contents, or analyze information available in it. How can I help?"
)
MORNING_REPLY = (
    "Good morning! 👋 I'm DocuBot. Upload a document and ask questions, summarize "
    "its contents, or analyze information available in it. How can I help?"
)
AFTERNOON_REPLY = (
    "Good afternoon! 👋 I'm DocuBot. Upload a document and ask questions, summarize "
    "its contents, or analyze information available in it. How can I help?"
)
EVENING_REPLY = (
    "Good evening! 👋 I'm DocuBot. Upload a document and ask questions, summarize "
    "its contents, or analyze information available in it. How can I help?"
)
HOW_ARE_YOU_REPLY = "I'm doing well and ready to help you with your documents!"
THANKS_REPLY = "You're welcome! Let me know if you have more questions about your documents."
BYE_REPLY = "Goodbye! 👋 Come back anytime you need help with your documents."
IDENTITY_REPLY = (
    "I'm DocuBot, a document assistant. I answer using only the documents you "
    "upload and I cite the sources I use."
)

CAPABILITIES_REPLY = (
    "I'm DocuBot, and I work only from the documents you upload. Here's what I can do:\n"
    "- **Document Q&A** — answer questions about your files, with source citations.\n"
    "- **Summarization** — summarize a whole document or a section.\n"
    "- **Extraction** — pull out names, dates, skills, qualifications and other details.\n"
    "- **Analysis** — count, tally and compare information across documents "
    "(for example, how many candidates list a given skill).\n"
    "- **Comparison** — compare information across multiple uploaded documents.\n"
    "Ask a question about your documents, or upload one to get started."
)

# Whole-message phrases (already normalised: lowercase, punctuation -> spaces).
_GREETING_WORDS = {
    "hi", "hello", "hey", "heya", "hiya", "yo", "howdy", "greetings",
}
_TIME_GREETINGS = {
    "good morning": GREETING_REPLY,
    "good afternoon": AFTERNOON_REPLY,
    "good evening": EVENING_REPLY,
    "good day": GREETING_REPLY,
}
# Words that may accompany a greeting without adding a real question.
_FILLER = {
    "there", "docubot", "bot", "please", "again", "everyone", "all", "team",
    "folks", "friend", "mate", "dude", "mr", "mrs", "sir", "madam",
}

_SMALL_TALK = {
    "how are you": HOW_ARE_YOU_REPLY,
    "how are you doing": HOW_ARE_YOU_REPLY,
    "how are you today": HOW_ARE_YOU_REPLY,
    "how is it going": HOW_ARE_YOU_REPLY,
    "how is everything": HOW_ARE_YOU_REPLY,
    "how do you do": HOW_ARE_YOU_REPLY,
    "thanks": THANKS_REPLY,
    "thank you": THANKS_REPLY,
    "thanks a lot": THANKS_REPLY,
    "thank you so much": THANKS_REPLY,
    "many thanks": THANKS_REPLY,
    "cheers": THANKS_REPLY,
    "bye": BYE_REPLY,
    "goodbye": BYE_REPLY,
    "good bye": BYE_REPLY,
    "see you": BYE_REPLY,
    "see ya": BYE_REPLY,
    "see you later": BYE_REPLY,
    "farewell": BYE_REPLY,
    "take care": BYE_REPLY,
}

_IDENTITY = {
    "who are you": IDENTITY_REPLY,
    "what are you": IDENTITY_REPLY,
    "who is this": IDENTITY_REPLY,
    "what is your name": IDENTITY_REPLY,
    "whats your name": IDENTITY_REPLY,
    "your name": IDENTITY_REPLY,
    "who am i talking to": IDENTITY_REPLY,
}

_HELP = {
    "help": CAPABILITIES_REPLY,
    "what can you do": CAPABILITIES_REPLY,
    "what can you do for me": CAPABILITIES_REPLY,
    "what do you do": CAPABILITIES_REPLY,
    "what can you help with": CAPABILITIES_REPLY,
    "what are your capabilities": CAPABILITIES_REPLY,
    "capabilities": CAPABILITIES_REPLY,
    "how can you help": CAPABILITIES_REPLY,
    "how do you work": CAPABILITIES_REPLY,
    "what can i ask": CAPABILITIES_REPLY,
    "what can i ask you": CAPABILITIES_REPLY,
}


@dataclass(frozen=True)
class Intent:
    kind: str
    response: str | None = None

    @property
    def is_conversational(self) -> bool:
        return self.kind in CONVERSATIONAL_KINDS


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    lowered = (text or "").lower()
    cleaned = "".join(ch if ch.isalnum() else " " for ch in lowered)
    return " ".join(cleaned.split())


def _strip_leading_greeting(norm: str) -> tuple[bool, str]:
    """Remove a leading greeting (and look-ahead filler) from a normalised string.

    Returns ``(stripped, remainder)``. Time-of-day greetings are matched as
    multi-word prefixes before single-word greetings.
    """
    for phrase in sorted(_TIME_GREETINGS, key=len, reverse=True):
        if norm == phrase:
            return True, ""
        if norm.startswith(phrase + " "):
            return True, norm[len(phrase) + 1 :].strip()

    tokens = norm.split()
    if not tokens or tokens[0] not in _GREETING_WORDS:
        return False, norm

    rest = tokens[1:]
    # Collapse a run of greetings ("hi hello there").
    while rest and rest[0] in _GREETING_WORDS:
        rest = rest[1:]
    return True, " ".join(rest)


def _match_phrase(norm: str) -> Intent | None:
    if norm in _HELP:
        return Intent(HELP, _HELP[norm])
    if norm in _IDENTITY:
        return Intent(IDENTITY, _IDENTITY[norm])
    if norm in _SMALL_TALK:
        return Intent(SMALL_TALK, _SMALL_TALK[norm])
    if norm in _TIME_GREETINGS:
        return Intent(GREETING, _TIME_GREETINGS[norm])
    if norm in _GREETING_WORDS:
        return Intent(GREETING, GREETING_REPLY)
    return None


def detect(message: str) -> Intent:
    """Classify a raw user message into a conversational intent or DOCUMENT."""
    norm = normalize(message)
    if not norm:
        return Intent(DOCUMENT)

    direct = _match_phrase(norm)
    if direct is not None:
        return direct

    stripped, remainder = _strip_leading_greeting(norm)
    if stripped:
        if not remainder or all(token in _FILLER for token in remainder.split()):
            return Intent(GREETING, GREETING_REPLY)
        follow_up = _match_phrase(remainder)
        if follow_up is not None:
            return follow_up
        # A greeting followed by a real question is a document request.
        return Intent(DOCUMENT)

    return Intent(DOCUMENT)
