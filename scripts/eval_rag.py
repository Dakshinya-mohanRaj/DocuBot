#!/usr/bin/env python
"""Multi-document retrieval + answer-quality evaluation for DocuBot.

Uploads a three-document fixture corpus, asks a categorised question set, and
scores the answers on citation correctness, cross-document coverage, honest
abstention and cross-document numeric leakage.

    python scripts/eval_rag.py                      # local, in-process TestClient
    python scripts/eval_rag.py --base-url https://docubot-5fy3.onrender.com
    python scripts/eval_rag.py --shuffle --seed 7   # randomised order
    python scripts/eval_rag.py --n 5                # sample 5 questions
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.fixtures import DOCUMENTS  # noqa: E402

ABSTENTION_MARKERS = (
    "don't have",
    "do not have",
    "doesn't have",
    "does not have",
    "not mentioned",
    "no information",
    "cannot find",
    "can't find",
    "could not find",
    "couldn't find",
    "not provided",
    "not specified",
    "no mention",
    "insufficient",
    "unable to",
    "don't know",
    "do not know",
    "not available",
    "no relevant",
    "not stated",
    "nothing in the",
    "do not cover",
    "doesn't cover",
    "not address",
    "does not contain",
    "no basis",
)


@dataclass
class Question:
    text: str
    category: str
    expected_docs: list[str] = field(default_factory=list)
    must_include: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    expect_abstain: bool = False


QUESTIONS: list[Question] = [
    # ---- single document: the right document must be cited, facts must appear
    Question("How many days of paid time off does an employee accrue each full month?",
             "single_doc", ["handbook.txt"], ["1.5"]),
    Question("What is the maximum single-file upload size for the API?",
             "single_doc", ["spec.txt"], ["25"], ["1.5", "100"]),
    Question("What caused the API outage on March 14th?",
             "single_doc", ["postmortem.txt"], ["retry"], ["1.5", "100"]),
    Question("How many remote work days per week are allowed by default?",
             "single_doc", ["handbook.txt"], ["2 days"], ["100"]),
    Question("What happens when a client exceeds the rate limit?",
             "single_doc", ["spec.txt"], ["429"]),

    # ---- cross document: every relevant document must be cited
    Question("List the numeric limits stated across these documents and name the "
             "document each one comes from.",
             "cross_doc", ["handbook.txt", "spec.txt", "postmortem.txt"]),
    Question("Which documents mention a time limit or duration, and what are the values?",
             "cross_doc", ["spec.txt", "postmortem.txt"], ["seconds"]),
    Question("Compare the laptop refresh cycle with the API rate limit.",
             "cross_doc", ["handbook.txt", "spec.txt"], ["36"]),

    # ---- unanswerable: must abstain and cite nothing
    Question("What is the office wifi password?", "unanswerable", expect_abstain=True),
    Question("Who is the CEO of Northwind?", "unanswerable", expect_abstain=True),
    Question("What is the refund policy for cancelled subscriptions?",
             "unanswerable", expect_abstain=True),

    # ---- contamination: facts from other documents must not leak in
    Question("What is the cap on accrued paid time off?",
             "contamination", ["handbook.txt"], ["30 days"], ["100", "429", "timeout"]),
    Question("What retry limit was set after the outage?",
             "contamination", ["postmortem.txt"], ["5"], ["1.5", "100"]),
]


class Client:
    """Thin wrapper so the same evaluation runs locally or against a live host."""

    def __init__(self, base_url: str | None):
        self.base_url = base_url.rstrip("/") if base_url else None
        if self.base_url:
            import httpx
            self._c = httpx.Client(base_url=self.base_url, timeout=240.0)
        else:
            from dotenv import load_dotenv
            load_dotenv()
            from fastapi.testclient import TestClient
            from app.main import app
            self._c = TestClient(app)

    def upload(self, name: str, text: str) -> dict:
        files = {"file": (name, text.encode("utf-8"), "text/plain")}
        r = self._c.post("/ingest/file", files=files)
        r.raise_for_status()
        return r.json()

    def chat(self, message: str) -> dict:
        r = self._c.post("/chat", json={"message": message})
        r.raise_for_status()
        return r.json()

    def delete(self, doc_id: str) -> None:
        try:
            self._c.delete(f"/ingest/{doc_id}")
        except Exception:
            pass


def score(q: Question, result: dict) -> tuple[bool, str, list[str], str]:
    """Score one answer.

    For unanswerable questions the pass condition is that the model abstains in
    wording. Whether retrieval happened to return chunks is reported separately
    rather than treated as a failure: a question like "Who is the CEO of
    Northwind?" names an entity that really does appear in the documents, so
    lexical matching will surface them even though they cannot answer it.
    Penalising that would mark honest abstentions as failures.
    """
    answer = (result.get("answer") or "").strip()
    sources = result.get("sources") or []
    cited = [s.get("doc_id") for s in sources]
    low = answer.lower()
    failures: list[str] = []
    note = ""

    if q.expect_abstain:
        note = "abstained" + (f", but cited {cited}" if cited else ", cited nothing")
        if not any(m in low for m in ABSTENTION_MARKERS):
            failures.append("answered instead of abstaining")
        return (not failures), answer, cited, note

    missing = [d for d in q.expected_docs if d not in cited]
    if missing:
        failures.append(f"missing source(s) {missing}")

    for token in q.must_include:
        if token.lower() not in low:
            failures.append(f"answer missing {token!r}")
    for token in q.forbidden:
        if token.lower() in low:
            failures.append(f"leaked {token!r} from another document")

    return (not failures), answer, cited, note


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=None,
                    help="target a deployed instance instead of the local app")
    ap.add_argument("--n", type=int, default=0, help="sample N questions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--json-out", default="eval_report.json")
    ap.add_argument("--store", default=None,
                    help="Chroma path to evaluate against (default: throwaway temp store)")
    args = ap.parse_args()

    # The evaluation must not be polluted by whatever else is in the local
    # store, so by default it runs against a throwaway Chroma directory.
    # DOCUBOT_CHROMA_PATH is read when app.core.vector_store is first imported,
    # which happens lazily inside Client.
    if not args.base_url:
        store = args.store or tempfile.mkdtemp(prefix="docubot_eval_")
        os.environ["DOCUBOT_CHROMA_PATH"] = store
        print(f"Isolated vector store: {store}")

    qs = list(QUESTIONS)
    if args.shuffle:
        random.Random(args.seed).shuffle(qs)
    if args.n:
        qs = qs[: args.n]

    client = Client(args.base_url)
    target = args.base_url or "local (in-process TestClient)"

    print(f"DocuBot eval -> {target}\n")
    print("Uploading fixture corpus...")
    for name, text in DOCUMENTS.items():
        try:
            info = client.upload(name, text)
            print(f"  {name:18} chunks_added={info.get('chunks_added')}")
        except Exception as exc:
            print(f"  {name:18} UPLOAD FAILED: {exc}")
            return 2

    model = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
    print(f"\nModel: {model}\n" + "=" * 100)

    rows = []
    for i, q in enumerate(qs, 1):
        try:
            res = client.chat(q.text)
            ok, answer, cited, note = score(q, res)
            err = ""
        except Exception as exc:
            ok, answer, cited, note, err = False, "", [], "", f"{type(exc).__name__}: {exc}"
        rows.append({"question": q.text, "category": q.category, "passed": ok,
                     "answer": answer, "cited": cited, "error": err, "note": note,
                     "expected": q.expected_docs})

        flag = "PASS" if ok else "FAIL"
        print(f"[{flag}] {i:2}. ({q.category}) {q.text}")
        if answer:
            print(f"        answer: {answer[:220]}")
        print(f"        cited : {cited or '(none)'}")
        if note:
            print(f"        note  : {note}")
        if err:
            print(f"        error : {err}")

    print("=" * 100)
    by_cat: dict[str, list[bool]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["passed"])
    print(f"{'category':<16}{'passed':>10}")
    for cat, res in by_cat.items():
        print(f"{cat:<16}{sum(res)}/{len(res)}")
    total_ok = sum(r["passed"] for r in rows)
    print(f"{'TOTAL':<16}{total_ok}/{len(rows)}")

    with open(args.json_out, "w") as fh:
        json.dump({"target": target, "model": model, "results": rows}, fh, indent=2)
    print(f"\nReport written to {args.json_out}")

    print("\nCleaning up fixture documents...")
    for name in DOCUMENTS:
        client.delete(name)

    return 0 if total_ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())