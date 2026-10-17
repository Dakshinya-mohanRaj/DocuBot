"""Deterministic document analysis: counts, percentages, dedup, gender safety."""

from app.core import analysis


def _rec(name, gender, source="resume.txt", **attrs):
    return {"name": name, "gender": gender, "source": source, "attributes": attrs}


# --- detection -------------------------------------------------------------

def test_analysis_request_detection():
    for message in [
        "How many candidates are there?",
        "Count the male and female applicants.",
        "What percentage list Python?",
        "List all the names.",
        "Extract the skills and experience.",
        "Give me a demographic breakdown.",
    ]:
        assert analysis.is_analysis_request(message), message


def test_non_analysis_questions_are_not_flagged():
    for message in [
        "What is this document about?",
        "Summarize this PDF.",
        "What are the candidate's qualifications?",
        "Which projects are mentioned?",
    ]:
        assert not analysis.is_analysis_request(message), message


# --- parsing ---------------------------------------------------------------

def test_parse_plain_json():
    parsed = analysis._parse_object('{"records": [{"name": "A"}], "insufficient": false}')
    assert parsed["records"] == [{"name": "A"}]
    assert parsed["insufficient"] is False


def test_parse_json_in_code_fences():
    raw = '```json\n{"records": [{"name": "A"}], "insufficient": false}\n```'
    assert analysis._parse_object(raw)["records"] == [{"name": "A"}]


def test_parse_json_with_surrounding_prose():
    raw = 'Here you go: {"records": [{"name": "A"}], "insufficient": false} Thanks!'
    assert analysis._parse_object(raw)["records"] == [{"name": "A"}]


def test_parse_bare_list():
    assert analysis._parse_object('[{"name": "A"}]')["records"] == [{"name": "A"}]


def test_parse_garbage_returns_none():
    assert analysis._parse_object("I cannot help with that.") is None
    assert analysis._parse_object("") is None


# --- gender guardrails -----------------------------------------------------

def test_gender_counts_use_only_explicit_labels():
    records = [
        _rec("Alice", "female"),
        _rec("Bob", "male"),
        _rec("John", "unknown"),      # name only, no documented gender
        _rec("Carol", "not_stated"),
    ]
    answer = analysis.format_analysis("How many are male and female?", {"records": records})
    assert "| Male | 1 |" in answer
    assert "| Female | 1 |" in answer
    assert "| Unknown / not stated | 2 |" in answer
    assert "examined 4 candidate" in answer.replace("**", "")
    # John must NOT be counted as male just because of his name.
    assert answer.count("| Male | 1 |") == 1


def test_gender_never_inferred_from_name():
    # A record the model could only populate from a name carries gender
    # "unknown"; it must be reported as undocumented, not guessed.
    records = [_rec("Michael", "unknown"), _rec("Sarah", "unknown")]
    answer = analysis.format_analysis("Break down the gender of applicants", {"records": records})
    assert "| Unknown / not stated | 2 |" in answer
    assert "| Male | 0 |" in answer
    assert "| Female | 0 |" in answer


def test_percentage_uses_explicit_denominator():
    records = [
        _rec("A", "male"), _rec("B", "male"),
        _rec("C", "female"),
        _rec("D", "unknown"),
    ]
    answer = analysis.format_analysis("What percentage are male?", {"records": records})
    assert "66.7%" in answer
    # The denominator is the 3 explicit records, not the 4 examined.
    assert "denominator of 3" in answer


def test_conflicting_gender_statements_are_separated():
    records = [
        _rec("Alex", "male"),
        _rec("Alex", "female"),  # same person, contradictory across chunks
    ]
    answer = analysis.format_analysis("How many male and female?", {"records": records})
    assert "Conflicting statements" in answer
    assert "| Male | 0 |" in answer
    assert "| Female | 0 |" in answer


# --- dedup / counting ------------------------------------------------------

def test_multiple_chunks_do_not_double_count():
    records = [
        _rec("Alice", "female"),
        _rec("Alice", "female", source="resume.txt"),
        _rec("Alice", "female", source="resume.txt"),
        _rec("Bob", "male"),
    ]
    answer = analysis.format_analysis("How many female candidates?", {"records": records})
    assert "| Female | 1 |" in answer
    assert "examined 2 candidate" in answer.replace("**", "")


def test_empty_or_insufficient_reports_honestly():
    assert analysis.format_analysis("How many?", {"records": [], "insufficient": True}) == analysis.INSUFFICIENT_ANSWER
    assert analysis.format_analysis("How many?", {"records": []}) == analysis.INSUFFICIENT_ANSWER


def test_record_listing_includes_sources():
    records = [
        _rec("Alice", "female", skills="Python, SQL"),
        _rec("Bob", "male", skills="Java"),
    ]
    answer = analysis.format_analysis("List all candidates and their skills", {"records": records})
    assert "| Alice |" in answer
    assert "resume.txt" in answer
    assert "Python, SQL" in answer


# --- fallback --------------------------------------------------------------

def test_unparseable_extraction_falls_back_to_grounded_answer(monkeypatch):
    monkeypatch.setattr("app.core.analysis.extract_records", lambda q, c: None)
    monkeypatch.setattr(
        "app.core.analysis.generate_answer",
        lambda q, c, h: "Grounded fallback [resume.txt]",
    )
    chunks = [{"text": "Alice. Gender: female.", "doc_id": "d", "filename": "resume.txt"}]
    assert analysis.answer_from_evidence("how many", chunks, []) == "Grounded fallback [resume.txt]"


def test_no_evidence_reports_insufficient():
    assert analysis.answer_from_evidence("how many", [], []) == analysis.INSUFFICIENT_ANSWER
