"""Deterministic intent routing for conversational messages."""

import pytest

from app.core import intents


@pytest.mark.parametrize(
    "message",
    [
        "Hi", "hi", "HI", "Hello", "hey", "heya", "Hi!", "hi there",
        "hello docubot", "Hey there!", "  hi  ", "Hello!",
    ],
)
def test_greetings(message):
    assert intents.detect(message).kind == intents.GREETING


@pytest.mark.parametrize(
    "message",
    ["Good morning", "good morning!", "Good Afternoon", "Good evening.", "good day"],
)
def test_time_greetings(message):
    assert intents.detect(message).kind == intents.GREETING


@pytest.mark.parametrize(
    "message",
    ["how are you?", "How are you doing?", "hey, how are you?", "how is it going"],
)
def test_how_are_you(message):
    assert intents.detect(message).kind == intents.SMALL_TALK


@pytest.mark.parametrize("message", ["thanks", "Thank you!", "Thanks a lot", "cheers"])
def test_thanks(message):
    assert intents.detect(message).kind == intents.SMALL_TALK


@pytest.mark.parametrize("message", ["bye", "Goodbye!", "see you later", "take care"])
def test_bye(message):
    assert intents.detect(message).kind == intents.SMALL_TALK


@pytest.mark.parametrize("message", ["who are you?", "What is your name?", "who is this"])
def test_identity(message):
    assert intents.detect(message).kind == intents.IDENTITY


@pytest.mark.parametrize(
    "message",
    ["help", "What can you do?", "what do you do", "What are your capabilities?",
     "how can you help", "what can i ask you"],
)
def test_help(message):
    assert intents.detect(message).kind == intents.HELP


@pytest.mark.parametrize(
    "message",
    [
        "What is the API rate limit?",
        "Summarize this PDF.",
        "What are the candidate's qualifications?",
        "Which projects are mentioned?",
        "Extract the dates and important details.",
        "What is this document about?",
        "How do I request a laptop?",  # "how do" must not match "how do you do"
    ],
)
def test_substantive_questions_are_document_requests(message):
    assert intents.detect(message).kind == intents.DOCUMENT


def test_greeting_followed_by_question_is_a_document_request():
    assert intents.detect("hi, what is the PTO policy?").kind == intents.DOCUMENT
    assert intents.detect("Hello, can you summarize the report?").kind == intents.DOCUMENT
    assert intents.detect("good morning, what are the working hours?").kind == intents.DOCUMENT


def test_embedded_greeting_word_is_not_a_greeting():
    # "hi" inside a word must not trigger greeting detection.
    assert intents.detect("What is the shipping policy?").kind == intents.DOCUMENT
    assert intents.detect("this document is large").kind == intents.DOCUMENT


def test_conversational_intents_carry_a_response():
    for message in ["hi", "how are you?", "thanks", "who are you?", "help"]:
        intent = intents.detect(message)
        assert intent.is_conversational
        assert intent.response

    assert intents.detect("what is the rate limit?").response is None
