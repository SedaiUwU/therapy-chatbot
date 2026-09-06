import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from safety import (
    AMBIGUOUS_CONCERN,
    EXPLICIT_HIGH_RISK,
    NORMAL,
    THIRD_PARTY_CONCERN,
    classify_safety,
)


def category(message, context=None):
    return classify_safety(message, context or []).category


def test_normal_distress_is_not_safety_risk():
    for message in (
        "I'm stressed about my exam.",
        "I've been feeling sad lately.",
        "I'm overwhelmed with school.",
        "I'm scared I'll fail.",
    ):
        assert category(message) == NORMAL


def test_explicit_current_first_person_signals():
    for message in (
        "I want to kill myself.",
        "I want to die.",
        "I'm going to hurt myself.",
        "I'm thinking about suicide.",
        "I might harm myself.",
        "I plan to end my life.",
        "I'm suicidal.",
    ):
        assert category(message) == EXPLICIT_HIGH_RISK


def test_negated_and_historical_language_is_not_current_explicit_risk():
    for message in (
        "I don't want to kill myself.",
        "I'm not suicidal.",
        "I don't have suicidal thoughts.",
        "I used to have suicidal thoughts, but I don't now.",
        "I thought about suicide last year.",
    ):
        assert category(message) != EXPLICIT_HIGH_RISK


def test_third_party_concern_has_dedicated_route():
    assert category("My friend says they want to die.") == THIRD_PARTY_CONCERN
    assert category("My sister said she wants to kill herself.") == THIRD_PARTY_CONCERN


def test_reference_language_is_normal():
    for message in (
        "I'm writing an essay about suicide.",
        "What does suicide mean?",
        "The movie discussed suicide.",
    ):
        assert category(message) == NORMAL


def test_current_explicit_language_overrides_framing():
    for message in (
        "I'm writing about suicide, but I've also been thinking about killing myself.",
        "I'm writing about suicide, but I've been thinking about killing myself.",
        "I'm studying suicide prevention, but I want to die.",
        "This is for an essay, but I'm suicidal.",
        "I used to write about suicide, but now I'm thinking about hurting myself.",
        "I used to have these thoughts, and now I want to die.",
    ):
        assert category(message) == EXPLICIT_HIGH_RISK


def test_ambiguous_concern_has_dedicated_route():
    for message in (
        "I can't do this anymore.",
        "I want everything to stop.",
        "I just want to disappear.",
        "There's no point.",
        "I can't keep going.",
    ):
        assert category(message) == AMBIGUOUS_CONCERN


def test_boundaries_and_hyphen_variants():
    assert category("I'M SUICIDAL!!!") == EXPLICIT_HIGH_RISK
    assert category("I might harm-myself.") == EXPLICIT_HIGH_RISK
    assert category("The word suicide is in this sentence.") == NORMAL


def test_assistant_context_is_not_safety_evidence():
    context = ["The assistant said: I want to die."]
    assert category("Nothing new today.", context) == NORMAL


def test_current_explicit_message_overrides_reassuring_context():
    context = ["I feel calm and safe today."]
    assert category("I want to die.", context) == EXPLICIT_HIGH_RISK


def test_safety_routes_bypass_provider(monkeypatch):
    import app

    def fail_if_called(_prompt):
        raise AssertionError("Groq must not be called for a safety route")

    monkeypatch.setattr(app, "get_ai_response", fail_if_called)
    for message in (
        "I want to die.",
        "I can't keep going.",
        "My friend says they want to die.",
        "I'm writing about suicide, but I've also been thinking about killing myself.",
    ):
        response = app.get_safety_response(message, [])
        assert response
        assert "988" not in response
        assert "911" not in response