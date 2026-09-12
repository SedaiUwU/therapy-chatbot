import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conversation_engine import ConversationEngine
from safety import (
    AMBIGUOUS_CONCERN,
    EXPLICIT_HIGH_RISK,
    THIRD_PARTY_CONCERN,
    deterministic_safety_response,
    classify_safety,
)


class RecordingProvider:
    def __init__(self, response="fake response"):
        self.response = response
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.response


def test_normal_message_uses_provider_once_and_returns_response():
    provider = RecordingProvider("provider response")
    engine = ConversationEngine(provider)

    result = engine.process_message(
        "What should I do next?",
        [
            {"role": "user", "content": "My exam is tomorrow."},
            {"role": "assistant", "content": "That sounds stressful."},
        ],
    )

    assert result.assistant_response == "provider response"
    assert result.safety_category is None
    assert len(provider.prompts) == 1
    assert provider.prompts[0].count("What should I do next?") == 1
    assert "USER: My exam is tomorrow." in provider.prompts[0]
    assert "ASSISTANT: That sounds stressful." in provider.prompts[0]


def test_engine_preserves_memory_window_and_does_not_mutate_messages():
    provider = RecordingProvider()
    engine = ConversationEngine(provider)
    messages = [
        {"role": "user", "content": "Turn 1 user"},
        {"role": "assistant", "content": "Turn 1 assistant"},
        {"role": "user", "content": "Turn 2 user"},
        {"role": "assistant", "content": "Turn 2 assistant"},
        {"role": "user", "content": "Turn 3 user"},
        {"role": "assistant", "content": "Turn 3 assistant"},
        {"role": "user", "content": "Turn 4 user"},
        {"role": "assistant", "content": "Turn 4 assistant"},
    ]
    original_messages = [message.copy() for message in messages]

    result = engine.process_message("A new current message", messages)

    prompt = provider.prompts[0]
    assert "Turn 1 user" not in prompt
    assert "Turn 2 user" in prompt
    assert "Turn 2 assistant" in prompt
    assert "Turn 3 user" in prompt
    assert "Turn 3 assistant" in prompt
    assert "Turn 4 user" in prompt
    assert "Turn 4 assistant" in prompt
    assert prompt.count("A new current message") == 1
    assert messages == original_messages
    assert result.mood == "neutral"


def test_explicit_high_risk_bypasses_provider_and_state_mutation():
    provider = RecordingProvider()
    engine = ConversationEngine(provider)

    result = engine.process_message(
        "I want to die.",
        [],
        emotion_streak=3,
        stuck_state=2,
    )

    analysis = classify_safety("I want to die.")
    assert result.safety_category == EXPLICIT_HIGH_RISK
    assert result.assistant_response == deterministic_safety_response(analysis)
    assert result.mood is None
    assert result.emotion_streak == 3
    assert result.stuck_state == 2
    assert provider.prompts == []


def test_ambiguous_and_third_party_routes_bypass_provider():
    for message, category in (
        ("I can't keep going.", AMBIGUOUS_CONCERN),
        ("My friend says they want to die.", THIRD_PARTY_CONCERN),
    ):
        provider = RecordingProvider()
        result = ConversationEngine(provider).process_message(message, [])

        assert result.safety_category == category
        assert result.assistant_response == deterministic_safety_response(
            classify_safety(message)
        )
        assert provider.prompts == []


def test_reference_message_remains_normal_provider_path():
    provider = RecordingProvider("reference response")
    result = ConversationEngine(provider).process_message(
        "I'm writing an essay about suicide.",
        [],
    )

    assert result.safety_category is None
    assert result.assistant_response == "reference response"
    assert result.mood == "neutral"
    assert len(provider.prompts) == 1


def test_state_and_mood_result_preserve_existing_semantics():
    provider = RecordingProvider()
    engine = ConversationEngine(provider)

    distressed = engine.process_message("I feel sad.", [], emotion_streak=1, stuck_state=4)
    uncertain = engine.process_message(
        "I don't know what to do.",
        [{"role": "user", "content": "I feel sad."}],
        emotion_streak=1,
        stuck_state=0,
    )

    assert distressed.mood == "sad"
    assert distressed.emotion_streak == 2
    assert distressed.stuck_state == 0
    assert uncertain.mood == "neutral"
    assert uncertain.emotion_streak == 0
    assert uncertain.stuck_state == 1


def test_structured_result_exposes_only_app_boundary_fields():
    provider = RecordingProvider()
    result = ConversationEngine(provider).process_message("Hello", [])

    assert {
        "assistant_response",
        "mood",
        "emotion_streak",
        "stuck_state",
        "safety_category",
    } == set(result.__dataclass_fields__)
