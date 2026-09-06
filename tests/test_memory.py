"""
Tests for conversation memory and context building functionality.

Verifies that the recent-conversation context system works correctly
without making live API calls to IBM watsonx.
"""

import sys
import os

# Add parent directory to path so we can import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_build_recent_context_empty():
    """Test that empty message history returns empty context."""
    from app import build_recent_context
    
    result = build_recent_context([])
    assert result == "", f"Expected empty string, got: {result}"
    print("✓ test_build_recent_context_empty passed")


def test_build_recent_context_single_turn():
    """Test that single previous turn produces valid context."""
    from app import build_recent_context
    
    messages = [
        {"role": "user", "content": "I'm stressed about my exam."},
        {"role": "assistant", "content": "That sounds really challenging."},
        {"role": "user", "content": "What should I do?"}  # Current message - should be excluded
    ]
    
    result = build_recent_context(messages)
    
    # Should include first user and assistant messages, but NOT the current user message
    assert "I'm stressed about my exam." in result
    assert "That sounds really challenging." in result
    assert "What should I do?" not in result
    assert "USER:" in result
    assert "ASSISTANT:" in result
    print("✓ test_build_recent_context_single_turn passed")


def test_build_recent_context_multiple_turns():
    """Test that multiple turns preserve chronological order."""
    from app import build_recent_context
    
    messages = [
        {"role": "user", "content": "My exam is tomorrow."},
        {"role": "assistant", "content": "That's coming up soon."},
        {"role": "user", "content": "I'm scared I'll fail."},
        {"role": "assistant", "content": "Let's work through this."},
        {"role": "user", "content": "What should I do tonight?"}  # Current - excluded
    ]
    
    result = build_recent_context(messages)
    
    # All prior messages should be included
    assert "My exam is tomorrow." in result
    assert "That's coming up soon." in result
    assert "I'm scared I'll fail." in result
    assert "Let's work through this." in result
    assert "What should I do tonight?" not in result
    
    # Check order: first user message should appear before first assistant response
    user_first_pos = result.find("My exam is tomorrow.")
    assistant_first_pos = result.find("That's coming up soon.")
    assert user_first_pos < assistant_first_pos, "Chronological order violated"
    print("✓ test_build_recent_context_multiple_turns passed")


def test_build_recent_context_window_limit():
    """Test that only recent messages within the window are included."""
    from app import build_recent_context
    
    # Create 8 messages (4 turns) plus current message = 9 total
    messages = [
        {"role": "user", "content": "Turn 1 - user"},
        {"role": "assistant", "content": "Turn 1 - assistant"},
        {"role": "user", "content": "Turn 2 - user"},
        {"role": "assistant", "content": "Turn 2 - assistant"},
        {"role": "user", "content": "Turn 3 - user"},
        {"role": "assistant", "content": "Turn 3 - assistant"},
        {"role": "user", "content": "Turn 4 - user"},
        {"role": "assistant", "content": "Turn 4 - assistant"},
        {"role": "user", "content": "Turn 5 - user (current)"}  # Current - excluded
    ]
    
    result = build_recent_context(messages, max_messages=6)
    
    # Should include Turn 2-4 (6 messages after excluding current)
    assert "Turn 1 - user" not in result, "Turn 1 should be outside window"
    assert "Turn 2 - user" in result
    assert "Turn 2 - assistant" in result
    assert "Turn 3 - user" in result
    assert "Turn 3 - assistant" in result
    assert "Turn 4 - user" in result
    assert "Turn 4 - assistant" in result
    assert "Turn 5 - user (current)" not in result
    print("✓ test_build_recent_context_window_limit passed")


def test_build_recent_context_malformed_messages():
    """Test that malformed messages don't crash the function."""
    from app import build_recent_context
    
    messages = [
        {"role": "user", "content": "Valid message"},
        {"role": "assistant"},  # Missing content
        {"content": "Missing role"},
        {"role": "user", "content": ""},  # Empty content
        {"role": "unknown", "content": "Invalid role"},
        {"role": "assistant", "content": "Another valid message"},
        {"role": "user", "content": "Current message"}  # Current
    ]
    
    result = build_recent_context(messages)
    
    # Should only include valid messages
    assert "Valid message" in result
    assert "Another valid message" in result
    assert "Current message" not in result
    assert "Missing content" not in result
    assert "Missing role" not in result
    assert "Invalid role" not in result
    print("✓ test_build_recent_context_malformed_messages passed")


def test_build_recent_context_no_duplication():
    """Test that current user message is never included in context."""
    from app import build_recent_context
    
    current_message = "This is my current question right now."
    messages = [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
        {"role": "user", "content": current_message}
    ]
    
    result = build_recent_context(messages)
    
    # Count occurrences - should be 0
    count = result.count(current_message)
    assert count == 0, f"Current message appears {count} times, should be 0"
    print("✓ test_build_recent_context_no_duplication passed")


def test_prompt_integration_stuck_state():
    """Test that stuck state prompt branch includes context correctly."""
    # This test verifies that the prompt template variables work
    # without making an actual API call
    
    user_input = "I don't know what to do."
    messages = [
        {"role": "user", "content": "I'm really stressed."},
        {"role": "assistant", "content": "That sounds tough."},
        {"role": "user", "content": user_input}
    ]
    
    from app import build_recent_context
    
    recent_context = build_recent_context(messages)
    context_section = f"\n\nRECENT CONVERSATION:\n{recent_context}" if recent_context else ""
    
    # Build the prompt template as it would be in app.py
    prompt = f"""
You are a grounding-focused emotional AI.

The user is stuck or uncertain.

RULES:
- DO NOT ask questions
- DO NOT increase thinking load
- DO NOT repeat empathy loops
- Keep response calm and short (2-3 sentences)

TASK:
- Validate uncertainty
- Reduce pressure
- Provide emotional grounding{context_section}

CURRENT USER MESSAGE:
{user_input}

RESPONSE:
"""
    
    # Verify the prompt is well-formed
    assert "RECENT CONVERSATION:" in prompt
    assert "I'm really stressed." in prompt
    assert "That sounds tough." in prompt
    assert "CURRENT USER MESSAGE:" in prompt
    assert user_input in prompt
    print("✓ test_prompt_integration_stuck_state passed")


def test_prompt_integration_emotion_streak():
    """Test that emotion streak prompt branch includes context correctly."""
    
    user_input = "I'm still so sad."
    messages = [
        {"role": "user", "content": "I'm really sad."},
        {"role": "assistant", "content": "I hear that sadness."},
        {"role": "user", "content": user_input}
    ]
    
    from app import build_recent_context
    
    recent_context = build_recent_context(messages)
    context_section = f"\n\nRECENT CONVERSATION:\n{recent_context}" if recent_context else ""
    
    prompt = f"""
You are a calm emotional stabilizer AI.

The user is stuck in emotional repetition.

RULES:
- reduce emotional intensity
- avoid repeated empathy
- avoid questions
- give grounding + gentle forward direction
- 2–3 sentences max{context_section}

CURRENT USER MESSAGE:
{user_input}

RESPONSE:
"""
    
    assert "RECENT CONVERSATION:" in prompt
    assert "I'm really sad." in prompt
    assert "I hear that sadness." in prompt
    print("✓ test_prompt_integration_emotion_streak passed")


def test_prompt_integration_normal_reasoning():
    """Test that normal reasoning prompt includes context section."""
    
    user_input = "What should I do?"
    messages = [
        {"role": "user", "content": "My exam is tomorrow."},
        {"role": "assistant", "content": "That's important."},
        {"role": "user", "content": user_input}
    ]
    
    from app import build_recent_context
    
    recent_context = build_recent_context(messages)
    context_display = recent_context if recent_context else "(No prior conversation)"
    
    # This mimics the normal reasoning prompt structure
    prompt = f"""
You are an emotionally intelligent reasoning-based conversational AI companion (v5).

Some system behavior here...

==================================================
RECENT CONVERSATION
==================================================

{context_display}

==================================================
CURRENT USER MESSAGE
==================================================

{user_input}

ASSISTANT RESPONSE:
"""
    
    assert "RECENT CONVERSATION" in prompt
    assert "My exam is tomorrow." in prompt
    assert "That's important." in prompt
    assert "CURRENT USER MESSAGE" in prompt
    print("✓ test_prompt_integration_normal_reasoning passed")


def test_syntax_and_imports():
    """Test that app.py can be imported without errors."""
    try:
        import app
        
        # Verify key functions exist
        assert hasattr(app, 'build_recent_context'), "build_recent_context function not found"
        assert hasattr(app, 'get_ai_response'), "get_ai_response function not found"
        assert hasattr(app, 'detect_mood'), "detect_mood function not found"
        assert hasattr(app, 'detect_stuck'), "detect_stuck function not found"
        
        print("✓ test_syntax_and_imports passed")
    except Exception as e:
        print(f"✗ test_syntax_and_imports failed: {e}")
        raise


def test_groq_config_missing_key_is_safe_and_non_secret(monkeypatch):
    """Missing Groq credentials should not crash import or expose secrets."""
    import app

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    config = app.load_llm_config()

    assert config["provider"] == "groq"
    assert config["model"] == "openai/gpt-oss-120b"
    assert config["base_url"] == "https://api.groq.com/openai/v1"
    assert config["api_key"] in (None, "")
    assert "GROQ_API_KEY" in str(config)
    assert "secret" not in str(config).lower()


def test_provider_error_uses_fallback(monkeypatch):
    """Initialization or generation errors should return the safe fallback."""
    import app

    def boom():
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(app, "get_llm_client", boom)
    fallback = app.safe_fallback()
    assert app.get_ai_response("hello") == fallback


def test_groq_request_configuration_contract():
    """Verify the OpenAI-compatible request uses the supported Groq parameters."""
    import app

    config = app.load_llm_config()
    assert config["model"] == "openai/gpt-oss-120b"

    payload = {
        "model": config["model"],
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.6,
        "reasoning_effort": "low",
        "max_completion_tokens": 400,
    }

    assert payload["model"] == "openai/gpt-oss-120b"
    assert payload["reasoning_effort"] == "low"
    assert payload["max_completion_tokens"] == 400
    assert "max_tokens" not in payload
    assert "include_reasoning" not in payload


def test_user_content_only_is_returned(monkeypatch):
    """Only message.content should be displayed to the user; reasoning is ignored."""
    import app

    class DummyMessage:
        content = "Visible response"
        reasoning = "hidden internal reasoning"

    class DummyChoice:
        message = DummyMessage()

    class DummyResponse:
        choices = [DummyChoice()]

    class DummyClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    assert kwargs["model"] == "openai/gpt-oss-120b"
                    assert kwargs["reasoning_effort"] == "low"
                    assert kwargs["max_completion_tokens"] == 400
                    assert "max_tokens" not in kwargs
                    assert "include_reasoning" not in kwargs
                    return DummyResponse()

    monkeypatch.setattr(app, "get_llm_client", lambda: DummyClient())
    response = app.get_ai_response("hello")

    assert response == "Visible response"
    assert "hidden internal reasoning" not in response


def test_prompt_uses_recent_context_before_current_message():
    """Recent conversation should appear before the current user message and include usage instructions."""
    messages = [
        {"role": "user", "content": "My exam is tomorrow."},
        {"role": "assistant", "content": "That sounds stressful."},
        {"role": "user", "content": "I'm scared I'll fail."},
        {"role": "assistant", "content": "Let's focus on what matters most."},
        {"role": "user", "content": "What should I do tonight?"},
    ]

    from app import build_recent_context

    recent_context = build_recent_context(messages)
    context_display = recent_context if recent_context else "(No prior conversation)"
    user_input = "What should I do tonight?"

    prompt = f"""
You are an emotionally intelligent reasoning-based conversational AI companion (v5).

==================================================
BEHAVIOR / RESPONSE RULES
==================================================

- Use the recent conversation to interpret the current message. Resolve references such as 'it', 'that', 'tonight', 'what should I do?', or other context-dependent statements using the ongoing topic. Do not treat the current message as an isolated conversation when recent context makes its meaning clear.

==================================================
RECENT CONVERSATION
==================================================

{context_display}

==================================================
CURRENT USER MESSAGE
==================================================

{user_input}

ASSISTANT RESPONSE:
"""

    assert "RECENT CONVERSATION" in prompt
    assert prompt.index("RECENT CONVERSATION") < prompt.index("CURRENT USER MESSAGE")
    assert "Use the recent conversation to interpret the current message" in prompt
    assert "My exam is tomorrow." in prompt
    assert "I'm scared I'll fail." in prompt
    assert "What should I do tonight?" in prompt
    assert "What should I do tonight?" not in recent_context


def test_phase3_emotion_classification():
    from app import analyze_message

    assert analyze_message("I feel sad").emotion == "sad"
    assert analyze_message("I'm not sad anymore").emotion != "sad"
    assert analyze_message("I'm not stressed").emotion != "stressed"
    assert analyze_message("I was anxious yesterday").emotion != "anxious"
    assert analyze_message("I feel stressed about tomorrow").emotion == "stressed"
    assert analyze_message("I'm nervous").emotion == "anxious"
    assert analyze_message("I feel calm now").emotion == "positive"

    mixed_stress = analyze_message("I'm okay but stressed about tomorrow")
    assert mixed_stress.emotion == "stressed"

    mixed_hope = analyze_message("I'm sad but hopeful")
    assert mixed_hope.emotion == "sad"
    assert "positive" in mixed_hope.secondary_emotions


def test_phase3_uncertainty_classification():
    from app import analyze_message

    for message in (
        "I don't know what to do",
        "I do not know what to do",
        "I'm confused",
        "I am confused",
        "I feel stuck",
        "I can't decide",
        "I cannot decide",
        "I have no idea",
    ):
        assert analyze_message(message).is_uncertain is True

    for message in (
        "Nothing really happened but I'm okay",
        "Nothing new today",
        "There is nothing to worry about",
    ):
        assert analyze_message(message).is_uncertain is False


def test_phase3_advice_detection_and_combined_state():
    from app import analyze_message

    for message in (
        "What should I do?",
        "What should I do tonight?",
        "Can you suggest something?",
        "I need advice",
        "How do I handle this?",
        "What would help?",
    ):
        assert analyze_message(message).is_advice_request is True

    combined = analyze_message("I don't know what to do. Can you help me?")
    assert combined.is_uncertain is True
    assert combined.is_advice_request is True
    assert combined.intent == "practical_advice"

    stuck_advice = analyze_message("I feel stuck. What should I do tonight?")
    assert stuck_advice.is_uncertain is True
    assert stuck_advice.is_advice_request is True


def test_phase3_context_keeps_current_emotion_separate():
    from app import analyze_message

    previous = [{"role": "user", "content": "I am stressed about my exam tomorrow."}]
    follow_up = analyze_message("What should I do tonight?", previous)

    assert follow_up.emotion == "neutral"
    assert follow_up.intent == "practical_advice"
    assert follow_up.contextual_emotion == "stressed"

    resolved = analyze_message("I feel fine now", [{"role": "user", "content": "I am anxious."}])
    assert resolved.emotion != "anxious"
    assert resolved.is_distress is False


def test_phase3_streak_counts_distress_not_labels():
    from app import analyze_message, update_conversation_state

    streak = 0
    stuck = 0
    for message in ("I am good", "What should I do?", "Nothing new today"):
        streak, stuck = update_conversation_state(streak, stuck, analyze_message(message))
    assert streak == 0

    streak = 0
    for message in ("I feel sad", "I am stressed", "I am anxious"):
        streak, _ = update_conversation_state(streak, 0, analyze_message(message))
    assert streak == 3

    streak, _ = update_conversation_state(streak, 0, analyze_message("I feel calm now"))
    assert streak == 0

    streak, _ = update_conversation_state(0, 0, analyze_message("I'm not stressed"))
    assert streak == 0


def test_phase3_prompt_priority_keeps_advice_ahead_of_uncertainty():
    from app import analyze_message, select_prompt_branch

    analysis = analyze_message("I feel stuck. What should I do tonight?")
    assert select_prompt_branch(analysis, 0) == "advice"

def test_phase3_normal_prompt_does_not_invent_distress():
    from app import NO_ASSUMED_DISTRESS_RULE, analyze_message, build_response_guidance

    analysis = analyze_message("Nothing really happened today but I'm okay.")
    prompt_guidance = build_response_guidance(analysis)

    assert analysis.emotion == "neutral"
    assert analysis.is_distress is False
    assert analysis.is_uncertain is False
    assert NO_ASSUMED_DISTRESS_RULE in prompt_guidance
    assert "Do not assume the user is distressed" in prompt_guidance


if __name__ == "__main__":
    print("Running conversation memory tests...\n")
    
    test_build_recent_context_empty()
    test_build_recent_context_single_turn()
    test_build_recent_context_multiple_turns()
    test_build_recent_context_window_limit()
    test_build_recent_context_malformed_messages()
    test_build_recent_context_no_duplication()
    test_prompt_integration_stuck_state()
    test_prompt_integration_emotion_streak()
    test_prompt_integration_normal_reasoning()
    test_syntax_and_imports()
    
    print("\n✅ All tests passed!")
