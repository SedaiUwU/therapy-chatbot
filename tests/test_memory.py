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
