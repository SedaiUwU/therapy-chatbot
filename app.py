import logging
import os
import re
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def get_secret(key, default=None):
    try:
        secrets = st.secrets
    except Exception:
        secrets = {}

    if isinstance(secrets, dict) and key in secrets and secrets[key] not in (None, ""):
        return secrets[key]

    value = st.session_state.get(key) if hasattr(st, "session_state") else None
    if value not in (None, ""):
        return value

    value = os.getenv(key)
    if value not in (None, ""):
        return value

    return default


def load_llm_config():
    api_key = get_secret("GROQ_API_KEY")
    config = {
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": api_key,
        "key_name": "GROQ_API_KEY",
    }
    if not api_key:
        config["missing_key"] = "GROQ_API_KEY"
    return config


_llm_client = None


def get_llm_client():
    global _llm_client

    if _llm_client is not None:
        return _llm_client

    config = load_llm_config()
    api_key = config["api_key"]

    if not api_key:
        logger.warning(
            "Groq API key not configured. Set GROQ_API_KEY in the local environment or Streamlit secrets."
        )
        return None

    try:
        from openai import OpenAI

        _llm_client = OpenAI(api_key=api_key, base_url=config["base_url"])
        return _llm_client
    except Exception:
        logger.exception("Failed to initialize the Groq/OpenAI-compatible client.")
        return None

st.set_page_config(page_title="Therapy AI", page_icon="🧠", layout="centered")

st.title("🧠 Therapy AI Companion")
st.caption("Emotionally intelligent conversational AI (not a therapist)")

st.markdown("""
<style>
.stChatMessage {
    padding: 12px;
    border-radius: 14px;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "mood_log" not in st.session_state:
    st.session_state.mood_log = []

if "emotion_streak" not in st.session_state:
    st.session_state.emotion_streak = 0

if "stuck_state" not in st.session_state:
    st.session_state.stuck_state = 0


def detect_mood(text):
    text = text.lower()
    if any(w in text for w in ["sad", "cry", "upset", "depressed"]):
        return "sad"
    if any(w in text for w in ["stress", "stressed", "overwhelmed", "exam"]):
        return "stressed"
    if any(w in text for w in ["worried", "anxious", "scared"]):
        return "anxious"
    if any(w in text for w in ["happy", "good", "great", "okay"]):
        return "neutral"
    return "neutral"


def detect_stuck(text):
    text = text.lower()
    return any(w in text for w in [
        "i don't know",
        "idk",
        "not sure",
        "nothing",
        "i'm not sure",
        "i have no idea"
    ])


def type_writer(text):
    placeholder = st.empty()
    typed = ""
    for char in text:
        typed += char
        time.sleep(0.01)
        placeholder.markdown(typed)


def clean_output(text):
    if not text:
        return ""

    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    if len(text) < 10:
        return ""

    return text


def safe_fallback():
    return "I hear you. That sounds like a lot to carry right now. We can just stay with this moment together."


def build_recent_context(messages, max_messages=6):
    """
    Build recent conversation context from session messages.
    
    Excludes the current (latest) user message to avoid duplication.
    Formats with clear role labels for the model.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        max_messages: Maximum number of recent messages to include
        
    Returns:
        Formatted context string or empty string if no prior messages
    """
    if not messages:
        return ""
    
    # Exclude the current user message (just added)
    prior_messages = messages[:-1]
    
    if not prior_messages:
        return ""
    
    # Keep only the last max_messages messages
    recent = prior_messages[-max_messages:]
    
    context_lines = []
    for msg in recent:
        role = msg.get("role", "").strip().upper()
        content = msg.get("content", "").strip()
        
        # Only include valid user and assistant messages with content
        if role in ("USER", "ASSISTANT") and content:
            context_lines.append(f"{role}: {content}")
    
    if not context_lines:
        return ""
    
    return "\n".join(context_lines)


def get_ai_response(prompt):
    config = load_llm_config()

    if not config.get("api_key"):
        logger.warning(
            "Groq is not configured. Set GROQ_API_KEY in the local environment or Streamlit secrets before starting the app."
        )
        return safe_fallback()

    try:
        client = get_llm_client()
    except Exception:
        logger.exception("Groq client initialization failed; using fallback response.")
        return safe_fallback()

    if client is None:
        logger.warning("Groq client unavailable; using fallback response.")
        return safe_fallback()

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=config["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                reasoning_effort="low",
                max_completion_tokens=400,
            )

            content = ""
            if getattr(response, "choices", None):
                message = response.choices[0].message
                content = getattr(message, "content", "") or ""
                if isinstance(content, list):
                    content = "".join(part.get("text", "") for part in content if isinstance(part, dict))

            text = clean_output(content)
            if text:
                return text

        except Exception as exc:  # pragma: no cover - exercised in runtime failures only
            logger.warning("Groq generation failed on attempt %s: %s", attempt + 1, type(exc).__name__)
            if attempt == 1:
                logger.exception("Final Groq generation attempt failed.")

    return safe_fallback()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_input = st.chat_input("How are you feeling today?")

if user_input:

    mood = detect_mood(user_input)
    stuck = detect_stuck(user_input)

    st.session_state.mood_log.append(mood)

    # emotion streak tracking
    if len(st.session_state.mood_log) > 1 and st.session_state.mood_log[-1] == st.session_state.mood_log[-2]:
        st.session_state.emotion_streak += 1
    else:
        st.session_state.emotion_streak = 0

    # stuck state tracking
    if stuck:
        st.session_state.stuck_state += 1
    else:
        st.session_state.stuck_state = 0

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    danger_words = ["suicide", "self harm", "kill myself", "hurt myself"]

    if any(word in user_input.lower() for word in danger_words):
        ai_reply = (
            "I'm really sorry you're feeling this way. You're not alone, and what you're feeling matters. "
            "If possible, please reach out to someone you trust or a support service right now."
        )

    else:

        # ==============================
        # EMOTIONAL REASONING ENGINE v5
        # ==============================

        # STUCK STATE OVERRIDE (CRITICAL FIX)
        if st.session_state.stuck_state >= 1:
            recent_context = build_recent_context(st.session_state.messages)
            context_section = f"\n\nRECENT CONVERSATION:\n{recent_context}" if recent_context else ""
            
            prompt = f"""
You are a grounding-focused emotional AI.

BEHAVIOR / RESPONSE RULES
- DO NOT ask questions
- DO NOT increase thinking load
- DO NOT repeat empathy loops
- Keep response calm and short (2-3 sentences)
- Use the recent conversation to interpret the current message. Resolve references such as 'it', 'that', 'tonight', 'what should I do?', or other context-dependent statements using the ongoing topic. Do not treat the current message as an isolated conversation when recent context makes its meaning clear.

TASK:
- Validate uncertainty
- Reduce pressure
- Provide emotional grounding{context_section}

CURRENT USER MESSAGE:
{user_input}

ASSISTANT RESPONSE:
"""

        # EMOTION LOOP PROTECTION
        elif st.session_state.emotion_streak >= 2:
            recent_context = build_recent_context(st.session_state.messages)
            context_section = f"\n\nRECENT CONVERSATION:\n{recent_context}" if recent_context else ""
            
            prompt = f"""
You are a calm emotional stabilizer AI.

The user is stuck in emotional repetition.

BEHAVIOR / RESPONSE RULES
- reduce emotional intensity
- avoid repeated empathy
- avoid questions
- give grounding + gentle forward direction
- 2–3 sentences max
- Use the recent conversation to interpret the current message. Resolve references such as 'it', 'that', 'tonight', 'what should I do?', or other context-dependent statements using the ongoing topic. Do not treat the current message as an isolated conversation when recent context makes its meaning clear.{context_section}

CURRENT USER MESSAGE:
{user_input}

ASSISTANT RESPONSE:
"""

        else:
            recent_context = build_recent_context(st.session_state.messages)
            context_display = recent_context if recent_context else "(No prior conversation)"

            prompt = f"""
You are an emotionally intelligent reasoning-based conversational AI companion (v5).

==================================================
BEHAVIOR / RESPONSE RULES
==================================================

You MUST dynamically choose:

1. EMOTIONAL REFLECTION (acknowledge)
2. SITUATIONAL CLARITY (what is happening)
3. GROUNDING OR DIRECTION (reduce emotional load OR gently move forward)

- Use the recent conversation to interpret the current message. Resolve references such as 'it', 'that', 'tonight', 'what should I do?', or other context-dependent statements using the ongoing topic. Do not treat the current message as an isolated conversation when recent context makes its meaning clear.
- NEVER loop empathy phrases
- NEVER ask multiple questions
- NEVER ignore confusion states
- NEVER escalate emotion
- NEVER behave like a therapist or coach
- NEVER overload advice

==================================================
STUCK STATE RULE (HIGHEST PRIORITY)
==================================================

If user is uncertain or says:
- "I don't know"
- "not sure"
- "nothing"

THEN:
- NO QUESTIONS ALLOWED
- reduce emotional pressure
- normalize uncertainty
- provide calm grounding

==================================================
OUTPUT STYLE
==================================================

- 2–5 sentences max
- natural human tone
- emotionally stable
- non-repetitive
- smooth flow (not robotic)

==================================================
GROUNDING PHRASES (USE VARIATION)
==================================================

- "That’s a lot to hold right now."
- "You don’t need to figure this out immediately."
- "We can just stay with this moment."

==================================================
USER INTENT OVERRIDE RULE (CRITICAL FIX)
==================================================

If the user explicitly asks for help, such as:
- "what can I do?"
- "can you suggest something?"
- "give me advice"
- "help me"

THEN YOU MUST:

✔ provide at least ONE simple, practical suggestion
✔ keep it emotionally gentle
✔ do NOT switch only to grounding
✔ do NOT ask another question immediately after

You may still include empathy, but MUST include direction.

Example format:
- acknowledgment
- 1 simple suggestion
- optional gentle reassurance line

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

        ai_reply = get_ai_response(prompt)

    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_reply
    })

    with st.chat_message("assistant"):
        type_writer(ai_reply)


with st.sidebar:
    st.header("📊 Mood Tracker")

    if st.session_state.mood_log:
        st.write("Recent moods:")
        st.write(st.session_state.mood_log[-10:])
    else:
        st.write("No data yet")

    st.markdown("---")
    st.write("Tip: Try talking about stress, exams, relationships, or anything on your mind.")