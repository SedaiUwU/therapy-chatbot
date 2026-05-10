import streamlit as st
import re
import time
import random

from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai import Credentials

API_KEY = st.secrets["IBM_API_KEY"]
PROJECT_ID = st.secrets["IBM_PROJECT_ID"]
URL = st.secrets["IBM_URL"]

credentials = Credentials(
    url=URL,
    api_key=API_KEY
)

model = Model(
    model_id="mistralai/mistral-small-3-1-24b-instruct-2503",
    credentials=credentials,
    project_id=PROJECT_ID
)

st.set_page_config(page_title="Therapy AI", page_icon="🧠", layout="centered")

st.title("🧠 Therapy AI Companion (v5 Engine)")
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

if "last_topic" not in st.session_state:
    st.session_state.last_topic = ""

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


def get_ai_response(prompt):
    for _ in range(2):
        try:
            response = model.generate(
                prompt=prompt,
                params={
                    "max_new_tokens": 250,
                    "temperature": 0.6
                }
            )

            text = response["results"][0].get("generated_text", "")
            text = clean_output(text)

            if text:
                return text

        except Exception:
            continue

    return safe_fallback()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_input = st.chat_input("How are you feeling today?")

if user_input:

    mood = detect_mood(user_input)
    stuck = detect_stuck(user_input)

    st.session_state.mood_log.append(mood)
    st.session_state.last_topic = user_input

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
- Provide emotional grounding

USER MESSAGE:
{user_input}

RESPONSE:
"""

        # EMOTION LOOP PROTECTION
        elif st.session_state.emotion_streak >= 2:
            prompt = f"""
You are a calm emotional stabilizer AI.

The user is stuck in emotional repetition.

RULES:
- reduce emotional intensity
- avoid repeated empathy
- avoid questions
- give grounding + gentle forward direction
- 2–3 sentences max

USER MESSAGE:
{user_input}

RESPONSE:
"""

        else:

            prompt = f"""
You are an emotionally intelligent reasoning-based conversational AI companion (v5).

==================================================
CORE SYSTEM BEHAVIOR
==================================================

You MUST dynamically choose:

1. EMOTIONAL REFLECTION (acknowledge)
2. SITUATIONAL CLARITY (what is happening)
3. GROUNDING OR DIRECTION (reduce emotional load OR gently move forward)

==================================================
CRITICAL RULES
==================================================

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
USER MESSAGE
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