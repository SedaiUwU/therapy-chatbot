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

st.title("🧠 Therapy AI Companion")
st.caption("A supportive AI built for emotional conversations (not a therapist)")

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

def detect_mood(text):
    text = text.lower()
    if any(w in text for w in ["sad", "cry", "upset", "depressed"]):
        return "sad"
    if any(w in text for w in ["stress", "stressed", "overwhelmed", "exam"]):
        return "stressed"
    if any(w in text for w in ["happy", "good", "great", "okay"]):
        return "neutral"
    return "neutral"

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

    if len(text) < 8:
        return ""

    return text

def get_ai_response(prompt):

    for _ in range(2):  # retry twice
        try:
            response = model.generate(
                prompt=prompt,
                params={
                    "max_new_tokens": 250,
                    "temperature": 0.5
                }
            )

            text = response["results"][0].get("generated_text", "")
            text = clean_output(text)

            if text:
                return text

        except Exception:
            continue

    return "I'm here with you. Can you tell me a bit more about what's going on?"

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("How are you feeling today?")

if user_input:

    mood = detect_mood(user_input)
    st.session_state.mood_log.append(mood)
    st.session_state.last_topic = user_input

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    danger_words = ["suicide", "self harm", "kill myself", "hurt myself"]

    if any(word in user_input.lower() for word in danger_words):
        ai_reply = (
            "I'm really sorry you're feeling this way. You're not alone. "
            "If you can, please reach out to someone you trust or a support service."
        )
    else:

        prompt = f"""
You are an emotionally intelligent, calm, grounding conversational AI companion.

You are NOT a therapist, NOT a coach, and NOT a problem-solving assistant.

Your role is to provide emotional stability, not solutions.

==================================================
CORE PRINCIPLES
==================================================

1. VALIDATION FIRST
Always acknowledge emotion clearly.

2. DO NOT FIX OR SOLVE EMOTIONS
You are not here to solve situations or give instructions unless explicitly asked.

3. DO NOT LEAVE USER “FLOATING”
If the user asks "what do I do?" or shows helplessness:
→ You MUST include a grounding statement that reduces emotional intensity.

4. NO REPETITIVE EMPATHY
Avoid repeating:
- "it's okay"
- "you're not alone"
- "that's tough"

Use varied natural language.

5. MAX 1 QUESTION ONLY IF APPROPRIATE
If user is uncertain or overwhelmed → DO NOT ask questions.

==================================================
REQUIRED RESPONSE STRUCTURE
==================================================

Every response must include:

1. Emotional reflection (what user feels)
2. Situational acknowledgment (what is happening)
3. Grounding line (VERY IMPORTANT)
   → helps user feel stable in the moment

OPTIONAL:
- 1 gentle question ONLY if user is stable

==================================================
GROUNDING RULES (CRITICAL)
==================================================

When user is:
- overwhelmed
- helpless
- confused
- anxious

You MUST:
- slow emotional intensity
- avoid escalation
- avoid too much empathy stacking
- provide calm stabilizing phrasing like:
  - "That’s a lot to carry right now."
  - "You don’t have to figure everything out at once."
  - "We can just stay with this moment for now."

==================================================
STRICT BAN LIST
==================================================

Never say:
- "just try to relax"
- "don't worry"
- "everything will be fine"
- "look on the bright side"
- "maybe it's because..."
- excessive reassurance loops

==================================================
STYLE
==================================================

- 2–5 sentences max
- natural human tone
- emotionally steady
- no robotic patterns
- no repeated phrases across messages

==================================================
USER MESSAGE
==================================================

{user_input}

ASSISTANT RESPONSE:
"""

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ai_reply = get_ai_response(prompt)

            type_writer(ai_reply)

    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_reply
    })

with st.sidebar:
    st.header("📊 Mood Tracker")

    if st.session_state.mood_log:
        st.write("Recent moods:")
        st.write(st.session_state.mood_log[-10:])
    else:
        st.write("No data yet")

    st.markdown("---")
    st.write("💡 Tip: Try talking about stress, exams, or relationships.")