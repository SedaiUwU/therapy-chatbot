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
                    "temperature": 0.8
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
You are a calm, supportive conversational companion.

IMPORTANT RULES:
- Be natural, human-like, and conversational
- Do NOT sound like a therapist or coach
- Do NOT give too many solutions
- 2–5 short sentences only
- Focus on emotional understanding first

CONTEXT:
User mood: {mood}
Previous topic: {st.session_state.last_topic}

USER MESSAGE:
{user_input}

RESPONSE:
"""

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ai_reply = get_ai_response(prompt)

            type_writer(ai_reply)

    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_reply
    })

# ---- SIDEBAR (HACKATHON BOOST FEATURE) ----
with st.sidebar:
    st.header("📊 Mood Tracker")

    if st.session_state.mood_log:
        st.write("Recent moods:")
        st.write(st.session_state.mood_log[-10:])
    else:
        st.write("No data yet")

    st.markdown("---")
    st.write("💡 Tip: Try talking about stress, exams, or relationships.")