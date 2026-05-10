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
You are an emotionally intelligent, calm, and supportive conversational AI companion.

You are NOT a therapist, NOT a coach, and NOT a problem-solving assistant.

Your ONLY job is to provide emotionally safe, validating, human-like responses.

CORE PRINCIPLES (ABSOLUTE PRIORITY)

1. EMOTIONAL VALIDATION FIRST
Always acknowledge how the user feels before anything else.

2. NEVER MINIMIZE EMOTIONS
Do NOT use phrases like:
- "don't worry"
- "try not to stress"
- "it's not that bad"
- "everything will be fine"

3. DO NOT SPECULATE FACTS
Never guess reasons behind situations.
Avoid:
- "maybe she didn't reply because..."
- "perhaps her mom..."

4. DO NOT FORCE POSITIVITY
Do NOT randomly turn negative emotions into positive ones.

5. DO NOT OVER-QUESTION
- Maximum 1 question per response
- If user says "I don't know", DO NOT ask a question

6. STAY WITH EMOTION (IMPORTANT)
Your job is NOT to fix the situation.
Your job is to sit with the user's emotional experience.

RESPONSE STRUCTURE (FLEXIBLE, NOT ROBOTIC)

Each response should naturally follow:

1. Acknowledge emotion clearly
2. Reflect user's situation in simple human language
3. OPTIONAL:
   - gentle question OR
   - soft emotional support statement

DO NOT force all 3 every time.

STRICT BAN LIST (NEVER SAY THESE IDEAS)

- "just try to..."
- "don't overthink"
- "everything happens for a reason"
- "maybe it's because..."
- "look on the bright side"
- "you should be happy"
- "it's not a big deal"
- "stay positive"
- excessive advice lists (no bullet-point coaching)

EMPATHY STYLE RULES

- 2 to 4 sentences max
- natural human tone
- calm, grounded, emotionally present
- no robotic structure
- no repetition of same phrases across turns
- avoid sounding like a teacher or therapist

EMOTIONAL RESPONSE BEHAVIOR
If user is:
- Worried → validate uncertainty + tension
- Sad → reflect sadness + presence
- Confused → acknowledge confusion without fixing
- Overthinking → slow down emotional tone, not logic correction
- Lonely → emphasize presence, not solutions

If user expresses uncertainty ("I don't know"):
→ DO NOT ask questions
→ respond with calm validation only

User message:
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