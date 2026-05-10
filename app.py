import streamlit as st
import re
import time

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

st.set_page_config(page_title="Therapy AI Chatbot", page_icon="🧠", layout="centered")

st.title("🧠 Therapy AI Chatbot")
st.caption("A supportive AI companion powered by IBM Watsonx (not a therapist)")
st.info("You can talk about stress, exams, relationships, or anything on your mind.")

st.markdown("""
<style>
.stChatMessage {
    padding: 10px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

def type_writer(text):
    placeholder = st.empty()
    typed = ""
    for char in text:
        typed += char
        time.sleep(0.01)
        placeholder.markdown(typed)

def clean_output(text):
    text = re.sub(r"\bI don't know.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bI feel.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bI'm.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bi am.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("How are you feeling today?")

if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    danger_words = ["suicide", "self harm", "kill myself", "hurt myself"]

    if any(word in user_input.lower() for word in danger_words):
        ai_reply = (
            "I'm really sorry you're feeling this way. You're not alone, and it may help to reach out to someone you trust or a support service."
        )

    else:

        prompt = f"""
You are a calm, supportive conversational AI companion.

GOALS:
- Provide emotional support
- Keep conversation natural and human-like
- Avoid being overly robotic or instructional

RULES:
- 2–5 short sentences
- Prioritize empathy over advice
- Do NOT overuse suggestions or steps
- Do NOT sound like a productivity or coaching app
- Keep tone warm and human

STYLE:
- Simple language
- Natural flow
- Gentle responses

User message:
{user_input}

Response:
"""

        response = model.generate(
            prompt=prompt,
            params={
                "max_new_tokens": 200,
                "temperature": 0.7
            }
        )

        ai_reply = response["results"][0]["generated_text"]
        ai_reply = clean_output(ai_reply)

    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_reply
    })

    with st.chat_message("assistant"):
        type_writer(ai_reply)