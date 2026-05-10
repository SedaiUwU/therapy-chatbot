import streamlit as st
import re

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

st.set_page_config(page_title="Therapy AI Chatbot", page_icon="🧠")

st.title("🧠 Therapy AI Chatbot")
st.caption("Supportive AI assistant (not a therapist)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("How are you feeling today?")

def clean_output(text):
    text = re.sub(r"\bI don't know.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bI feel.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bI'm.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bi am.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

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
            "I'm really sorry you're feeling this way. "
            "You're not alone, and talking to someone you trust can really help. "
            "I'm here to support you."
        )

    else:

        prompt = f"""
SYSTEM MODE: SUPPORTIVE CONVERSATIONAL ASSISTANT

You are a calm, natural, human-like supportive chatbot.

Your goal is to have realistic supportive conversations, NOT to constantly give advice.

CORE RULES:

1. DO NOT give advice in every response
2. PRIORITIZE conversation over problem-solving
3. NEVER force action steps in emotional situations
4. NEVER repeat the same suggestion style
5. NEVER sound like a productivity app

RESPONSE RULE:

Choose ONE:
A) Reflect
B) Explore
C) Light suggestion (only if appropriate)

2–4 sentences max.

USER MESSAGE:
{user_input}

ASSISTANT RESPONSE:
"""

        response = model.generate(
            prompt=prompt,
            params={
                "max_new_tokens": 200,
                "temperature": 0.6
            }
        )

        ai_reply = response["results"][0]["generated_text"]
        ai_reply = clean_output(ai_reply)

    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_reply
    })

    with st.chat_message("assistant"):
        st.markdown(ai_reply)