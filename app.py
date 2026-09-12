import logging
import os
import re
import time

import streamlit as st
from dotenv import load_dotenv
from conversation_engine import (
    ConversationEngine,
    NO_ASSUMED_DISTRESS_RULE,
    analyze_message,
    build_prompt,
    build_recent_context,
    build_response_guidance,
    classify_emotion,
    detect_advice_request,
    detect_mood,
    detect_stuck,
    detect_uncertainty,
    is_distress_emotion,
    match_phrases,
    normalize_text,
    select_prompt_branch,
    update_conversation_state,
)
from safety import (
    AMBIGUOUS_CONCERN,
    EXPLICIT_HIGH_RISK,
    THIRD_PARTY_CONCERN,
    classify_safety,
    deterministic_safety_response,
)

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

st.set_page_config(
    page_title="Therapy AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {
    --therapy-background: #F5EEE6;
    --therapy-secondary-background: #EDE2D5;
    --therapy-sidebar: rgba(237, 226, 213, 0.82);
    --therapy-surface: rgba(255, 255, 255, 0.68);
    --therapy-assistant-surface: rgba(255, 255, 255, 0.62);
    --therapy-user-surface: rgba(184, 166, 217, 0.28);
    --therapy-blue: #79B9E8;
    --therapy-blue-surface: #DDEBF6;
    --therapy-green: #8FB68E;
    --therapy-green-surface: #DFEADF;
    --therapy-lavender: #B8A6D9;
    --therapy-lavender-surface: #E9E2F3;
    --therapy-peach: #E7A66A;
    --therapy-peach-surface: #F5E1CF;
    --therapy-rose: #D88A8A;
    --therapy-rose-surface: #F3DDDD;
    --therapy-text: #2F2A28;
    --therapy-secondary: #6E625B;
    --therapy-muted: #8B7E75;
    --therapy-border: rgba(110, 98, 91, 0.18);
}

html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"],
[data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {
    background: var(--therapy-background);
}

[data-testid="stAppViewContainer"] {
    background:
    radial-gradient(circle at 15% 20%, rgba(143, 182, 142, 0.20), transparent 32%),
    radial-gradient(circle at 82% 18%, rgba(121, 185, 232, 0.20), transparent 34%),
    radial-gradient(circle at 55% 48%, rgba(184, 166, 217, 0.16), transparent 36%),
    radial-gradient(circle at 85% 85%, rgba(231, 166, 106, 0.17), transparent 32%),
        var(--therapy-background);
}

[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background: transparent;
}

[data-testid="stMainBlockContainer"] p,
[data-testid="stMainBlockContainer"] label,
[data-testid="stMainBlockContainer"] span,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: var(--therapy-secondary);
}

[data-testid="stMainBlockContainer"] {
    max-width: 1180px;
    padding: 2.5rem 2rem 7rem;
}

.therapy-header,
.therapy-empty-state {
    width: min(100%, 820px);
    margin-left: 0;
    margin-right: auto;
}

.therapy-header {
    padding: 0.25rem 0 1.4rem;
}

.therapy-brand {
    margin: 0;
    color: var(--therapy-text);
    font-family: "Aptos", "Segoe UI", sans-serif;
    font-size: clamp(1.9rem, 3vw, 2.15rem);
    font-weight: 700;
    line-height: 1.15;
}

.therapy-brand-mark {
    display: inline-block;
    width: 0.65rem;
    height: 0.65rem;
    margin-right: 0.45rem;
    border-radius: 50%;
    background: var(--therapy-green);
    box-shadow: 0 0 0 5px var(--therapy-green-surface);
    vertical-align: 0.14em;
}

.therapy-tagline {
    margin: 0.55rem 0 0;
    color: var(--therapy-secondary);
    font-family: "Aptos", "Segoe UI", sans-serif;
    font-size: 1rem;
    line-height: 1.55;
}

.therapy-boundary {
    margin: 0.45rem 0 0;
    color: var(--therapy-muted);
    font-family: "Aptos", "Segoe UI", sans-serif;
    font-size: 0.78rem;
    line-height: 1.5;
}

[data-testid="stChatMessage"] {
    display: flex;
    align-items: flex-start;
    width: min(100%, 820px);
    margin: 0 auto 1.15rem;
    padding: 0;
    border: 0;
    border-radius: 0;
    background: transparent;
    box-shadow: none;
    font-family: "Aptos", "Segoe UI", sans-serif;
    font-size: 0.97rem;
    line-height: 1.6;
    overflow-wrap: anywhere;
}

[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
    width: fit-content;
    max-width: 84%;
    min-width: 0;
    flex: 0 1 auto;
    padding: 0.95rem 1.15rem;
    border: 1px solid var(--therapy-border);
    border-radius: 20px;
    background: var(--therapy-assistant-surface);
    backdrop-filter: blur(10px);
    box-shadow: 0 5px 18px rgba(47, 42, 40, 0.07);
    color: var(--therapy-text);
}

[data-testid="stChatMessage"] p {
    color: var(--therapy-text);
    line-height: 1.6;
}

[data-testid="stChatMessage"] [data-testid="stChatMessageContent"]:hover {
    transform: translateY(-1px);
    border-color: rgba(110, 98, 91, 0.28);
    box-shadow: 0 8px 22px rgba(47, 42, 40, 0.1);
}

[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
    justify-content: flex-end;
    margin-right: 0;
    margin-left: 0;
}

[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) > div:first-child {
    display: none !important;
}

[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stChatMessageContent"] {
    max-width: 72%;
    margin: 0;
    border-color: rgba(184, 166, 217, 0.42);
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(231, 166, 106, 0.30), rgba(184, 166, 217, 0.27));
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 14px rgba(47, 42, 40, 0.08);
}

[data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) {
    justify-content: flex-start;
    column-gap: 0.35rem;
    margin-left: 0;
    margin-right: 0;
}

[data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) [data-testid="stChatMessageContent"] {
    max-width: 84%;
    margin: 0;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.68), rgba(221, 235, 246, 0.55), rgba(223, 234, 223, 0.42));
    backdrop-filter: blur(10px);
}

[data-testid="stBottom"] {
    background: transparent;
}

[data-testid="stBottomBlockContainer"] {
    background: transparent;
    padding: 0.25rem 0 1rem;
}

[data-testid="stBottom"] > div {
    background: transparent;
}

[data-testid="stChatInput"] {
    width: min(100%, 820px);
    margin-left: max(0px, calc((100% - 1116px) / 2));
    margin-right: 0;
}

[data-testid="stChatInput"] > div {
    border: 1px solid var(--therapy-border);
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.66);
    backdrop-filter: blur(12px);
    box-shadow: 0 6px 22px rgba(47, 42, 40, 0.08);
}

[data-testid="stChatInput"] textarea {
    color: var(--therapy-text);
    caret-color: var(--therapy-lavender);
    font-family: "Aptos", "Segoe UI", sans-serif;
    font-size: 0.98rem;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--therapy-muted);
    opacity: 1;
}

[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--therapy-blue);
    box-shadow: 0 0 0 2px rgba(121, 185, 232, 0.24);
}

.therapy-empty-state {
    margin-top: 1rem;
    padding: 2.65rem 1.5rem 2.8rem;
    border: 1px solid var(--therapy-border);
    border-radius: 22px;
    background: rgba(255, 255, 255, 0.58);
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 26px rgba(47, 42, 40, 0.07);
    text-align: center;
}

.therapy-empty-state h2 {
    margin: 0;
    color: var(--therapy-text);
    font-family: "Aptos", "Segoe UI", sans-serif;
    font-size: clamp(1.45rem, 2.4vw, 1.8rem);
    line-height: 1.25;
}

.therapy-empty-state p {
    max-width: 560px;
    margin: 0.75rem auto 0;
    color: var(--therapy-secondary);
    font-family: "Aptos", "Segoe UI", sans-serif;
    line-height: 1.55;
}

.therapy-suggestions {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.65rem;
    margin-top: 1.5rem;
}

.therapy-suggestion {
    padding: 0.7rem 1.05rem;
    border: 1px solid rgba(121, 185, 232, 0.42);
    border-radius: 999px;
    background: rgba(221, 235, 246, 0.82);
    color: var(--therapy-text);
    font-family: "Aptos", "Segoe UI", sans-serif;
    font-size: 0.88rem;
    transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
}

.therapy-suggestion:hover {
    transform: translateY(-1px);
    border-color: rgba(110, 98, 91, 0.28);
    box-shadow: 0 5px 14px rgba(47, 42, 40, 0.08);
}

.therapy-suggestion:nth-child(2) {
    background: rgba(223, 234, 223, 0.84);
    border-color: rgba(143, 182, 142, 0.45);
}

.therapy-suggestion:nth-child(3) {
    background: rgba(233, 226, 243, 0.84);
    border-color: rgba(184, 166, 217, 0.46);
}

.therapy-suggestion:nth-child(4) {
    background: rgba(245, 225, 207, 0.86);
    border-color: rgba(231, 166, 106, 0.46);
}

[data-testid="stSidebar"] {
    border-right: 1px solid var(--therapy-border);
    background: linear-gradient(180deg, rgba(223, 234, 223, 0.78), rgba(245, 225, 207, 0.55));
    backdrop-filter: blur(10px);
}

[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    background: transparent;
    padding: 1.5rem 1.15rem;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--therapy-text);
    font-family: "Aptos", "Segoe UI", sans-serif;
    font-weight: 700;
}

.mood-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin-top: 0.75rem;
}

.mood-chip {
    padding: 0.55rem 0.7rem;
    border: 1px solid var(--therapy-border);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.58);
    color: var(--therapy-text);
    font-family: "Aptos", "Segoe UI", sans-serif;
    font-size: 0.88rem;
}

.mood-chip:nth-child(3n + 1) {
    background: rgba(221, 235, 246, 0.72);
    border-color: rgba(121, 185, 232, 0.34);
}

.mood-chip:nth-child(3n + 2) {
    background: rgba(223, 234, 223, 0.74);
    border-color: rgba(143, 182, 142, 0.36);
}

.mood-chip:nth-child(3n) {
    background: rgba(233, 226, 243, 0.74);
    border-color: rgba(184, 166, 217, 0.38);
}

.mood-chip.mood-stressed,
.mood-chip.mood-sad {
    background: linear-gradient(135deg, rgba(245, 225, 207, 0.82), rgba(233, 226, 243, 0.78));
    border-color: rgba(184, 166, 217, 0.42);
}

.mood-chip.mood-anxious {
    background: linear-gradient(135deg, rgba(221, 235, 246, 0.84), rgba(233, 226, 243, 0.76));
    border-color: rgba(121, 185, 232, 0.38);
}

.mood-chip.mood-positive {
    background: rgba(223, 234, 223, 0.86);
    border-color: rgba(143, 182, 142, 0.42);
}

.mood-chip.mood-neutral {
    background: rgba(221, 235, 246, 0.76);
    border-color: rgba(121, 185, 232, 0.34);
}

.sidebar-tip,
.sidebar-boundary {
    color: var(--therapy-secondary) !important;
    font-family: "Aptos", "Segoe UI", sans-serif;
    font-size: 0.82rem;
    line-height: 1.5;
}

@media (prefers-reduced-motion: reduce) {
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"],
    .therapy-suggestion {
        transition: none;
    }

    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"]:hover,
    .therapy-suggestion:hover {
        transform: none;
    }
}

@media (max-width: 700px) {
    [data-testid="stMainBlockContainer"] {
        padding: 1.5rem 0.85rem 6.5rem;
    }

    .therapy-header {
        padding-bottom: 1.5rem;
    }

    [data-testid="stChatMessage"],
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        width: 100%;
        margin-left: auto;
        margin-right: auto;
    }

    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) [data-testid="stChatMessageContent"] {
        max-width: 94%;
        padding: 0.85rem 0.95rem;
    }

    [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stChatMessageContent"] {
        max-width: 90%;
    }

    [data-testid="stChatInput"] {
        width: min(100%, calc(100% - 1.7rem));
        margin-left: auto;
        margin-right: auto;
    }

    .therapy-empty-state {
        margin-top: 1.5rem;
        padding-left: 0;
        padding-right: 0;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<header class="therapy-header">
    <h1 class="therapy-brand"><span class="therapy-brand-mark" aria-hidden="true"></span>Therapy AI</h1>
    <p class="therapy-tagline">Your space to talk, reflect, and think things through.</p>
    <p class="therapy-boundary">AI wellness companion &bull; Not a therapist or emergency service</p>
</header>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "mood_log" not in st.session_state:
    st.session_state.mood_log = []

if "emotion_streak" not in st.session_state:
    st.session_state.emotion_streak = 0

if "stuck_state" not in st.session_state:
    st.session_state.stuck_state = 0


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


def get_safety_response(user_input, recent_messages):
    recent_user_messages = [
        message.get("content", "")
        for message in recent_messages
        if message.get("role") == "user" and message.get("content")
    ][-6:]
    analysis = classify_safety(user_input, recent_user_messages)
    if analysis.category in {EXPLICIT_HIGH_RISK, AMBIGUOUS_CONCERN, THIRD_PARTY_CONCERN}:
        return deterministic_safety_response(analysis)
    return None


user_input = st.chat_input("Share what's on your mind...")

if not st.session_state.messages and not user_input:
    st.markdown("""
    <section class="therapy-empty-state" aria-label="Welcome">
        <h2>What's on your mind today?</h2>
        <p>You can talk things through, reflect, or ask for a little direction.</p>
        <div class="therapy-suggestions" aria-label="Conversation ideas">
            <span class="therapy-suggestion">Talk it out</span>
            <span class="therapy-suggestion">Help me decide</span>
            <span class="therapy-suggestion">Check in</span>
            <span class="therapy-suggestion">I'm feeling stressed</span>
        </div>
    </section>
    """, unsafe_allow_html=True)


for msg in st.session_state.messages:
    avatar = "✨" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


conversation_engine = ConversationEngine(get_ai_response)

if user_input:
    engine_result = conversation_engine.process_message(
        user_input,
        st.session_state.messages,
        st.session_state.emotion_streak,
        st.session_state.stuck_state,
    )
    ai_reply = engine_result.assistant_response

    if engine_result.mood is not None:
        st.session_state.mood_log.append(engine_result.mood)
        st.session_state.emotion_streak = engine_result.emotion_streak
        st.session_state.stuck_state = engine_result.stuck_state

    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_reply,
    })

    with st.chat_message("assistant", avatar="✨"):
        type_writer(ai_reply)


with st.sidebar:
    st.header("Recent moods")

    if st.session_state.mood_log:
        mood_labels = {
            "sad": "😔 Sad",
            "stressed": "😟 Stressed",
            "anxious": "😰 Anxious",
            "positive": "🌿 Positive",
            "neutral": "😐 Neutral",
        }
        mood_chips = "".join(
            f'<div class="mood-chip mood-{mood}">{mood_labels.get(mood, "😐 Neutral")}</div>'
            for mood in st.session_state.mood_log[-5:]
        )
        st.markdown(f'<div class="mood-list">{mood_chips}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="sidebar-tip">Your recent check-ins will appear here.</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<p class="sidebar-tip">Try talking about stress, exams, relationships, or anything on your mind.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sidebar-boundary">AI wellness companion. Not a therapist or emergency service.</p>',
        unsafe_allow_html=True,
    )