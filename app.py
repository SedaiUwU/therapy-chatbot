import logging
import os
import re
import time
from dataclasses import dataclass

import streamlit as st
from dotenv import load_dotenv
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
    margin-left: auto;
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
    background: var(--therapy-background);
}

[data-testid="stBottomBlockContainer"] {
    background: var(--therapy-background);
    padding: 0.25rem 0 1rem;
}

[data-testid="stChatInput"] {
    width: min(100%, 820px);
    margin: 0 auto;
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


@dataclass(frozen=True)
class MessageAnalysis:
    emotion: str
    emotion_signals: tuple[str, ...]
    secondary_emotions: tuple[str, ...]
    intent: str
    is_uncertain: bool
    is_advice_request: bool
    is_distress: bool
    is_explicitly_resolved: bool
    contextual_emotion: str | None = None


EMOTION_PHRASES = {
    "sad": ("sad", "down", "unhappy", "cry", "upset"),
    "stressed": ("stress", "stressed", "overwhelmed", "tense"),
    "anxious": ("anxious", "worried", "nervous", "scared"),
    "positive": ("good", "happy", "calm", "relieved", "hopeful"),
}

DISTRESS_EMOTIONS = {"sad", "stressed", "anxious"}
NO_ASSUMED_DISTRESS_RULE = (
    "Do not assume the user is distressed, overwhelmed, struggling, or carrying something difficult "
    "unless the current message or relevant recent context provides evidence. For neutral or positive "
    "ordinary conversation, respond naturally and proportionately rather than forcing therapeutic reassurance."
)

UNCERTAINTY_PHRASES = (
    "i don't know what to do",
    "i do not know what to do",
    "i'm confused",
    "i am confused",
    "i feel stuck",
    "i can't decide",
    "i cannot decide",
    "i have no idea",
)

ADVICE_PHRASES = (
    "what should i do",
    "what can i do",
    "can you suggest something",
    "can you give me advice",
    "can you help me",
    "i need advice",
    "how do i handle this",
    "what would help",
)


def normalize_text(text):
    """Normalize user text for small, deterministic phrase checks."""
    text = (text or "").lower().replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def match_phrases(text, phrases):
    """Match complete words or phrases, avoiding unrestricted substrings."""
    normalized = normalize_text(text)
    return [
        phrase for phrase in phrases
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized)
    ]


def _is_negated_or_historical(text, start):
    prefix = normalize_text(text[:start])
    recent_words = prefix.split()[-5:]
    recent = " ".join(recent_words)
    return (
        re.search(r"\b(?:not|no longer)\s*$", recent) is not None
        or re.search(r"\b(?:don't|do not)\s+(?:feel|seem)\s*$", recent) is not None
        or re.search(r"\b(?:was|were|used to be)\s*$", recent) is not None
    )


def _emotion_signals(text):
    normalized = normalize_text(text)
    signals = {emotion: [] for emotion in EMOTION_PHRASES}
    for emotion, phrases in EMOTION_PHRASES.items():
        for phrase in phrases:
            for match in re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized):
                if not _is_negated_or_historical(normalized, match.start()):
                    signals[emotion].append(phrase)
    return signals


def classify_emotion(text):
    signals = _emotion_signals(text)
    detected = [emotion for emotion, phrases in signals.items() if phrases]
    primary = next(
        (emotion for emotion in ("sad", "stressed", "anxious", "positive") if emotion in detected),
        "neutral",
    )
    secondary = tuple(emotion for emotion in detected if emotion != primary)
    return primary, tuple(detected), secondary


def detect_mood(text):
    return classify_emotion(text)[0]


def detect_uncertainty(text):
    return bool(match_phrases(text, UNCERTAINTY_PHRASES))


def detect_stuck(text):
    return detect_uncertainty(text)


def detect_advice_request(text):
    return bool(match_phrases(text, ADVICE_PHRASES))


def _is_explicitly_resolved(text):
    normalized = normalize_text(text)
    return bool(
        re.search(r"\b(?:not|no longer)\s+(?:feel\s+)?(?:sad|stressed|anxious)\b", normalized)
        or re.search(r"\b(?:don't|do not)\s+feel\s+(?:sad|stressed|anxious)\b", normalized)
        or re.search(r"\b(?:was|were|used to be)\s+(?:sad|stressed|anxious)\b", normalized)
        or re.search(r"\b(?:feel|am|i'm)\s+(?:calm|relieved)\b", normalized)
    )


def _recent_context_emotion(messages):
    for message in reversed(messages[-6:]):
        if message.get("role") != "user":
            continue
        emotion = detect_mood(message.get("content", ""))
        if emotion != "neutral":
            return emotion
    return None


def analyze_message(text, recent_messages=None):
    emotion, signals, secondary = classify_emotion(text)
    is_advice_request = detect_advice_request(text)
    is_uncertain = detect_uncertainty(text)
    is_resolved = _is_explicitly_resolved(text)
    return MessageAnalysis(
        emotion=emotion,
        emotion_signals=signals,
        secondary_emotions=secondary,
        intent="practical_advice" if is_advice_request else (
            "emotional_support" if emotion in DISTRESS_EMOTIONS else "normal_conversation"
        ),
        is_uncertain=is_uncertain,
        is_advice_request=is_advice_request,
        is_distress=emotion in DISTRESS_EMOTIONS and not is_resolved,
        is_explicitly_resolved=is_resolved,
        contextual_emotion=_recent_context_emotion(recent_messages or []),
    )


def is_distress_emotion(emotion):
    return emotion in DISTRESS_EMOTIONS


def update_conversation_state(emotion_streak, stuck_state, analysis):
    if analysis.is_distress:
        emotion_streak += 1
    else:
        emotion_streak = 0

    stuck_state = stuck_state + 1 if analysis.is_uncertain else 0
    return emotion_streak, stuck_state


def select_prompt_branch(analysis, emotion_streak):
    if analysis.is_advice_request:
        return "advice"
    if analysis.is_distress and emotion_streak >= 2:
        return "repeated_distress"
    if analysis.is_distress:
        return "emotional_support"
    if analysis.is_uncertain:
        return "uncertainty"
    return "normal"


def build_response_guidance(analysis):
    if (
        analysis.emotion in {"neutral", "positive"}
        and not analysis.is_distress
        and not analysis.is_uncertain
        and analysis.contextual_emotion not in DISTRESS_EMOTIONS
    ):
        return NO_ASSUMED_DISTRESS_RULE
    return ""


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


if user_input:

    ai_reply = get_safety_response(user_input, st.session_state.messages)

    if ai_reply is not None:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
    else:

        analysis = analyze_message(user_input, st.session_state.messages)
        mood = analysis.emotion
        emotion_streak, stuck_state = update_conversation_state(
            st.session_state.emotion_streak,
            st.session_state.stuck_state,
            analysis,
        )

        st.session_state.mood_log.append(mood)
        st.session_state.emotion_streak = emotion_streak
        st.session_state.stuck_state = stuck_state

        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # ==============================
        # EMOTIONAL REASONING ENGINE v6
        # ==============================

        recent_context = build_recent_context(st.session_state.messages)
        context_section = f"\n\nRECENT CONVERSATION:\n{recent_context}" if recent_context else ""
        analysis_section = f"""

    CURRENT MESSAGE ANALYSIS:
    - current emotion: {analysis.emotion}
    - contextual recent emotion: {analysis.contextual_emotion or "none"}
    - intent: {analysis.intent}
    - uncertain: {analysis.is_uncertain}
    - explicit advice request: {analysis.is_advice_request}
    - meaningful distress: {analysis.is_distress}
    """
        response_guidance = build_response_guidance(analysis)
        prompt_branch = select_prompt_branch(analysis, st.session_state.emotion_streak)

        if prompt_branch == "advice":
            prompt = f"""
    You are a supportive mental-wellness conversation companion, not a therapist or diagnostic system.

    BEHAVIOR / RESPONSE RULES
    - Recognize the user's current emotion without overstating it.
    - The current message is authoritative; recent context may clarify the topic or a short-lived contextual emotion.
    - Answer the explicit practical advice request with one or two simple, realistic suggestions.
    - If the user is uncertain, reduce pressure while still providing direction.
    - Do not ask a question immediately after giving advice.
    - Keep the response gentle, natural, and 2-5 sentences.
    {analysis_section}{context_section}

    CURRENT USER MESSAGE:
    {user_input}

    ASSISTANT RESPONSE:
    """

        elif prompt_branch == "uncertainty":
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
{analysis_section}

CURRENT USER MESSAGE:
{user_input}

ASSISTANT RESPONSE:
"""

        elif prompt_branch == "repeated_distress":
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
{analysis_section}

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
- Treat the current message as authoritative. Use contextual recent emotion only to interpret an ambiguous follow-up.
- {response_guidance}

==================================================
STUCK STATE RULE (HIGHEST PRIORITY)
==================================================

If user is uncertain or says:
- "I don't know"
- "not sure"

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

{analysis_section}

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