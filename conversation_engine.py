import re
from dataclasses import dataclass
from typing import Callable

from safety import (
    AMBIGUOUS_CONCERN,
    EXPLICIT_HIGH_RISK,
    THIRD_PARTY_CONCERN,
    classify_safety,
    deterministic_safety_response,
)


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


@dataclass(frozen=True)
class ConversationResult:
    assistant_response: str
    mood: str | None
    emotion_streak: int
    stuck_state: int
    safety_category: str | None = None


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


def build_recent_context(messages, max_messages=6):
    """Build prior user and assistant messages without including the current message."""
    if not messages:
        return ""

    prior_messages = messages[:-1]
    if not prior_messages:
        return ""

    recent = prior_messages[-max_messages:]
    context_lines = []
    for message in recent:
        role = message.get("role", "").strip().upper()
        content = message.get("content", "").strip()
        if role in ("USER", "ASSISTANT") and content:
            context_lines.append(f"{role}: {content}")

    return "\n".join(context_lines)


def _analysis_section(analysis):
    return f"""

    CURRENT MESSAGE ANALYSIS:
    - current emotion: {analysis.emotion}
    - contextual recent emotion: {analysis.contextual_emotion or "none"}
    - intent: {analysis.intent}
    - uncertain: {analysis.is_uncertain}
    - explicit advice request: {analysis.is_advice_request}
    - meaningful distress: {analysis.is_distress}
    """


def _context_section(messages):
    recent_context = build_recent_context(messages)
    return f"\n\nRECENT CONVERSATION:\n{recent_context}" if recent_context else ""


def build_prompt(user_input, messages, analysis, emotion_streak):
    analysis_section = _analysis_section(analysis)
    prompt_branch = select_prompt_branch(analysis, emotion_streak)

    if prompt_branch == "advice":
        return f"""
    You are a supportive mental-wellness conversation companion, not a therapist or diagnostic system.

    BEHAVIOR / RESPONSE RULES
    - Recognize the user's current emotion without overstating it.
    - The current message is authoritative; recent context may clarify the topic or a short-lived contextual emotion.
    - Answer the explicit practical advice request with one or two simple, realistic suggestions.
    - If the user is uncertain, reduce pressure while still providing direction.
    - Do not ask a question immediately after giving advice.
    - Keep the response gentle, natural, and 2-5 sentences.
    {analysis_section}{_context_section(messages)}

    CURRENT USER MESSAGE:
    {user_input}

    ASSISTANT RESPONSE:
    """

    if prompt_branch == "uncertainty":
        return f"""
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
- Provide emotional grounding{_context_section(messages)}
{analysis_section}

CURRENT USER MESSAGE:
{user_input}

ASSISTANT RESPONSE:
"""

    if prompt_branch == "repeated_distress":
        return f"""
You are a calm emotional stabilizer AI.

The user is stuck in emotional repetition.

BEHAVIOR / RESPONSE RULES
- reduce emotional intensity
- avoid repeated empathy
- avoid questions
- give grounding + gentle forward direction
- 2–3 sentences max
- Use the recent conversation to interpret the current message. Resolve references such as 'it', 'that', 'tonight', 'what should I do?', or other context-dependent statements using the ongoing topic. Do not treat the current message as an isolated conversation when recent context makes its meaning clear.{_context_section(messages)}
{analysis_section}

CURRENT USER MESSAGE:
{user_input}

ASSISTANT RESPONSE:
"""

    recent_context = build_recent_context(messages)
    context_display = recent_context if recent_context else "(No prior conversation)"
    response_guidance = build_response_guidance(analysis)
    return f"""
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


class ConversationEngine:
    """Reusable conversation orchestration independent of Streamlit."""

    def __init__(self, response_provider: Callable[[str], str]):
        self.response_provider = response_provider

    def process_message(
        self,
        user_input,
        messages,
        emotion_streak=0,
        stuck_state=0,
    ):
        recent_user_messages = [
            message.get("content", "")
            for message in messages
            if message.get("role") == "user" and message.get("content")
        ][-6:]
        safety_analysis = classify_safety(user_input, recent_user_messages)
        if safety_analysis.category in {
            EXPLICIT_HIGH_RISK,
            AMBIGUOUS_CONCERN,
            THIRD_PARTY_CONCERN,
        }:
            return ConversationResult(
                assistant_response=deterministic_safety_response(safety_analysis),
                mood=None,
                emotion_streak=emotion_streak,
                stuck_state=stuck_state,
                safety_category=safety_analysis.category,
            )

        analysis = analyze_message(user_input, messages)
        emotion_streak, stuck_state = update_conversation_state(
            emotion_streak,
            stuck_state,
            analysis,
        )
        next_messages = [*messages, {"role": "user", "content": user_input}]
        prompt = build_prompt(user_input, next_messages, analysis, emotion_streak)
        return ConversationResult(
            assistant_response=self.response_provider(prompt),
            mood=analysis.emotion,
            emotion_streak=emotion_streak,
            stuck_state=stuck_state,
        )
