import re
from dataclasses import dataclass


NORMAL = "normal"
THIRD_PARTY_CONCERN = "third_party_concern"
AMBIGUOUS_CONCERN = "ambiguous_concern"
EXPLICIT_HIGH_RISK = "explicit_high_risk"


@dataclass(frozen=True)
class SafetyAnalysis:
    category: str
    matched_signal: str | None = None
    scope: str = "current"
    source: str = "current_message"


EXPLICIT_PATTERNS = (
    r"\bi\s+want\s+to\s+(?:kill|hurt|harm)\s+myself\b",
    r"\bi(?:\s+am|'m)\s+going\s+to\s+(?:kill|hurt|harm)\s+myself\b",
    r"\bi(?:\s+am|'m)\s+thinking\s+(?:about|of)\s+(?:suicide|killing|hurting|harming)\b",
    r"\bi(?:'ve|\s+have)\s+(?:also\s+)?been\s+thinking\s+(?:about|of)\s+(?:suicide|killing|hurting|harming)\b",
    r"\bi\s+might\s+(?:kill|hurt|harm)\s+myself\b",
    r"\bi\s+(?:want|plan)\s+to\s+end\s+my\s+life\b",
    r"\bi\s+(?:want|plan)\s+to\s+take\s+my\s+own\s+life\b",
    r"\bi(?:\s+am|'m)\s+suicidal\b",
    r"\bi\s+have\s+(?:suicidal\s+thoughts|a\s+plan\s+to\s+(?:kill|hurt|harm)\s+myself)\b",
    r"\bi\s+want\s+to\s+die\b",
)

THIRD_PARTY_PATTERN = re.compile(
    r"\b(?:my\s+(?:friend|sister|brother|partner)|a\s+friend|someone\s+i\s+know)\s+"
    r"(?:says?|said|told\s+me)\s+"
    r"(?:they|he|she)\s+(?:want(?:s)?\s+to\s+die|want(?:s)?\s+to\s+(?:kill|hurt|harm)\s+(?:themselves|himself|herself)|"
    r"might\s+(?:hurt|harm|kill)\s+themselves|(?:is|are)\s+suicidal)")

REFERENCE_PATTERNS = (
    r"\bwhat\s+does\s+suicide\s+mean\b",
    r"\bi(?:\s+am|'m)\s+writing\s+(?:an\s+)?essay\s+(?:about|on)\s+suicide\b",
    r"\bsuicide\s+prevention\s+resources\b",
    r"\bthe\s+movie\s+(?:discussed|mentioned|depicted)\s+suicide\b",
)

HISTORICAL_MARKERS = (
    r"\bused\s+to\b",
    r"\b(?:last|this)\s+year\b",
    r"\bwhen\s+i\s+was\s+younger\b",
    r"\bin\s+the\s+past\b",
    r"\bno\s+longer\b",
    r"\bdon't\s+now\b",
    r"\bdo\s+not\s+now\b",
)

NEGATED_PATTERNS = (
    r"\bi\s+(?:don't|do\s+not)\s+want\s+to\s+(?:kill|hurt|harm)\s+myself\b",
    r"\bi(?:\s+am|'m)\s+not\s+suicidal\b",
    r"\bi\s+(?:don't|do\s+not)\s+have\s+suicidal\s+thoughts\b",
    r"\bi\s+no\s+longer\s+want\s+to\s+(?:kill|hurt|harm)\s+myself\b",
)

AMBIGUOUS_PATTERNS = (
    r"\bi\s+(?:can't|cannot)\s+do\s+this\s+anymore\b",
    r"\bi\s+want\s+everything\s+to\s+stop\b",
    r"\bi\s+just\s+want\s+to\s+disappear\b",
    r"\bthere(?:\s+is|'s)\s+no\s+point\b",
    r"\bi\s+(?:can't|cannot)\s+keep\s+going\b",
)


def normalize_safety_text(text):
    text = (text or "").lower().replace("’", "'")
    text = re.sub(r"[-\u2013\u2014]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _first_match(text, patterns):
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None


def classify_safety(current_message, recent_user_messages=None):
    text = normalize_safety_text(current_message)

    explicit_signal = _first_match(text, EXPLICIT_PATTERNS)
    if explicit_signal:
        return SafetyAnalysis(EXPLICIT_HIGH_RISK, explicit_signal, "current_first_person")

    third_party_signal = THIRD_PARTY_PATTERN.search(text)
    if third_party_signal:
        return SafetyAnalysis(
            THIRD_PARTY_CONCERN,
            third_party_signal.group(0),
            "third_person",
        )

    reference_signal = _first_match(text, REFERENCE_PATTERNS)
    if reference_signal:
        return SafetyAnalysis(NORMAL, reference_signal, "reference")

    negated_signal = _first_match(text, NEGATED_PATTERNS)
    if negated_signal:
        return SafetyAnalysis(NORMAL, negated_signal, "negated")

    historical_signal = _first_match(text, HISTORICAL_MARKERS)
    if historical_signal and re.search(r"\b(?:suicide|suicidal|die|kill|hurt|harm)\b", text):
        return SafetyAnalysis(NORMAL, historical_signal, "historical")

    ambiguous_signal = _first_match(text, AMBIGUOUS_PATTERNS)
    if ambiguous_signal:
        return SafetyAnalysis(AMBIGUOUS_CONCERN, ambiguous_signal, "ambiguous")

    return SafetyAnalysis(NORMAL)


EXPLICIT_RESPONSE = (
    "I'm sorry you're going through this. If you might act soon, move away from anything "
    "you could use to hurt yourself, contact local emergency services or go to the nearest "
    "emergency department, and stay with someone you trust. Please reach out now to a trusted "
    "person and a qualified crisis or mental-health support service."
)

AMBIGUOUS_RESPONSE = (
    "What you said sounds serious, and I want to check directly: are you thinking about hurting "
    "yourself or ending your life right now? If you might be in immediate danger, contact local "
    "emergency services or go to the nearest emergency department, and stay with someone you trust."
)

THIRD_PARTY_RESPONSE = (
    "That sounds serious. If it is safe, stay connected with them and involve another trusted "
    "person or qualified crisis support. If they may be in immediate danger, contact local "
    "emergency services or help them get to the nearest emergency department."
)


def deterministic_safety_response(analysis):
    if analysis.category == EXPLICIT_HIGH_RISK:
        return EXPLICIT_RESPONSE
    if analysis.category == AMBIGUOUS_CONCERN:
        return AMBIGUOUS_RESPONSE
    if analysis.category == THIRD_PARTY_CONCERN:
        return THIRD_PARTY_RESPONSE
    return None