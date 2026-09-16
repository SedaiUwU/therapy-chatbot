import logging

from llm_provider import build_groq_config, create_groq_client

logger = logging.getLogger(__name__)

FALLBACK_RESPONSE = (
    "I hear you. That sounds like a lot to carry right now. "
    "We can just stay with this moment together."
)


def create_response_provider():
    config = build_groq_config()
    client = None

    def respond(prompt):
        nonlocal client
        if not config["api_key"]:
            logger.warning("Groq is not configured; using the fallback response.")
            return FALLBACK_RESPONSE

        if client is None:
            try:
                client = create_groq_client(config["api_key"], config["base_url"])
            except Exception:
                logger.exception("Groq client initialization failed; using fallback response.")
                return FALLBACK_RESPONSE

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
                    content = getattr(response.choices[0].message, "content", "") or ""
                    if isinstance(content, list):
                        content = "".join(
                            part.get("text", "")
                            for part in content
                            if isinstance(part, dict)
                        )
                content = " ".join(content.split())
                if len(content) >= 10:
                    return content
            except Exception as exc:
                logger.warning(
                    "Groq generation failed on attempt %s: %s",
                    attempt + 1,
                    type(exc).__name__,
                )

        return FALLBACK_RESPONSE

    return respond
