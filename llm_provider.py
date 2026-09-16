import os

from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def build_groq_config(api_key=None):
    config = {
        "provider": "groq",
        "model": GROQ_MODEL,
        "base_url": GROQ_BASE_URL,
        "api_key": api_key if api_key is not None else os.getenv("GROQ_API_KEY"),
        "key_name": "GROQ_API_KEY",
    }
    if not config["api_key"]:
        config["missing_key"] = "GROQ_API_KEY"
    return config


def create_groq_client(api_key, base_url=GROQ_BASE_URL):
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)
