import os
import streamlit as st
from dotenv import load_dotenv


load_dotenv()


def get_secret(key: str, default: str | None = None) -> str | None:
    """
    Reads secrets from Streamlit Cloud first, then local .env.
    """
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)


GROQ_API_KEY = get_secret("GROQ_API_KEY")
TAVILY_API_KEY = get_secret("TAVILY_API_KEY")


def validate_required_keys() -> list[str]:
    missing = []

    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")

    if not TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY")

    return missing