"""OpenAI-compatible client."""

import os

from graph_memory.config import settings


def make_client():
    """Client for `settings.openai_base_url`.

    OPENAI_API_KEY wins if set; Tokens are short-lived, so the client is
    built when a command starts rather than kept for days.
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    return OpenAI(base_url=settings.openai_base_url, api_key=api_key)
