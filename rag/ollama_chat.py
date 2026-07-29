"""Shared Ollama chat helper used by query expansion, contextual retrieval,
and metadata extraction services.
"""

from __future__ import annotations

import logging

from rag import config

logger = logging.getLogger(__name__)

# Default Ollama client (lazy-initialized by callers)


def ollama_chat(
    prompt: str,
    *,
    model: str | None = None,
    num_predict: int = 200,
    ollama_client=None,
) -> str:
    """Call Ollama chat API and return the assistant message content.

    Args:
        prompt: The user message to send.
        model: Model name. Falls back to config.CHAT_MODEL.
        num_predict: Max tokens to generate.
        ollama_client: httpx.Client connected to Ollama.

    Returns:
        The assistant message content.

    Raises:
        RuntimeError: If ollama_client is None.
        httpx.HTTPStatusError: If the API call fails.
    """
    if ollama_client is None:
        raise RuntimeError("Ollama client not available")

    chat_url = config.OLLAMA_EMBED_URL.replace("/api/embed", "/api/chat")
    model = model or config.CHAT_MODEL

    resp = ollama_client.post(
        chat_url,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": num_predict, "temperature": 0},
        },
    )
    resp.raise_for_status()
    msg = resp.json()["message"]
    content = msg.get("content", "")
    if not content:
        content = msg.get("thinking", "")
    return content
