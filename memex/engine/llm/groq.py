"""Groq LLM provider — OpenAI-compatible endpoint.

Config keys: ``llm.api_key``, ``llm.model``.
"""

from __future__ import annotations

from memex.engine.llm.base import LLMProvider
from memex.engine.llm.openai import _OpenAIBase


class GroqLLM(LLMProvider):
    """Groq chat via OpenAI-compatible API at ``https://api.groq.com/openai/v1``.

    Config keys: ``llm.api_key``, ``llm.model``.
    """

    def __init__(self, api_key: str, model: str = "llama3-70b-8192") -> None:
        self._http = _OpenAIBase(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            model=model,
        )

    async def chat(self, prompt: str, *, model: str | None = None) -> str:
        client = self._http._get_client()
        resp = await client.post(
            "/chat/completions",
            json={
                "model": model or self._http._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
