"""OpenRouter LLM provider — OpenAI-compatible endpoint.

Uses the same interface as OpenAILLM but defaults to OpenRouter's base URL.
"""

from __future__ import annotations

from memex.engine.llm.base import LLMProvider
from memex.engine.llm.openai import _OpenAIBase


class OpenRouterLLM(LLMProvider):
    """OpenRouter chat via OpenAI-compatible API.

    Config keys: ``llm.api_key``, ``llm.model``.
    Base URL defaults to ``https://openrouter.ai/api/v1``.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-4o",
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        self._http = _OpenAIBase(base_url=base_url, api_key=api_key, model=model)

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
