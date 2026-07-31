"""Anthropic LLM provider using the ``anthropic`` SDK.

Requires ``pip install anthropic``.
"""

from __future__ import annotations

import logging

from memex.engine.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class AnthropicLLM(LLMProvider):
    """Anthropic chat via the ``anthropic`` Python SDK.

    Config keys: ``llm.api_key``, ``llm.model``.
    """

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-latest") -> None:
        try:
            import anthropic as _anthropic
        except ImportError as err:
            raise ImportError(
                "Anthropic provider requires the 'anthropic' package.\n"
                "Install it with: pip install anthropic"
            ) from err
        self._client = _anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    async def chat(self, prompt: str, *, model: str | None = None) -> str:
        message = await self._client.messages.create(
            model=model or self._model,
            max_tokens=1024,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
