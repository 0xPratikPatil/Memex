"""Abstract base classes for LLM and embedding providers."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Async chat completion interface.

    Implementations: OllamaLLM, OpenAILLM, AnthropicLLM, GroqLLM, GoogleLLM.
    """

    @abstractmethod
    async def chat(self, prompt: str, *, model: str | None = None) -> str:
        """Generate a chat completion for a single-turn prompt."""

    def chat_sync(self, prompt: str, *, model: str | None = None) -> str:
        """Synchronous wrapper for chat()."""
        try:
            _ = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.chat(prompt, model=model))
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, self.chat(prompt, model=model)).result()


class EmbedProvider(ABC):
    """Synchronous embedding interface.

    Returns one vector per input text. Implementations: OllamaEmbedder,
    OpenAIEmbedder, HuggingFaceEmbedder, FastEmbedEmbedder.
    """

    @abstractmethod
    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        """Generate dense embeddings for a batch of texts."""
