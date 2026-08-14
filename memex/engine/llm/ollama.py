"""Ollama LLM and embedding providers via HTTP API.

Implements both LLMProvider (async chat) and EmbedProvider (sync embed)
backed by a local or remote Ollama instance.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from memex.engine.llm.base import EmbedProvider, LLMProvider

logger = logging.getLogger(__name__)


class OllamaLLM(LLMProvider):
    """Ollama chat via ``POST /api/chat``.

    Config keys: ``llm.base_url``, ``llm.model``.
    """

    def __init__(self, base_url: str, model: str = "qwen2.5:1.5b", timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None

    def _get_client(self) -> httpx.AsyncClient:
        # Track the event loop that created the client. chat_sync() runs
        # asyncio.run() in a worker thread, so the client is bound to a
        # temporary loop that closes afterwards. Recreate the client for
        # the current loop instead of reusing the dead one.
        loop = asyncio.get_running_loop()
        if self._client is None or self._client.is_closed or self._client_loop is not loop:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout, connect=10.0),
                limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
            )
            self._client_loop = loop
        return self._client

    async def chat(self, prompt: str, *, model: str | None = None) -> str:
        """Post to ``/api/chat`` and return assistant content.

        Handles responses where the model outputs a ``thinking`` field
        instead of ``content`` (e.g. for reasoning models).
        """
        client = self._get_client()
        resp = await client.post(
            f"{self._base_url}/api/chat",
            json={
                "model": model or self._model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0},
            },
        )
        resp.raise_for_status()
        msg = resp.json()["message"]
        content = msg.get("content", "") or msg.get("thinking", "")
        return content

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class OllamaEmbedder(EmbedProvider):
    """Ollama embeddings via ``POST /api/embed`` (batch endpoint).

    Config keys: ``embedding.base_url``, ``embedding.model``.
    """

    def __init__(self, base_url: str, model: str = "qwen3-embedding:0.6b", timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        if "/api/embed" in self._base_url:
            self._base_url = self._base_url.rsplit("/api/embed", 1)[0]
        if "/api/embeddings" in self._base_url:
            self._base_url = self._base_url.rsplit("/api/embeddings", 1)[0]
        self._model = model
        self._timeout = timeout
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                timeout=httpx.Timeout(self._timeout, connect=10.0),
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            )
        return self._client

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        """POST to ``/api/embed`` with batched input.

        Uses Ollama's native batch endpoint for efficiency.
        """
        client = self._get_client()
        resp = client.post(
            f"{self._base_url}/api/embed",
            json={"model": model or self._model, "input": texts},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            self._client.close()
            self._client = None


