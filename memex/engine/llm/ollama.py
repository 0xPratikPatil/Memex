"""Ollama LLM and embedding providers via HTTP API.

Implements both LLMProvider (async chat) and EmbedProvider (sync embed)
backed by a local or remote Ollama instance.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

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
        self._client_local = threading.local()

    def _get_client(self) -> httpx.AsyncClient:
        # Per-thread client, paired with the per-thread event loop from
        # base.py's chat_sync. A shared client across threads is fatal:
        # thread B's _get_client() would replace the client thread A is
        # mid-request on, orphaning A's await forever.
        client = getattr(self._client_local, "client", None)
        if client is None or client.is_closed:
            from memex.engine.core import config

            client = httpx.AsyncClient(
                # Phase-split timeouts: a single read must not burn the whole
                # total budget (a transient stall fails fast and retries).
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=config.LLM_READ_TIMEOUT,
                    write=30.0,
                    pool=30.0,
                ),
                limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
            )
            self._client_local.client = client
        return client

    async def chat(self, prompt: str, *, model: str | None = None, num_predict: int | None = None) -> str:
        """Post to ``/api/chat`` and return assistant content.

        Handles responses where the model outputs a ``thinking`` field
        instead of ``content`` (e.g. for reasoning models).
        """
        client = self._get_client()
        options: dict[str, Any] = {"temperature": 0}
        if num_predict is not None:
            options["num_predict"] = num_predict
        resp = await client.post(
            f"{self._base_url}/api/chat",
            json={
                "model": model or self._model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": options,
            },
        )
        resp.raise_for_status()
        msg = resp.json()["message"]
        content = msg.get("content", "") or msg.get("thinking", "")
        return content

    async def close(self) -> None:
        clients = [getattr(self._client_local, "client", None)]
        for client in clients:
            if client is not None and not client.is_closed:
                await client.aclose()
        self._client_local.client = None


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
