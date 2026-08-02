"""OpenAI LLM and embedding providers."""

from __future__ import annotations

import asyncio
import logging

import httpx

from memex.engine.llm.base import EmbedProvider, LLMProvider

logger = logging.getLogger(__name__)


class _OpenAIBase:
    """Shared httpx client management for OpenAI-compatible APIs."""

    _client: httpx.AsyncClient | None = None
    _client_loop: asyncio.AbstractEventLoop | None = None

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    def _get_client(self) -> httpx.AsyncClient:
        # chat_sync() runs asyncio.run() in a worker thread, which creates a
        # client bound to a temporary loop. Recreate the client for the
        # current loop instead of reusing the dead one.
        loop = asyncio.get_running_loop()
        if self._client is None or self._client.is_closed or self._client_loop is not loop:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout, connect=10.0),
                limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            self._client_loop = loop
        return self._client


class OpenAILLM(LLMProvider):
    """OpenAI chat completions via ``/chat/completions``.

    Config keys: ``llm.api_key``, ``llm.model``.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str = "https://api.openai.com/v1") -> None:
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


class OpenAIEmbedder(EmbedProvider):
    """OpenAI embeddings via ``/embeddings``.

    Config keys: ``embedding.api_key``, ``embedding.model``.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self._http = _OpenAIBase(base_url=base_url, api_key=api_key, model=model)
        self._sync_client: httpx.Client | None = None

    def _get_sync_client(self) -> httpx.Client:
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(
                base_url=self._http._base_url,
                timeout=httpx.Timeout(self._http._timeout, connect=10.0),
                headers={"Authorization": f"Bearer {self._http._api_key}"},
            )
        return self._sync_client

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        client = self._get_sync_client()
        resp = client.post(
            "/embeddings",
            json={"model": model or self._http._model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]
