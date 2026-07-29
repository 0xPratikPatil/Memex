"""EmbeddingService — batched text embedding via Ollama with caching and fallback.

Guarantees O(N / EMBED_BATCH_SIZE) HTTP calls for N texts by using
Ollama's native ``/api/embed`` batch endpoint.

On model failure, falls back to EMBED_MODEL_FALLBACK only if it differs
from the primary model (otherwise retrying the same model is waste).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rag import config

logger = logging.getLogger("embedding-service")


class EmbeddingService:
    """Batched embedding via Ollama with Redis caching and model fallback.

    Contract::

        svc = EmbeddingService(ollama_client)
        vecs = svc.embed(["hello", "world"])  # 1 HTTP call for both

    Guarantees at most ``ceil(len(texts) / EMBED_BATCH_SIZE)`` HTTP calls.
    Results are returned in the same order as input texts.
    """

    def __init__(self, ollama_client: httpx.Client) -> None:
        self._client = ollama_client
        self._embed_url = self._resolve_embed_url()

    # ── Public API ────────────────────────────────────────────────────────

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Embed *texts* via Ollama with caching and model fallback.

        Args:
            texts: List of text strings to embed.
            model: Model name override (default: config.EMBED_MODEL).

        Returns:
            List of embedding vectors, one per input text, in input order.
        """
        from rag.services.cache import get_cached_embedding

        if model is None:
            model = config.EMBED_MODEL

        if not texts:
            return []

        # 1. Check cache per-text, collect uncached
        uncached: list[tuple[int, str]] = []
        cached_map: dict[int, list[float]] = {}

        for idx, text in enumerate(texts):
            cached = get_cached_embedding(text, model=model)
            if cached is not None:
                cached_map[idx] = cached
            else:
                uncached.append((idx, text))

        if not uncached:
            return [cached_map[i] for i in range(len(texts))]

        # 2. Split uncached into sub-batches
        for batch_start in range(0, len(uncached), config.EMBED_BATCH_SIZE):
            batch_end = min(batch_start + config.EMBED_BATCH_SIZE, len(uncached))
            sub_batch = uncached[batch_start:batch_end]
            self._embed_batch(sub_batch, cached_map, model)

        # 3. Reassemble in original order
        return [cached_map[i] for i in range(len(texts))]

    # ── Private helpers ───────────────────────────────────────────────────

    def _embed_batch(
        self,
        batch: list[tuple[int, str]],
        cached_map: dict[int, list[float]],
        model: str,
    ) -> None:
        """Embed a sub-batch with fallback on failure."""
        from rag.services.cache import cache_embedding

        texts_to_embed = [t for _, t in batch]

        try:
            vectors = self._post_batch(texts_to_embed, model)
        except Exception as exc:
            fallback = config.EMBED_MODEL_FALLBACK
            if fallback and fallback != model:
                logger.warning(
                    "Embedding with %s failed (%s), falling back to %s",
                    model, exc, fallback,
                )
                vectors = self._post_batch(texts_to_embed, fallback)
                model = fallback  # cache under fallback model name
            else:
                raise

        for (idx, text), vec in zip(batch, vectors, strict=True):
            cache_embedding(text, vec, model=model)
            cached_map[idx] = vec

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        stop=stop_after_attempt(config.HTTP_MAX_RETRIES),
        wait=wait_exponential(multiplier=config.HTTP_RETRY_BACKOFF, max=10),
        reraise=True,
    )
    def _post_batch(self, texts: list[str], model: str) -> list[list[float]]:
        """POST to Ollama /api/embed with batched input and dimension validation."""
        resp = self._client.post(
            self._embed_url,
            json={"model": model, "input": texts},
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        vectors = data["embeddings"]

        # Validate dimensions match expected DENSE_DIM
        expected_dim = config.DENSE_DIM
        for _i, vec in enumerate(vectors):
            if len(vec) != expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: model {model} returned {len(vec)}d, "
                    f"expected {expected_dim}d (DENSE_DIM). Check model config."
                )

        return vectors

    @staticmethod
    def _resolve_embed_url() -> str:
        """Ensure the embed URL points to the batched /api/embed endpoint.

        If the user set a custom URL pointing to /api/embeddings (legacy),
        we warn but leave it untouched — the user chose this deliberately.
        If the URL is the default, we transparently switch to /api/embed.
        """
        url = config.OLLAMA_EMBED_URL
        if "/api/embeddings" in url:
            from rag import config as cfg

            default_embeddings = f"http://localhost:{cfg.OLLAMA_PORT}/api/embeddings"
            if url.rstrip("/") == default_embeddings.rstrip("/"):
                new_url = url.replace("/api/embeddings", "/api/embed")
                logger.info("Switching default OLLAMA_EMBED_URL to batched endpoint: %s", new_url)
                return new_url
            else:
                logger.warning(
                    "Custom OLLAMA_EMBED_URL uses legacy /api/embeddings endpoint — "
                    "embedding will be slow. Switch to /api/embed for native batching."
                )
        return url
