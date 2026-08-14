"""EmbeddingService — batched text embedding with caching and fallback.

Wraps an EmbedProvider to add Redis caching, sub-batching, and model
fallback. Guarantees O(N / EMBED_BATCH_SIZE) calls for N texts.
"""

from __future__ import annotations

import logging

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from memex.engine.core import config
from memex.engine.llm.base import EmbedProvider

logger = logging.getLogger("embedding-service")


class EmbeddingService:
    """Batched embedding with caching and fallback.

    Contract::

        svc = EmbeddingService(embed_provider)
        vecs = svc.embed(["hello", "world"])  # at most 1 provider call

    Guarantees at most ``ceil(len(texts) / EMBED_BATCH_SIZE)`` calls.
    Results are returned in the same order as input texts.
    """

    def __init__(self, embed_provider: EmbedProvider) -> None:
        self._provider = embed_provider

    # ── Public API ────────────────────────────────────────────────────────

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Embed *texts* with caching and model fallback.

        Args:
            texts: List of text strings to embed.
            model: Model name override (default: config.EMBED_MODEL).

        Returns:
            List of embedding vectors, one per input text, in input order.
        """
        from memex.engine.utils.cache import get_cached_embedding

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
        from memex.engine.utils.cache import cache_embedding
        from memex.engine.utils.gpu_lock import gpu_lock

        texts_to_embed = [t for _, t in batch]

        try:
            gpu_lock.acquire("embed")
            try:
                vectors = self._post_batch(texts_to_embed, model)
            finally:
                gpu_lock.release("embed")
        except Exception as exc:
            fallback = config.EMBED_MODEL_FALLBACK
            if fallback and fallback != model:
                logger.warning(
                    "Embedding with %s failed (%s), falling back to %s",
                    model,
                    exc,
                    fallback,
                )
                gpu_lock.acquire("embed")
                try:
                    vectors = self._post_batch(texts_to_embed, fallback)
                finally:
                    gpu_lock.release("embed")
                model = fallback
            else:
                raise

        for (idx, text), vec in zip(batch, vectors, strict=True):
            cache_embedding(text, vec, model=model)
            cached_map[idx] = vec

    @retry(
        retry=retry_if_exception_type((Exception,)),
        stop=stop_after_attempt(config.HTTP_MAX_RETRIES),
        wait=wait_exponential(multiplier=config.HTTP_RETRY_BACKOFF, max=10),
        reraise=True,
    )
    def _post_batch(self, texts: list[str], model: str) -> list[list[float]]:
        """Delegate to the configured EmbedProvider and validate dimensions."""
        vectors = self._provider.embed(texts, model=model)

        # Validate dimensions match expected DENSE_DIM
        expected_dim = config.DENSE_DIM
        for _i, vec in enumerate(vectors):
            if len(vec) != expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: model {model} returned {len(vec)}d, "
                    f"expected {expected_dim}d (DENSE_DIM). Check model config."
                )

        return vectors
