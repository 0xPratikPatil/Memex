"""Query expansion: HyDE, Multi-Query, Query Rewriting.

Runs *before* hybrid search to improve recall for complex or ambiguous queries.
All techniques are optional and controlled by feature flags in ``config``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from memex.engine.core import config
from memex.engine.llm.base import EmbedProvider, LLMProvider

logger = logging.getLogger("query-expansion")


@dataclass
class ExpandedQuery:
    """Result of query expansion.

    Attributes:
        original: The raw user query.
        rewritten: Query after LLM rewriting (``None`` if rewrite disabled).
        hyde_vector: Embedding of a hypothetical answer document.
        paraphrases: List of paraphrased query strings from multi-query.
    """

    original: str
    rewritten: str | None = None
    hyde_vector: list[float] | None = None
    paraphrases: list[str] | None = field(default=None)


class QueryExpander:
    """Orchestrates query expansion techniques.

    Uses the configured LLM provider for chat (generation) and embedding
    provider for HyDE embeddings. Each technique is gated behind its own
    ``config.ENABLE_*`` flag.
    """

    def __init__(self, llm_provider: LLMProvider, embed_provider: EmbedProvider) -> None:
        self._llm = llm_provider
        self._embed_provider = embed_provider
        self._embedding_svc: Any = None

    # ── Public API ────────────────────────────────────────────────────────

    def expand(self, query: str) -> ExpandedQuery:
        """Run enabled expansion techniques and return results.

        Techniques execute sequentially: rewrite → HyDE → multi-query.
        Each step is independent; failures are logged and skipped.
        Results are cached in Redis using ``CACHE_TTL_EXPANSION``.
        """
        from memex.engine.utils.cache import get_cached, set_cached

        cache_key = f"expand:{query}"
        if config.ENABLE_CACHE:
            cached = get_cached("expansion", cache_key)
            if cached is not None:
                logger.debug("Query expansion cache hit for: %s", query[:50])
                return ExpandedQuery(
                    original=cached["original"],
                    rewritten=cached.get("rewritten"),
                    hyde_vector=cached.get("hyde_vector"),
                    paraphrases=cached.get("paraphrases"),
                )

        result = ExpandedQuery(original=query)

        if config.ENABLE_QUERY_REWRITE:
            try:
                result.rewritten = self._rewrite(query)
                logger.debug("Rewritten query: %s", result.rewritten)
            except Exception:
                logger.warning("Query rewrite failed, using original", exc_info=True)

        effective_query = result.rewritten or query

        hyde_task = None
        multi_task = None

        if config.ENABLE_HYDE:
            hyde_task = self._hyde_embed
        if config.ENABLE_MULTI_QUERY:
            multi_task = self._multi_query

        if hyde_task and multi_task:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                hyde_future = pool.submit(hyde_task, effective_query)
                multi_future = pool.submit(multi_task, effective_query)
                try:
                    result.hyde_vector = hyde_future.result()
                    logger.debug("HyDE vector computed (%d dims)", len(result.hyde_vector))
                except Exception:
                    logger.warning("HyDE failed, skipping", exc_info=True)
                try:
                    result.paraphrases = multi_future.result()
                    logger.debug("Generated %d paraphrases", len(result.paraphrases))
                except Exception:
                    logger.warning("Multi-query failed, skipping", exc_info=True)
        else:
            if hyde_task:
                try:
                    result.hyde_vector = hyde_task(effective_query)
                    logger.debug("HyDE vector computed (%d dims)", len(result.hyde_vector))
                except Exception:
                    logger.warning("HyDE failed, skipping", exc_info=True)
            if multi_task:
                try:
                    result.paraphrases = multi_task(effective_query)
                    logger.debug("Generated %d paraphrases", len(result.paraphrases))
                except Exception:
                    logger.warning("Multi-query failed, skipping", exc_info=True)

        if config.ENABLE_CACHE:
            try:
                set_cached(
                    "expansion",
                    cache_key,
                    {
                        "original": result.original,
                        "rewritten": result.rewritten,
                        "hyde_vector": result.hyde_vector,
                        "paraphrases": result.paraphrases,
                    },
                    ttl=config.CACHE_TTL_EXPANSION,
                )
            except Exception:
                logger.debug("Failed to cache query expansion", exc_info=True)

        return result

    # ── Private helpers ───────────────────────────────────────────────────

    def _rewrite(self, query: str) -> str:
        """Rewrite query using LLM to expand abbreviations and fix phrasing."""
        prompt = (
            "Rewrite this search query to be more specific and clear. "
            "Keep it under 50 words. Only output the rewritten query.\n\n"
            f"Query: {query}"
        )
        model = config.QUERY_REWRITE_MODEL or config.CHAT_MODEL
        return self._chat(prompt, model_override=model)

    def _hyde_embed(self, query: str) -> list[float]:
        """Generate a hypothetical answer document and return its embedding."""
        prompt = (
            "Write a short paragraph that would be a perfect answer to this query. "
            "Be factual and specific. 3-5 sentences.\n\n"
            f"Query: {query}"
        )
        model = config.HYDE_MODEL or config.CHAT_MODEL
        hypothetical = self._chat(prompt, model_override=model)
        return self._embed(hypothetical)

    def _multi_query(self, query: str) -> list[str]:
        """Generate N paraphrases of the query."""
        count = config.MULTI_QUERY_COUNT
        prompt = (
            f"Generate {count} diverse paraphrases of this search query. "
            "Each on a new line. Only the paraphrases, no numbering.\n\n"
            f"Query: {query}"
        )
        model = config.MULTI_QUERY_MODEL or config.CHAT_MODEL
        response = self._chat(prompt, model_override=model)
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
        return lines[:count]

    def _chat(self, prompt: str, num_predict: int = 150, model_override: str = "") -> str:
        """Call LLM provider synchronously."""
        model = model_override or config.CHAT_MODEL
        return self._llm.chat_sync(prompt, model=model)

    def _embed(self, text: str, model: str | None = None) -> list[float]:
        """Embed text via EmbeddingService (singleton, batched transport, caching)."""
        if self._embedding_svc is None:
            from memex.engine.ingestion.embedding import EmbeddingService

            self._embedding_svc = EmbeddingService(self._embed_provider)
        return self._embedding_svc.embed([text], model=model)[0]
