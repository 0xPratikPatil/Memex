"""Query expansion: HyDE, Multi-Query, Query Rewriting.

Runs *before* hybrid search to improve recall for complex or ambiguous queries.
All techniques are optional and controlled by feature flags in ``config``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

from rag import config

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

    Uses the Ollama HTTP API for both chat (generation) and embeddings.
    Each technique is gated behind its own ``config.ENABLE_*`` flag.
    """

    def __init__(self, ollama_client: httpx.Client) -> None:
        self._ollama = ollama_client

    # ── Public API ────────────────────────────────────────────────────────

    def expand(self, query: str) -> ExpandedQuery:
        """Run enabled expansion techniques and return results.

        Techniques execute sequentially: rewrite → HyDE → multi-query.
        Each step is independent; failures are logged and skipped.
        """
        result = ExpandedQuery(original=query)

        if config.ENABLE_QUERY_REWRITE:
            try:
                result.rewritten = self._rewrite(query)
                logger.debug("Rewritten query: %s", result.rewritten)
            except Exception:
                logger.warning("Query rewrite failed, using original", exc_info=True)

        effective_query = result.rewritten or query

        if config.ENABLE_HYDE:
            try:
                result.hyde_vector = self._hyde_embed(effective_query)
                logger.debug("HyDE vector computed (%d dims)", len(result.hyde_vector))
            except Exception:
                logger.warning("HyDE failed, skipping", exc_info=True)

        if config.ENABLE_MULTI_QUERY:
            try:
                result.paraphrases = self._multi_query(effective_query)
                logger.debug("Generated %d paraphrases", len(result.paraphrases))
            except Exception:
                logger.warning("Multi-query failed, skipping", exc_info=True)

        return result

    # ── Private helpers ───────────────────────────────────────────────────

    def _rewrite(self, query: str) -> str:
        """Rewrite query using LLM to expand abbreviations and fix phrasing."""
        prompt = (
            "Rewrite this search query to be more specific and clear. "
            "Keep it under 50 words. Only output the rewritten query.\n\n"
            f"Query: {query}"
        )
        return self._chat(prompt)

    def _hyde_embed(self, query: str) -> list[float]:
        """Generate a hypothetical answer document and return its embedding."""
        prompt = (
            "Write a short paragraph that would be a perfect answer to this query. "
            "Be factual and specific. 3-5 sentences.\n\n"
            f"Query: {query}"
        )
        hypothetical = self._chat(prompt)
        return self._embed(hypothetical)

    def _multi_query(self, query: str) -> list[str]:
        """Generate N paraphrases of the query."""
        count = config.MULTI_QUERY_COUNT
        prompt = (
            f"Generate {count} diverse paraphrases of this search query. "
            "Each on a new line. Only the paraphrases, no numbering.\n\n"
            f"Query: {query}"
        )
        response = self._chat(prompt)
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
        return lines[:count]

    def _chat(self, prompt: str) -> str:
        """Call Ollama chat API and return the assistant message content."""
        chat_url = config.OLLAMA_EMBED_URL.replace("/api/embeddings", "/api/chat")
        model = config.HYDE_MODEL or config.MULTI_QUERY_MODEL or config.QUERY_REWRITE_MODEL or config.EMBED_MODEL
        resp = self._ollama.post(
            chat_url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _embed(self, text: str) -> list[float]:
        """Embed text via Ollama."""
        resp = self._ollama.post(
            config.OLLAMA_EMBED_URL,
            json={"model": config.EMBED_MODEL, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
