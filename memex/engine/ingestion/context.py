"""Contextual retrieval: generate context prefixes for chunks.

Based on Anthropic's contextual retrieval technique. Each chunk gets a short
contextual prefix that situates it within the document, improving embedding
quality and retrieval accuracy.

Strategies:
  - header: Use section header as context (fastest, no LLM calls)
  - summary: Use LLM to generate context from document summary
  - surrounding: Use LLM to generate context from adjacent chunks
"""

from __future__ import annotations

import logging
import re
from typing import Any

from memex.engine.core import config
from memex.engine.llm.base import LLMProvider

logger = logging.getLogger("contextual-retrieval")

# Mirrors _LLM_MAX_ATTEMPTS in llm/base.py — used for concise warning counts.
_LLM_MAX_ATTEMPTS = 3


class ContextGenerator:
    """Generates contextual prefixes for document chunks."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm = llm_provider

    def generate_document_summary(self, document_text: str) -> str:
        """Generate a brief summary of the entire document for context generation."""
        prompt = (
            "Summarize this document in 2-3 sentences. Focus on what the document is about, "
            "who wrote it, and its main topics.\n\n"
            f"Document:\n{document_text[:4000]}"
        )
        return self._chat(prompt)

    def generate_context(
        self,
        chunk: str,
        document_summary: str = "",
        section_header: str = "",
        prev_chunk: str = "",
        next_chunk: str = "",
    ) -> str:
        """Generate a contextual prefix for a chunk based on configured strategy."""
        strategy = config.CONTEXT_STRATEGY.lower()

        if strategy == "header":
            return self._context_from_header(section_header)
        elif strategy == "summary":
            return self._context_from_summary(chunk, document_summary)
        elif strategy == "surrounding":
            return self._context_from_surrounding(chunk, prev_chunk, next_chunk)
        else:
            return self._context_from_header(section_header)

    def _context_from_header(self, header: str) -> str:
        """Fastest: derive context from section header only."""
        if not header:
            return ""
        return f"[Context: {header}]"

    def _context_from_summary(self, chunk: str, summary: str) -> str:
        """LLM-based: use document summary to generate context."""
        if not summary:
            return ""
        prompt = (
            "Given this document summary and a text chunk, write a short contextual "
            "prefix (under 30 words) that situates the chunk within the document. "
            "Do not repeat the chunk content. Only output the prefix.\n\n"
            f"Document summary: {summary}\n\n"
            f"Chunk: {chunk[:500]}"
        )
        response = self._chat(prompt)
        return f"[Context: {response.strip()}]"

    def _context_from_surrounding(self, chunk: str, prev: str, next: str) -> str:
        """LLM-based: use surrounding chunks for context."""
        context_parts = []
        if prev:
            context_parts.append(f"Previous content: {prev[:200]}")
        if next:
            context_parts.append(f"Following content: {next[:200]}")

        if not context_parts:
            return ""

        prompt = (
            "Given the surrounding content of a text chunk, write a short contextual "
            "prefix (under 30 words) that situates the chunk. Only output the prefix.\n\n"
            + "\n".join(context_parts)
            + f"\n\nChunk: {chunk[:300]}"
        )
        response = self._chat(prompt)
        return f"[Context: {response.strip()}]"

    def _apply_chunk_context(
        self,
        chunk: dict[str, Any],
        context: str,
    ) -> dict[str, Any]:
        enriched_chunk = {**chunk}
        enriched_chunk["context_prefix"] = context
        if context:
            enriched_chunk["content"] = f"{context} {chunk['content']}".strip()
        return enriched_chunk

    def enrich_chunks(
        self,
        chunks: list[dict[str, Any]],
        document_summary: str = "",
    ) -> list[dict[str, Any]]:
        """Add context prefixes to all chunks in a document.

        Each chunk dict is copied and augmented with:
          - ``context_prefix``: the generated context string
          - ``content``: original content prefixed with the context

        Context generation follows a resilience chain:
          1. Batch LLM call (summary or surrounding strategy)
          2. Per-chunk LLM fallback on parse gaps
          3. Section-header fallback on LLM failure
          4. Empty string (no prefix) as last resort

        For LLM-based strategies, chunks are processed in batches of
        ``CONTEXT_BATCH_SIZE`` to reduce LLM round-trips. Summary batches
        are run concurrently when there are multiple batches.
        """
        if not chunks:
            return chunks

        strategy = config.CONTEXT_STRATEGY.lower()

        if strategy == "header":
            return [
                self._apply_chunk_context(c, self._context_from_header(c.get("section_header", ""))) for c in chunks
            ]

        # LLM-based strategies: batch chunks
        batch_size = config.CONTEXT_BATCH_SIZE

        all_batches: list[tuple[list[dict[str, Any]], int]] = []
        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start : batch_start + batch_size]
            all_batches.append((batch, batch_start))

        # Cap sequential LLM batches — pathological docs must not trigger
        # dozens of LLM calls. Beyond the cap, use header fallback.
        llm_batches = all_batches[: config.CONTEXT_MAX_BATCHES]
        if len(all_batches) > config.CONTEXT_MAX_BATCHES:
            logger.warning(
                "Context generation capped at %d LLM batches (%d total) — "
                "remaining chunks use header fallback",
                config.CONTEXT_MAX_BATCHES,
                len(all_batches),
            )

        # For summary strategy, process batches sequentially.
        # Ollama processes requests sequentially anyway, and using
        # ThreadPoolExecutor causes "Event loop is closed" errors
        # when asyncio.run() creates/destroys event loops in threads.
        if strategy == "summary":
            batch_results: dict[int, list[str]] = {}
            for batch, batch_start in llm_batches:
                batch_results[batch_start] = self._batch_context_from_summary(batch, document_summary)

            enriched: list[dict[str, Any]] = []
            for batch, batch_start in all_batches:
                contexts = batch_results.get(batch_start)
                if contexts is None:
                    # Beyond the LLM batch cap — header fallback only.
                    for chunk in batch:
                        enriched.append(
                            self._apply_chunk_context(
                                chunk, self._context_from_header(chunk.get("section_header", ""))
                            )
                        )
                    continue
                for chunk, context in zip(batch, contexts, strict=True):
                    if not context:
                        context = self._fallback_context(chunk, document_summary)
                    enriched.append(self._apply_chunk_context(chunk, context))
            return enriched

        # Surrounding strategy
        enriched = []
        for batch, batch_start in all_batches:
            if batch_start not in [b[1] for b in llm_batches]:
                for chunk in batch:
                    enriched.append(
                        self._apply_chunk_context(
                            chunk, self._context_from_header(chunk.get("section_header", ""))
                        )
                    )
                continue
            contexts = self._batch_context_from_surrounding(chunks, batch_start, batch_size)
            for chunk, context in zip(batch, contexts, strict=True):
                if not context:
                    context = self._context_from_header(chunk.get("section_header", ""))
                enriched.append(self._apply_chunk_context(chunk, context))

        return enriched

    def _fallback_context(self, chunk: dict[str, Any], document_summary: str) -> str:
        """Resilience chain: per-chunk summary → header → empty."""
        if document_summary:
            try:
                ctx = self._context_from_summary(chunk["content"], document_summary)
                if ctx:
                    return ctx
            except Exception:
                logger.debug("Per-chunk context fallback failed for chunk: %s...", chunk["content"][:60])
        return self._context_from_header(chunk.get("section_header", ""))

    def _batch_context_from_summary(
        self,
        batch: list[dict[str, Any]],
        summary: str,
    ) -> list[str]:
        """Generate context for a batch of chunks using document summary.

        Returns a list of context strings, one per chunk. Empty strings
        indicate contexts that could not be generated; the caller's
        resilience chain handles fallback.
        """
        if not summary:
            return [""] * len(batch)

        chunks_text = "\n\n".join(f"[Chunk {i}]: {c['content'][:500]}" for i, c in enumerate(batch))
        prompt = (
            "Given this document summary and a batch of text chunks, write a short "
            "contextual prefix (under 30 words) for each chunk that situates it "
            "within the document. Do not repeat chunk content.\n\n"
            f"Document summary: {summary}\n\n"
            f"Chunks:\n{chunks_text}\n\n"
            f"Output exactly {len(batch)} lines, one prefix per chunk, "
            "numbered like: 1. prefix text\n2. prefix text\n..."
        )
        try:
            response, attempts = self._chat_with_attempts(prompt)
        except Exception:
            logger.warning(
                "LLM batch context failed after %d attempts — returning empty contexts",
                _LLM_MAX_ATTEMPTS,
            )
            logger.debug("Batch context LLM call failure detail", exc_info=True)
            return [""] * len(batch)
        if attempts > 1:
            logger.debug("Batch context succeeded after %d attempts", attempts)

        lines = re.findall(r"(?:^|\n)\s*\d+[.)]\s*(.+)", response)
        if len(lines) < len(batch):
            logger.debug(
                "Batch context parse: got %d lines for %d chunks (model=%s)",
                len(lines),
                len(batch),
                config.CONTEXT_MODEL or config.CHAT_MODEL,
            )
        while len(lines) < len(batch):
            lines.append("")
        return [f"[Context: {p.strip()}]" if p.strip() else "" for p in lines[: len(batch)]]

    def _batch_context_from_surrounding(
        self,
        all_chunks: list[dict[str, Any]],
        batch_start: int,
        batch_size: int,
    ) -> list[str]:
        """Generate context for a batch using surrounding chunks.

        Uses a single LLM call with all chunk contexts to avoid N+1 pattern.
        Falls back to per-chunk calls on parse failure.
        """
        batch = all_chunks[batch_start : batch_start + batch_size]
        if len(batch) == 1:
            return [
                self._context_from_surrounding(
                    batch[0]["content"],
                    all_chunks[batch_start - 1]["content"] if batch_start > 0 else "",
                    all_chunks[batch_start + 1]["content"] if batch_start + 1 < len(all_chunks) else "",
                )
            ]

        chunks_with_context = []
        for i, chunk in enumerate(batch):
            global_idx = batch_start + i
            prev = all_chunks[global_idx - 1]["content"][:200] if global_idx > 0 else ""
            next_ = all_chunks[global_idx + 1]["content"][:200] if global_idx < len(all_chunks) - 1 else ""
            context_parts = []
            if prev:
                context_parts.append(f"Previous: {prev}")
            if next_:
                context_parts.append(f"Following: {next_}")
            context_str = " | ".join(context_parts) if context_parts else "(no surrounding context)"
            chunks_with_context.append(f"[{i}] Context: {context_str}\nChunk: {chunk['content'][:300]}")

        all_chunks_text = "\n\n".join(chunks_with_context)
        prompt = (
            f"Given the surrounding content of {len(batch)} text chunks, write a short contextual "
            "prefix (under 30 words) for each chunk that situates it. "
            "Do not repeat chunk content.\n\n"
            f"{all_chunks_text}\n\n"
            f"Output exactly {len(batch)} lines, one prefix per chunk, "
            "numbered like: 1. prefix text\n2. prefix text\n..."
        )
        try:
            response, attempts = self._chat_with_attempts(prompt, num_predict=200 * len(batch))
        except Exception:
            logger.warning(
                "LLM batch context failed after %d attempts — returning empty contexts",
                _LLM_MAX_ATTEMPTS,
            )
            logger.debug("Batch surrounding context LLM call failure detail", exc_info=True)
            return [""] * len(batch)
        if attempts > 1:
            logger.debug("Batch surrounding context succeeded after %d attempts", attempts)
        # Parse numbered lines
        lines = re.findall(r"(?:^|\n)\s*\d+[.)]\s*(.+)", response)
        while len(lines) < len(batch):
            lines.append("")
        return [f"[Context: {p.strip()}]" if p.strip() else "" for p in lines[: len(batch)]]

    def _chat(self, prompt: str, num_predict: int = 200) -> str:
        """Chat, returning just the result (retry handled internally)."""
        return self._chat_with_attempts(prompt, num_predict=num_predict)[0]

    def _chat_with_attempts(self, prompt: str, num_predict: int = 200) -> tuple[str, int]:
        """Chat with retry, returning (result, attempts_used).

        attempts_used lets callers report how many tries a call took before
        success or final failure — 1 = no retry, 3 = both retries exhausted.
        """
        model = config.CONTEXT_MODEL or config.CHAT_MODEL
        return self._llm.chat_sync_with_attempts(prompt, model=model, num_predict=num_predict)


def strip_context_prefix(content: str) -> str:
    """Remove ``[Context: ...]`` prefix from content for clean display."""
    return re.sub(r"^\[Context:.*?\]\s*", "", content)
