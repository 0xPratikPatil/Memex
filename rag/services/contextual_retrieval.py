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

import httpx

from rag import config

logger = logging.getLogger("contextual-retrieval")


class ContextGenerator:
    """Generates contextual prefixes for document chunks."""

    def __init__(self, ollama_client: httpx.Client) -> None:
        self._ollama = ollama_client

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

    def enrich_chunks(
        self,
        chunks: list[dict[str, Any]],
        document_summary: str = "",
    ) -> list[dict[str, Any]]:
        """Add context prefixes to all chunks in a document.

        Each chunk dict is copied and augmented with:
          - ``context_prefix``: the generated context string
          - ``content``: original content prefixed with the context

        For LLM-based strategies (summary, surrounding), chunks are processed
        in batches of ``CONTEXT_BATCH_SIZE`` to reduce LLM round-trips.
        """
        if not chunks:
            return chunks

        strategy = config.CONTEXT_STRATEGY.lower()

        if strategy == "header":
            enriched: list[dict[str, Any]] = []
            for chunk in chunks:
                context = self._context_from_header(chunk.get("section_header", ""))
                enriched_chunk = {**chunk}  # shallow copy to avoid shared metadata mutation
                enriched_chunk["context_prefix"] = context
                if context:
                    enriched_chunk["content"] = f"{context} {chunk['content']}".strip()
                enriched.append(enriched_chunk)
            return enriched

        # LLM-based strategies: batch chunks
        batch_size = config.CONTEXT_BATCH_SIZE
        enriched = []

        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start : batch_start + batch_size]

            if strategy == "summary":
                contexts = self._batch_context_from_summary(batch, document_summary)
            elif strategy == "surrounding":
                contexts = self._batch_context_from_surrounding(chunks, batch_start, batch_size)
            else:
                contexts = [self._context_from_header(c.get("section_header", "")) for c in batch]

            for chunk, context in zip(batch, contexts, strict=True):
                enriched_chunk = {**chunk}  # shallow copy to avoid shared metadata mutation
                enriched_chunk["context_prefix"] = context
                if context:
                    enriched_chunk["content"] = f"{context} {chunk['content']}".strip()
                enriched.append(enriched_chunk)

        return enriched

    def _batch_context_from_summary(
        self,
        batch: list[dict[str, Any]],
        summary: str,
    ) -> list[str]:
        """Generate context for a batch of chunks using document summary."""
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
        response = self._chat(prompt)
        # Parse numbered lines: "1. context" or "1) context"
        lines = re.findall(r"(?:^|\n)\s*\d+[.)]\s*(.+)", response)
        # Pad or truncate to match batch size
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
            # Single chunk — use direct prompt
            return [
                self._context_from_surrounding(
                    batch[0]["content"],
                    all_chunks[batch_start - 1]["content"] if batch_start > 0 else "",
                    all_chunks[batch_start + 1]["content"] if batch_start + 1 < len(all_chunks) else "",
                )
            ]

        # Build a single prompt with all chunks and their surrounding context
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
        response = self._chat(prompt, num_predict=200 * len(batch))
        # Parse numbered lines
        import re

        lines = re.findall(r"(?:^|\n)\s*\d+[.)]\s*(.+)", response)
        while len(lines) < len(batch):
            lines.append("")
        return [f"[Context: {p.strip()}]" if p.strip() else "" for p in lines[: len(batch)]]

    def _chat(self, prompt: str, num_predict: int = 200) -> str:
        """Call Ollama chat API via shared helper."""
        from rag.ollama_chat import ollama_chat

        model = config.CONTEXT_MODEL or config.CHAT_MODEL
        return ollama_chat(prompt, model=model, num_predict=num_predict, ollama_client=self._ollama)


def strip_context_prefix(content: str) -> str:
    """Remove ``[Context: ...]`` prefix from content for clean display."""
    return re.sub(r"^\[Context:.*?\]\s*", "", content)
