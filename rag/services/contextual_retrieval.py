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
        """
        if not chunks:
            return chunks

        enriched: list[dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            prev = chunks[i - 1]["content"] if i > 0 else ""
            next_ = chunks[i + 1]["content"] if i < len(chunks) - 1 else ""

            context = self.generate_context(
                chunk=chunk["content"],
                document_summary=document_summary,
                section_header=chunk.get("section_header", ""),
                prev_chunk=prev,
                next_chunk=next_,
            )

            enriched_chunk = chunk.copy()
            enriched_chunk["context_prefix"] = context
            if context:
                enriched_chunk["content"] = f"{context} {chunk['content']}".strip()
            enriched.append(enriched_chunk)

        return enriched

    def _chat(self, prompt: str) -> str:
        """Call Ollama chat API and return the assistant message content."""
        chat_url = config.OLLAMA_EMBED_URL.replace("/api/embeddings", "/api/chat")
        resp = self._ollama.post(
            chat_url,
            json={
                "model": config.CONTEXT_MODEL or config.EMBED_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def strip_context_prefix(content: str) -> str:
    """Remove ``[Context: ...]`` prefix from content for clean display."""
    return re.sub(r"^\[Context:.*?\]\s*", "", content)
