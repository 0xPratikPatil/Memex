# Contextual Retrieval Design

**Date**: 2026-07-26
**Status**: Draft
**Author**: Opencode

---

## Problem Statement

When a document is chunked, each chunk loses its surrounding context. A chunk about "quarterly revenue" doesn't know it's from "Q3 2025 Financial Report" or that the preceding section discussed "operating expenses". This degrades retrieval because:

1. **Ambiguous chunks**: "Revenue increased 15%" could belong to any company's report.
2. **Lost document structure**: Chunks don't carry information about their parent section, document type, or position.
3. **Weaker embeddings**: Without context, embeddings are less semantically rich.

The current chunking pipeline (`pipeline.py:60-196`) already extracts `section_header` but nothing else about the chunk's position in the document.

---

## Solution Overview

Based on Anthropic's contextual retrieval technique:

1. **Context Generation**: For each chunk, use an LLM to generate a short contextual prefix that situates the chunk within the document.
2. **Context Prefixing**: Prepend the context to the chunk content before embedding and storing.
3. **Context Storage**: Store the context separately in metadata for retrieval-side use.

```
Original chunk:  "Revenue increased 15% to $2.3B"
Context prefix:  "From Q3 2025 Financial Report, Revenue section:"
Enriched chunk:  "From Q3 2025 Financial Report, Revenue section: Revenue increased 15% to $2.3B"
```

---

## Architecture

```
┌──────────────┐
│ Document Text │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Chunking     │  (existing recursive chunker)
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│  Context Generator    │  (new)
│                       │
│  For each chunk:      │
│  - Send chunk +       │
│    doc summary to LLM │
│  - Get 1-sentence     │
│    context prefix     │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Enriched Chunks      │
│  content = prefix +   │
│            original    │
│  metadata.context =   │
│            prefix      │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Embedding + Storage   │  (existing pipeline)
└──────────────────────┘
```

### Context Generation Strategies

1. **Document Summary + Chunk** (recommended): LLM receives a document summary + the chunk, generates context.
2. **Surrounding Chunks**: LLM receives the previous and next chunk + current chunk.
3. **Header + Chunk**: LLM receives section header + chunk (fastest, no extra LLM calls for structure).

Strategy 1 is most effective but requires an initial LLM call per document. Strategy 3 is cheapest and a good starting point.

---

## Files to Modify/Create

| File | Action | Purpose |
|------|--------|---------|
| `src/contextual_retrieval.py` | **Create** | Context generation, prefixing logic |
| `src/config.py` | **Modify** | Add contextual retrieval toggles |
| `src/pipeline.py` | **Modify** | Integrate context generation into `ingest_text` |
| `tests/unit/test_contextual_retrieval.py` | **Create** | Unit tests for context generation |
| `tests/integration/test_contextual_ingest.py` | **Create** | Test enriched ingestion end-to-end |

---

## Implementation Details

### 1. Configuration (`config.py` additions)

```python
# ── Contextual Retrieval ────────────────────────────────────────────────────
ENABLE_CONTEXTUAL_RETRIEVAL: bool = _env_bool("ENABLE_CONTEXTUAL_RETRIEVAL", False)
CONTEXT_STRATEGY: str = _env("CONTEXT_STRATEGY", "header")  # header | summary | surrounding
CONTEXT_PREFIX_MAX_TOKENS: int = _env_int("CONTEXT_PREFIX_MAX_TOKENS", 50)
CONTEXT_BATCH_SIZE: int = _env_int("CONTEXT_BATCH_SIZE", 10)  # chunks per LLM call
```

### 2. Context Generation Module (`src/contextual_retrieval.py`)

```python
"""Contextual retrieval: generate context prefixes for chunks."""

from __future__ import annotations
import logging
from typing import Any
import httpx
from . import config

logger = logging.getLogger("contextual-retrieval")

class ContextGenerator:
    """Generates contextual prefixes for document chunks."""

    def __init__(self, ollama_client: httpx.Client):
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
            return self._context_from_header(chunk, section_header)
        elif strategy == "summary":
            return self._context_from_summary(chunk, document_summary)
        elif strategy == "surrounding":
            return self._context_from_surrounding(chunk, prev_chunk, next_chunk)
        else:
            return self._context_from_header(chunk, section_header)

    def _context_from_header(self, chunk: str, header: str) -> str:
        """Fastest: derive context from section header only."""
        if not header:
            return ""
        return f"[Context: {header}]"

    def _context_from_summary(self, chunk: str, summary: str) -> str:
        """LLM-based: use document summary to generate context."""
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
            + "\n".join(context_parts) + f"\n\nChunk: {chunk[:300]}"
        )
        response = self._chat(prompt)
        return f"[Context: {response.strip()}]"

    def enrich_chunks(
        self,
        chunks: list[dict[str, Any]],
        document_summary: str = "",
    ) -> list[dict[str, Any]]:
        """Add context prefixes to all chunks in a document."""
        if not chunks:
            return chunks

        enriched = []
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
            enriched_chunk["content"] = f"{context} {chunk['content']}".strip()
            enriched.append(enriched_chunk)

        return enriched

    def _chat(self, prompt: str) -> str:
        """Call Ollama chat API."""
        resp = self._ollama.post(
            config.OLLAMA_EMBED_URL.replace("/api/embeddings", "/api/chat"),
            json={
                "model": config.EMBED_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
```

### 3. Pipeline Integration (`pipeline.py` changes)

Modify `ingest_text` to optionally generate contexts:

```python
def ingest_text(self, text, source_identifier, metadata=None, content_hash="", progress_cb=None):
    # ... existing chunking ...
    raw_chunks = create_chunks(text)

    # NEW: Contextual retrieval
    if config.ENABLE_CONTEXTUAL_RETRIEVAL:
        from src.contextual_retrieval import ContextGenerator
        ctx_gen = ContextGenerator(self._get_ollama())

        _progress("Generating document context...", 72)
        summary = ctx_gen.generate_document_summary(text) if config.CONTEXT_STRATEGY == "summary" else ""

        _progress("Adding context to chunks...", 73)
        raw_chunks = ctx_gen.enrich_chunks(raw_chunks, document_summary=summary)

    # ... existing embedding and storage ...
```

### 4. Metadata Storage

Each chunk's metadata already includes `section_header`. Add `context_prefix`:

```python
point_meta = {
    "source": source_identifier,
    "chunk_index": idx,
    "total_chunks": len(raw_chunks),
    "content": chunk["content"],  # now includes context prefix
    "section_header": chunk.get("section_header", ""),
    "context_prefix": chunk.get("context_prefix", ""),  # NEW
    "ingested_at": now,
    "content_hash": content_hash,
    **base_meta,
}
```

### 5. Retrieval-Side Use

The context prefix is embedded in the content, so it naturally improves dense retrieval. For display, optionally strip the prefix:

```python
def strip_context_prefix(content: str) -> str:
    """Remove [Context: ...] prefix from content for clean display."""
    import re
    return re.sub(r"^\[Context:.*?\]\s*", "", content)
```

---

## Testing Strategy

### Unit Tests (`tests/unit/test_contextual_retrieval.py`)

- `test_header_context_returns_string`: Header strategy returns non-empty string.
- `test_summary_context_returns_string`: Summary strategy returns non-empty string.
- `test_empty_header_returns_empty`: No header → empty context.
- `test_enrich_chunks_preserves_count`: Enriched chunks count equals input count.
- `test_enrich_chunks_adds_context_prefix`: Each chunk has `context_prefix` field.
- `test_enrich_chunks_modifies_content`: Content starts with context prefix.
- `test_strip_context_prefix`: Verify prefix removal works correctly.

### Integration Tests (`tests/integration/test_contextual_ingest.py`)

- `test_ingest_with_context`: Ingest document with contextual retrieval enabled, verify chunks have context prefixes in Qdrant payload.
- `test_retrieval_with_context`: Verify context-enriched chunks are retrieved more accurately than plain chunks.
- `test_ingest_without_context`: Verify backward compatibility when feature is disabled.

### Evaluation

Compare retrieval quality with/without contextual retrieval:
- Ingest 10 documents, run 20 queries
- Measure hit@5, MRR with and without context
- Expected improvement: 5-15% on ambiguous queries

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM calls slow down ingestion significantly | Slow ingest for large documents | Use header strategy (no LLM calls) as default; batch context generation |
| Context prefix adds noise to embeddings | Worse retrieval quality | Test prefix format; keep prefix short (< 30 words); A/B test |
| Increased storage from longer chunk content | Higher Qdrant storage | Each prefix is ~100-200 chars; negligible impact |
| LLM generates inaccurate context | Misleading retrieval | Use header strategy to avoid LLM hallucination; validate in tests |
| Context prefixes become stale on re-ingest | Inconsistent data | Context generated fresh on each ingest; no caching needed |

---

## Priority & Effort

- **Priority**: High (directly improves retrieval quality with minimal code changes)
- **Estimated effort**: 1-2 days
- **Dependencies**: None (uses existing Ollama)
- **Rollback**: Feature flag `ENABLE_CONTEXTUAL_RETRIEVAL` defaults to `False`
