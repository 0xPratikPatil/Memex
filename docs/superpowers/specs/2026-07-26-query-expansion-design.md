# Query Expansion Design: HyDE + Multi-Query

**Date**: 2026-07-26
**Status**: Draft
**Author**: Opencode

---

## Problem Statement

Complex or ambiguous queries like "how does the revenue compare to last year" or "security implications of the architecture" fail to retrieve relevant chunks because:

1. **Lexical mismatch**: User query terms don't appear verbatim in documents (e.g., "revenue" vs. "financial performance", "security" vs. "authentication/authorization").
2. **Single-query limitation**: One query formulation captures only one angle of a multi-faceted question.
3. **Under-specified queries**: Short queries lack the specificity needed to discriminate between many similar chunks.

Current pipeline (`pipeline.py:449-539`) runs a single dense+sparse search per query. This is a bottleneck for recall on complex questions.

---

## Solution Overview

Add a query expansion layer that runs **before** the existing hybrid search:

1. **HyDE (Hypothetical Document Embeddings)**: Generate a hypothetical answer, embed it, and use that embedding for dense retrieval. The hypothetical document acts as a bridge between query space and document space.
2. **Multi-Query**: Generate N paraphrases of the original query, run search for each, and merge results via RRF.
3. **Query Rewriting**: Normalize/rewrite the query for better keyword matching.

All three are optional, composable, and controlled by feature flags.

---

## Architecture

```
┌─────────────┐
│  User Query  │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  Query Expansion  │  (new module)
│  Layer            │
│                   │
│  ┌─────────────┐  │
│  │ Query Rewriter│  │  normalize, expand abbreviations
│  └──────┬──────┘  │
│         │         │
│  ┌──────▼──────┐  │
│  │ HyDE Module  │  │  generate hypothetical doc → embed
│  └──────┬──────┘  │
│         │         │
│  ┌──────▼──────┐  │
│  │ Multi-Query  │  │  N paraphrases → embed each
│  │ Generator    │  │
│  └──────┬──────┘  │
│         │         │
└─────────┼─────────┘
          │
          ▼
┌──────────────────────┐
│  Hybrid Search        │  (existing pipeline.py)
│  (dense + sparse +   │
│   RRF + rerank)       │
└──────────────────────┘
```

### Flow

1. User sends query `q`.
2. **Query Rewriter** (optional): rewrites `q` → `q'` (e.g., expands "revenue" to "revenue financial performance income").
3. **HyDE** (optional): generates hypothetical document `d_hyde`, embeds it to get `v_hyde`.
4. **Multi-Query** (optional): generates `[q1, q2, ..., qN]` paraphrases, embeds each.
5. All vectors (`v_hyde`, `v_q1...v_qN`) feed into dense search alongside the original sparse search.
6. Results merged via RRF, then reranked as before.

---

## Files to Modify/Create

| File | Action | Purpose |
|------|--------|---------|
| `src/query_expansion.py` | **Create** | Query rewriter, HyDE, multi-query generator |
| `src/config.py` | **Modify** | Add feature toggles and model config |
| `src/pipeline.py` | **Modify** | Integrate expansion into `hybrid_search` |
| `tests/unit/test_query_expansion.py` | **Create** | Unit tests for each expansion module |
| `tests/integration/test_expansion_search.py` | **Create** | End-to-end tests with Qdrant |

---

## Implementation Details

### 1. Configuration (`config.py` additions)

```python
# ── Query Expansion ──────────────────────────────────────────────────────────
ENABLE_QUERY_EXPANSION: bool = _env_bool("ENABLE_QUERY_EXPANSION", False)
ENABLE_HYDE: bool = _env_bool("ENABLE_HYDE", False)
ENABLE_MULTI_QUERY: bool = _env_bool("ENABLE_MULTI_QUERY", False)
ENABLE_QUERY_REWRITE: bool = _env_bool("ENABLE_QUERY_REWRITE", False)

HYDE_MODEL: str = _env("HYDE_MODEL", "")  # empty = use same EMBED_MODEL via Ollama
MULTI_QUERY_COUNT: int = _env_int("MULTI_QUERY_COUNT", 3)
MULTI_QUERY_MODEL: str = _env("MULTI_QUERY_MODEL", "")  # empty = use Ollama chat
QUERY_REWRITE_MODEL: str = _env("QUERY_REWRITE_MODEL", "")
```

### 2. Query Expansion Module (`src/query_expansion.py`)

```python
"""Query expansion: HyDE, Multi-Query, Query Rewriting."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import httpx
from . import config

@dataclass
class ExpandedQuery:
    """Result of query expansion."""
    original: str
    rewritten: str | None = None          # after rewrite
    hyde_vector: list[float] | None = None # hypothetical doc embedding
    paraphrases: list[str] | None = None   # multi-query paraphrases

class QueryExpander:
    """Orchestrates query expansion techniques."""

    def __init__(self, ollama_client: httpx.Client):
        self._ollama = ollama_client

    def expand(self, query: str) -> ExpandedQuery:
        result = ExpandedQuery(original=query)

        if config.ENABLE_QUERY_REWRITE:
            result.rewritten = self._rewrite(query)

        effective_query = result.rewritten or query

        if config.ENABLE_HYDE:
            result.hyde_vector = self._hyde_embed(effective_query)

        if config.ENABLE_MULTI_QUERY:
            result.paraphrases = self._multi_query(effective_query)

        return result

    def _rewrite(self, query: str) -> str:
        """Rewrite query using LLM to expand abbreviations, fix phrasing."""
        prompt = f"Rewrite this search query to be more specific and clear. " \
                 f"Keep it under 50 words. Only output the rewritten query.\n\n" \
                 f"Query: {query}"
        return self._chat(prompt)

    def _hyde_embed(self, query: str) -> list[float]:
        """Generate hypothetical document and return its embedding."""
        prompt = f"Write a short paragraph that would be a perfect answer to this query. " \
                 f"Be factual and specific. 3-5 sentences.\n\n" \
                 f"Query: {query}"
        hypothetical = self._chat(prompt)
        return self._embed(hypothetical)

    def _multi_query(self, query: str) -> list[str]:
        """Generate N paraphrases of the query."""
        prompt = f"Generate {config.MULTI_QUERY_COUNT} diverse paraphrases of this search query. " \
                 f"Each on a new line. Only the paraphrases, no numbering.\n\n" \
                 f"Query: {query}"
        response = self._chat(prompt)
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        return lines[:config.MULTI_QUERY_COUNT]

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

    def _embed(self, text: str) -> list[float]:
        """Embed text via Ollama."""
        resp = self._ollama.post(
            config.OLLAMA_EMBED_URL,
            json={"model": config.EMBED_MODEL, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
```

### 3. Pipeline Integration (`pipeline.py` changes)

Modify `hybrid_search` to accept and use expanded queries:

```python
def hybrid_search(
    self,
    query: str,
    top_k: int = 5,
    rerank: bool = True,
    source_filter: str | None = None,
    expanded_query: ExpandedQuery | None = None,  # NEW
) -> list[dict[str, Any]]:
    # Use expanded query for dense search if available
    dense_query = query
    if expanded_query:
        if expanded_query.rewritten:
            dense_query = expanded_query.rewritten

    query_dense = self._dense_embed(dense_query)
    # ... rest of existing logic

    # If HyDE vector available, merge it as an additional dense result
    if expanded_query and expanded_query.hyde_vector:
        hyde_hits = qdrant.query_points(
            collection_name=config.COLLECTION_NAME,
            query=expanded_query.hyde_vector,
            using="dense",
            limit=candidate_k,
            query_filter=qdrant_filter,
        ).points
        # Merge into RRF scoring (same loop as dense_hits)

    # If paraphrases available, run search for each and merge via RRF
    if expanded_query and expanded_query.paraphrases:
        for para in expanded_query.paraphrases:
            para_dense = self._dense_embed(para)
            para_hits = qdrant.query_points(...)
            # Merge into RRF scoring
```

### 4. MCP Server Integration

In `server.py`, the `rag_query` tool:

```python
async def rag_query(query, top_k=5, use_reranking=True, source_filter=None, ...):
    engine = _get_engine()

    # Query expansion
    expanded = None
    if config.ENABLE_QUERY_EXPANSION:
        from src.query_expansion import QueryExpander
        expander = QueryExpander(engine._get_ollama())
        expanded = expander.expand(query)

    results = engine.hybrid_search(
        query=query, top_k=top_k, rerank=use_reranking,
        source_filter=source_filter, expanded_query=expanded,
    )
    # ... format output
```

---

## Testing Strategy

### Unit Tests (`tests/unit/test_query_expansion.py`)

- `test_rewrite_returns_string`: Verify rewrite produces non-empty string.
- `test_hyde_returns_vector`: Verify HyDE embedding has correct dimensions.
- `test_multi_query_returns_n_paraphrases`: Verify correct count of paraphrases.
- `test_expansion_disabled_passthrough`: When all flags off, `expand()` returns original query only.
- `test_expanded_query_dataclass`: Verify fields populated correctly.

### Integration Tests (`tests/integration/test_expansion_search.py`)

- `test_hyde_improves_recall`: Ingest known document, verify HyDE finds relevant chunk that plain query misses.
- `test_multi_query_merges_results`: Verify paraphrase results are merged and deduplicated.
- `test_end_to_end_with_expansion`: Full flow from query → expansion → search → results.

### Mock Strategy

- Mock Ollama HTTP calls with `httpx.MockTransport` or `pytest-httpx`.
- Use real Qdrant (test instance) for integration tests.
- Pre-ingest a small test corpus (3-5 documents with known content).

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM hallucination in HyDE generates irrelevant document | Poor dense retrieval quality | Keep HyDE output short; fallback to original embedding if HyDE retrieval quality is poor (measure via rerank score) |
| Latency increase from N+1 LLM calls | Slower search response | Make all expansion optional; cache paraphrases for repeated similar queries; set strict timeouts on LLM calls |
| Paraphrases diverge from original intent | Retrieved results off-topic | Use constrained prompting; validate paraphrase similarity via cosine distance on embeddings |
| Ollama overloaded with concurrent expansion calls | degraded embedding quality | Rate-limit expansion calls; share connection pool with existing embedding calls |
| Memory increase from multiple embedding vectors | Higher RAM usage | Each vector is ~4KB (1024 dims × 4 bytes); 5 paraphrases = ~20KB — negligible |

---

## Priority & Effort

- **Priority**: Medium (improves recall for complex queries, not critical for basic functionality)
- **Estimated effort**: 2-3 days
- **Dependencies**: None (uses existing Ollama/Qdrant stack)
- **Rollback**: Feature flags default to `False`; disable via env vars to revert
