# Ingestion Performance Refactor — Batched Embedding, Async Pipeline, Checkpointing

**Date**: 2026-07-28
**Status**: Draft
**Author**: Opencode

---

## Problem Statement

Ingestion of even 2-3 documents (5-10 MB total) takes far too long. The root causes span the embedding transport layer, idempotency check ordering, fallback model configuration, and lack of batch resilience:

1. **N+1 embedding HTTP calls**: `_embed_via_ollama` sends one HTTP request per text chunk, even when the caller provides a batch. For a 64-chunk document with contextual retrieval enabled, that is 128 sequential round-trips to Ollama at ~0.23 s each — roughly 29 seconds wasted on HTTP overhead alone.

2. **Idempotency check after Docling**: The content hash comparison runs _after_ the expensive Docling parse. Already-ingested files still pay the full conversion cost (up to 300 s timeout) before the system discovers they are unchanged.

3. **Fallback models are no-ops**: The `.env` sets `EMBED_MODEL_FALLBACK` to the same value as `EMBED_MODEL`, and likewise for the reranker. When a model call fails, the "fallback" retries the identical model, burning retry time with zero benefit.

4. **No batch checkpointing**: If the MCP client disconnects or times out mid-batch, restarting reprocesses everything from scratch. There is no persisted state.

5. **Fully sequential ingestion**: Docling parsing runs one file at a time, blocking the next file on network-I/O wait. Concurrent parsing would shave minutes off multi-file batches.

6. **Query expansion and contextual retrieval on by default**: Every search fires 4+ LLM calls (rewrite + HyDE + multi-query × 3). Contextual retrieval doubles embedding work during ingestion. These are good features but inappropriate defaults for small document collections.

7. **Stale metadata visible after disabling**: When metadata extraction is turned off, previously extracted metadata (entities, topics, keywords) remains in Qdrant payloads and is rendered in search results, giving the false impression that extraction is still running.

---

## Solution Overview

A performance refactor across the ingestion pipeline, embedding service, configuration defaults, and batch resilience:

1. **EmbeddingService**: Extract a standalone service with a single `embed(texts)` contract guaranteeing O(N/batch) HTTP calls via Ollama's batched `/api/embed` endpoint.
2. **Two-phase idempotency**: Phase 1 checks Qdrant for source existence. Phase 2 compares file mtime+size for local files (skips Docling entirely if unchanged).
3. **Async ingestion orchestrator**: Concurrent Docling parsing with semaphore-capped parallelism via `asyncio.to_thread`.
4. **Batch checkpointing**: JSON state file on disk tracks per-file progress; batch resumes on restart.
5. **Per-document timeouts**: Configurable parse and total timeouts per file; stuck files are skipped, not fatal.
6. **Sensible config defaults**: Batched embedding endpoint, different fallback models, query expansion off, contextual retrieval off.
7. **Metadata cleanup on ingest**: When metadata extraction is disabled, explicitly write empty metadata fields to Qdrant, clearing stale data.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  MCP Tool Layer (memex/server.py)                                │
│                                                                  │
│  rag_ingest_file / rag_ingest_url / rag_ingest_batch             │
│  - Progress reporting, friendly error translation                │
│  - Delegates to IngestionOrchestrator                            │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│  IngestionOrchestrator (new: rag/ingestion.py)                   │
│                                                                  │
│  - Pre-check: source existence in Qdrant                         │
│  - Concurrent Docling parsing (asyncio + semaphore)              │
│  - Batch checkpointing (JSON state file)                         │
│  - Per-document timeouts with graceful skip                      │
│  - Per-stage timing / metrics                                    │
│  - Delegates chunk→embed→upsert to RAGEngine                     │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│  EmbeddingService (new: rag/embedding.py)                        │
│                                                                  │
│  embed(texts: list[str]) -> list[list[float]]                    │
│  - Cache check per-text → collect uncached                       │
│  - Batch via /api/embed (single POST per sub-batch of 64)        │
│  - Model fallback per sub-batch, not per text                    │
│  - Skips fallback if primary == fallback model                   │
└───────────────────────────┬──────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────┐
│  RAGEngine (rag/pipeline.py — modified)                          │
│                                                                  │
│  - _dense_embed_batch delegates to EmbeddingService              │
│  - ingest_text: writes empty metadata when extraction disabled   │
│  - Chunking, upsert, search unchanged                            │
└──────────────────────────────────────────────────────────────────┘
```

**Isolation**: The orchestrator owns scheduling, timeouts, and progress. The EmbeddingService owns transport batching and caching. RAGEngine owns document processing logic. Each can be tested independently.

---

## Section 1: EmbeddingService — Batched Transport

### Contract

```python
class EmbeddingService:
    """Embed text batches via Ollama with native batching and model fallback."""

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Return embeddings for all texts.

        Guarantees at most ceil(len(texts) / EMBED_BATCH_SIZE) HTTP calls.
        Results are returned in the same order as input texts.
        """
```

### Internal flow

1. Per-text Redis cache lookup → collect uncached texts with original indices
2. Split uncached into sub-batches of `EMBED_BATCH_SIZE` (default 64)
3. For each sub-batch: single POST to `OLLAMA_EMBED_URL` (`/api/embed`) with `{"model": model, "input": [text, text, ...]}`
4. On model failure: if `EMBED_MODEL_FALLBACK` differs from primary, switch model for that sub-batch only and retry once. If identical, skip fallback and raise.
5. Write cache per-text on success via `cache_embedding()`
6. Reassemble results in original index order

### Config changes

| Var | Old Default | New Default |
|-----|-------------|-------------|
| `OLLAMA_EMBED_URL` | `/api/embeddings` | `/api/embed` |
| `EMBED_MODEL_FALLBACK` | `bge-m3` (same as primary) | `qwen3-embedding:0.6b` (different) |

### Backward compatibility

At startup, if `OLLAMA_EMBED_URL` is the _default_ value (i.e., the user hasn't set it in `.env`), the URL is transparently set to `/api/embed`. If the user has set a custom `OLLAMA_EMBED_URL` explicitly (in `.env` or env var), it is left untouched — only a warning is logged if it contains `/api/embeddings`. This avoids breaking anyone who deliberately chose the legacy endpoint.

### Impact

128 sequential round-trips → 1-2 batched calls. ~29 s → ~0.5 s per document on embedding.

---

## Section 2: Two-Phase Idempotency

### Phase 1 (cheap): Source-level check

Before any Docling call, query Qdrant for a point where `source == identifier`. If no matching points exist, the file is new and proceeds to ingestion.

### Phase 2 (medium cost): Content change detection

For already-ingested sources:

- **Local files**: Compare `os.stat().st_mtime` and `os.stat().st_size` against stored payload fields `file_mtime` and `file_size`. If both match, skip parsing and ingestion entirely — return "unchanged, skipped" immediately.
- **URLs**: Must still fetch and parse to get the content hash. The existing Redis parse cache (7-day TTL) mitigates this. The cache lookup in `docling_client.py` is moved to execute _before_ the Docling HTTP call, not inside it — so a cache hit skips conversion.

### New Qdrant payload fields

| Field | Type | Description |
|-------|------|-------------|
| `file_mtime` | float | `st_mtime` at time of ingestion (local files only) |
| `file_size` | int | `st_size` at time of ingestion (local files only) |
| `content_hash` | str | SHA256 of markdown content (existing, unchanged) |

### Fallback behavior

If Qdrant is unreachable during the pre-check, the system proceeds with ingestion (fail-open) and logs a warning. It is better to re-ingest than to refuse service.

---

## Section 3: Async Ingestion Orchestrator

### File: `rag/ingestion.py`

```python
class IngestionOrchestrator:
    """Coordinates concurrent parsing, checkpointing, and timeout management."""

    def __init__(self, engine: RAGEngine) -> None: ...
    async def ingest_batch(self, items: list[str]) -> dict[str, str]: ...
    async def ingest_single(self, item: str) -> str: ...
```

### Concurrent parsing

Uses `asyncio.to_thread` to run synchronous `parse_file()` calls in a thread pool. A semaphore caps concurrent parses at `MAX_CONCURRENT_PARSES=3` to avoid overwhelming Docling's GPU memory.

```
Time →
File A: [Docling parse (60s)] ───── [chunk→embed→upsert (5s)]
File B: [Docling parse (45s)] ────────────────── [chunk→embed→upsert (4s)]
File C: [Docling parse (30s)] ───────────────────────────── [chunk→embed→upsert (3s)]

Sequential: 60+5+45+4+30+3 = 147 s
Pipelined:  max(60,45,30) + 5 + 4 + 3 = 72 s
```

### Ingestion phase

After all parses complete, each document's chunk→embed→upsert proceeds _sequentially_ (not concurrently). Embedding is already batched and fast after Section 1's fix. Concurrent Qdrant upserts would add locking complexity for marginal gain.

### Design decision — why not rewrite docling_client

The docling_client is well-tested with tenacity retries and a clean `ConversionResult` interface. Wrapping in `run_in_executor` gives us concurrency without touching it. If Docling ever ships a native async client, we swap the executor for async calls behind the same `IngestionOrchestrator` interface.

---

## Section 4: Batch Checkpointing & Resume

### State file

```
{memex_data_dir}/batches/{batch_id}.json
```

Where `memex_data_dir` = `$MEMEX_DATA_DIR` env var, or `~/.memex` (default), or `/tmp/memex` if `$HOME` is unavailable.

```json
{
  "batch_id": "a1b2c3d4",
  "created_at": "2026-07-28T10:00:00Z",
  "completed_at": null,
  "items": ["/docs/a.pdf", "/docs/b.pdf", "/docs/c.pdf"],
  "completed": ["/docs/a.pdf"],
  "failed": {
    "/docs/b.pdf": "timeout: parse exceeded 120s"
  }
}
```

### Behavior

- **On batch start**: Write state file, mark all items pending. `batch_id` is a SHA256 of the sorted item list — same items → same ID → resume.
- **After each file succeeds**: Append to `completed` list, flush to disk.
- **On file failure**: Record in `failed` dict with error reason, continue to next file.
- **On crash/disconnect/timeout**: State file remains on disk.
- **On next `rag_ingest_batch` with same items**: Detect matching incomplete batch, auto-resume — skip `completed` items, reprocess the rest. Log: "Resuming batch a1b2c3d4 — 2 of 5 completed, 1 failed, 2 remaining."
- **On `rag_ingest_batch` with different items**: Warn about orphaned incomplete batch from a previous run, offer to abandon it.

### Why a file, not Redis

The state file survives `docker compose down`, `docker compose down -v`, and system reboot. Redis data is ephemeral by design and gets wiped when the volume is removed. A JSON file on the host filesystem is simpler, zero-dependency, and the right durability guarantee.

### Cleanup

Completed batches older than 7 days are cleaned on orchestrator init. Incomplete batches older than 30 days are cleaned with a warning.

---

## Section 5: Per-Document Timeouts

### Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `INGEST_TIMEOUT_PARSE` | 120 | Max seconds for Docling parse of one file |
| `INGEST_TIMEOUT_TOTAL` | 300 | Max seconds for full parse+chunk+embed+upsert per file |
| `MAX_CONCURRENT_PARSES` | 3 | Max concurrent Docling parse operations |

### Implementation

- `INGEST_TIMEOUT_PARSE`: wraps the Docling HTTP call in `asyncio.wait_for()`. If exceeded, cancel the parse for that file only, log as `failed: timeout`, continue to next file.
- `INGEST_TIMEOUT_TOTAL`: wraps the full per-file pipeline. If embedding or upsert gets stuck, the file is skipped — the batch never dies because of one bad file.

---

## Section 6: Config Defaults & Startup Checks

### New defaults (config.py)

| Var | Old Default | New Default | Rationale |
|-----|-------------|-------------|-----------|
| `OLLAMA_EMBED_URL` | `http://localhost:11434/api/embeddings` | `http://localhost:11434/api/embed` | Native batching |
| `EMBED_MODEL_FALLBACK` | `bge-m3` | `qwen3-embedding:0.6b` | Different model, already pulled |
| `RERANK_MODEL_FALLBACK` | (same as primary) | `BAAI/bge-reranker-base` | Actual fallback |
| `ENABLE_QUERY_EXPANSION` | `true` | `false` | Opt-in; 4+ LLM calls per search |
| `ENABLE_HYDE` | `true` | `false` | Sub-flag of expansion |
| `ENABLE_MULTI_QUERY` | `true` | `false` | Sub-flag of expansion |
| `ENABLE_QUERY_REWRITE` | `true` | `false` | Sub-flag of expansion |
| `ENABLE_CONTEXTUAL_RETRIEVAL` | `true` | `false` | Halves embedding cost; opt-in |
| `INGEST_TIMEOUT_PARSE` | (none) | `120` | New |
| `INGEST_TIMEOUT_TOTAL` | (none) | `300` | New |
| `MAX_CONCURRENT_PARSES` | (none) | `3` | New |

### Runtime safety checks (startup, logged at WARNING level)

1. If `EMBED_MODEL == EMBED_MODEL_FALLBACK`: "Embedding fallback model is identical to primary — fallback will have no effect."
2. If `RERANK_MODEL == RERANK_MODEL_FALLBACK`: "Rerank fallback model is identical to primary — fallback will have no effect."
3. If `OLLAMA_EMBED_URL` contains `/api/embeddings`: auto-rewritten to `/api/embed` with a warning.
4. If `ENABLE_CONTEXTUAL_RETRIEVAL` is on but the collection has no `contextual_dense` vector: warn that re-creation may be needed.

### Backward compatibility

Existing `.env` files are unaffected — env var > .env > default. Only out-of-box behavior changes. `.env.example` is updated.

---

## Section 7: Stale Metadata Cleanup

### Problem

When metadata extraction is disabled via env vars, previously extracted entities, topics, and keywords remain in Qdrant payloads. The search result display code renders whatever is in the payload, giving the false impression that extraction is still active.

### Fix — two parts

**Ingestion side** (`pipeline.py:ingest_text`): When `ENABLE_METADATA_EXTRACTION` is false, explicitly write empty metadata fields:

```python
if not config.ENABLE_METADATA_EXTRACTION:
    point_meta["doc_type"] = ""
    point_meta["topics"] = []
    point_meta["language"] = ""
    point_meta["keywords"] = []
    point_meta["entities"] = {}
```

Re-ingestion of previously-ingested files will overwrite stale metadata with empty values.

**Display side** (`server.py:rag_query` markdown output): Only render metadata fields when they contain meaningful data (non-empty string, non-empty list). This prevents displaying "Type: " or "Topics: " with empty values.

---

## Section 8: Error Handling Strategy

| Failure | Behavior |
|---------|----------|
| Docling unreachable | Translated to actionable message: "Run: docker compose up -d docling" |
| Docling timeout on one file | File skipped, batch continues |
| Ollama unreachable | Translated to actionable message, batch halted (embedding can't proceed) |
| Qdrant unreachable during pre-check | Fail-open: proceed with ingestion, log warning |
| Qdrant unreachable during upsert | Error propagated, file marked failed, batch continues |
| Redis unreachable | Cache becomes no-op, ingestion proceeds normally |
| Batch state file corrupted | Warn, delete stale file, start fresh batch |
| Content hash mismatch (file changed since pre-check) | Normal: re-ingest (correct behavior) |

---

## Section 9: Testing Strategy

### Unit tests (existing framework: pytest)

- **EmbeddingService**: Mock Ollama client, verify correct batching (1 call for 64 texts, 2 calls for 65), cache integration, fallback on failure, skip fallback when models identical
- **IngestionOrchestrator**: Mock docling_client and RAGEngine, verify concurrent dispatch, semaphore cap, timeout enforcement, checkpoint file read/write, resume logic
- **Two-phase idempotency**: Verify Phase 1 skips new files, Phase 2 detects mtime+size match, Phase 2 detects change
- **Config defaults**: Verify new default values, startup warnings for identical fallback models

### Integration tests

- End-to-end ingestion with a small PDF via batched embedding endpoint
- Batch resume: ingest 2 files, simulate crash after 1, verify resume skips completed
- Timeout: ingest a file known to hang Docling, verify skip + batch completion
- Metadata cleanup: ingest with `ENABLE_METADATA_EXTRACTION=false`, verify empty payload fields

### Test file naming

Tests follow existing patterns in `tests/unit/` and `tests/integration/` with `test_` prefix.

---

## Section 10: Migration Path

### Files to create

| File | Purpose |
|------|---------|
| `rag/embedding.py` | New `EmbeddingService` class |
| `rag/ingestion.py` | New `IngestionOrchestrator` class |

### Files to modify

| File | Changes |
|------|---------|
| `rag/config.py` | New defaults for `/api/embed`, fallback models, expansion flags, timeouts; new env vars; startup safety checks |
| `rag/pipeline.py` | `_dense_embed_batch` delegates to `EmbeddingService`; `ingest_text` writes empty metadata when extraction disabled; store `file_mtime`/`file_size` in payload |
| `memex/server.py` | `rag_ingest_batch` delegates to `IngestionOrchestrator`; markdown result display skips empty metadata fields |
| `rag/services/query_expansion.py` | `_embed_single` uses `EmbeddingService.embed()` instead of manual HTTP |
| `.env.example` | Updated to reflect new defaults |
| `docker-compose.yml` | No changes needed |
| `Dockerfile` | No changes needed |

### Rollback

All changes are additive or default-only. Existing `.env` files retain current behavior. Rolling back to previous commit restores old behavior without data migration.

### No data migration

Existing Qdrant collections work with the new code as-is. The `contextual_dense` vector in existing collections stays unused when `ENABLE_CONTEXTUAL_RETRIEVAL` is off. Stale metadata is cleaned on next re-ingestion (Section 7).

---

## Section 11: Expected Performance Improvement

| Scenario | Before | After | Improvement |
|----------|--------|-------|-------------|
| Single 64-chunk doc (embedding) | ~29 s (128 sequential calls) | ~0.5 s (1-2 batched calls) | **50x** |
| 3 docs, 5 MB total (full ingest) | ~90-180 s (sequential parse + N+1 embed) | ~30-60 s (concurrent parse + batched embed) | **3-6x** |
| Batch resume after crash | Full re-process | Skip completed, resume remaining | **Instant for completed files** |
| Re-ingestion of unchanged file | Docling parse + hash check + skip | mtime check + skip (no Docling) | **100x+ (no parse)** |
| Search (query expansion off by default) | 4 LLM calls per query | 0 LLM calls | **Instant** |
