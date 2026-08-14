# File Status State Machine + Retry Orchestration

**Date**: 2026-08-14
**Status**: Draft
**Supersedes**: Status/retry portions of `2026-08-14-production-reliability-design.md`

## Problem Statement

### Flaws found in the current pipeline

1. **F1 — Critical bug: StatusTracker writes to the wrong payload key.**
   `status_tracker.py:63` scrolls Qdrant by `key="source_id"`, but chunks are stored
   with payload key `"source"` (`pipeline.py:681`). `update_status()` silently no-ops —
   the entire status system never records anything.

2. **F2 — Three parallel status systems that don't share a model.**
   `StatusTracker` (Qdrant chunk payload, sync path only), `FileProgress`/`ProgressCallback`
   (in-memory, ephemeral CLI live view), and `IngestionOrchestrator` batch JSON checkpoints
   (`~/.memex/batches/*.json`, completed/failed only). No unified view.

3. **F3 — No per-file state machine.**
   `pipeline.py:_ingest_chunks` emits `(msg, pct)` strings; `ingestion.py:176` uses
   fragile `status.startswith("Success")` prefix matching. No valid-transition enforcement,
   no resume-from-interrupted-state.

4. **F4 — `rag_processing_status` MCP tool is shallow.**
   Returns only aggregate counts (wrong, due to F1). Docstring claims "retry information
   for files with errors" but no per-file detail, stages, or hints exist.

5. **F5 — Retry orchestration is dead code.**
   `RetryQueue` (`sources/retry_queue.py`) + `DoclingAsyncClient` are never wired into any
   path. `FileStatus.RETRY` is never set; `get_pending_retries()` is never called.

6. **F6 — Failures before chunk-storage are lost.**
   `update_status()` finds an existing point then `set_payload`s it. If conversion fails,
   no chunks exist → no point → the failure is invisible.

7. **F7 — Error strings instead of typed errors.**
   `ingestion.py` returns `f"Failed: {exc}"`; CLI uses `except Exception` catch-alls;
   `server.py:_friendly_error` string-matches messages (`"cannot reach docling" in msg`).

8. **F8 — CLI `ingest` has no stage granularity.**
   Only Parsing → Ingesting → Done/Error; ignores the `PipelineStage` enum.

### Research basis

Production RAG ingestion pipelines model processing as a **resumable state machine with
persisted per-file checkpoints** (Nil Monfort, "The 5-Stage RAG Ingestion Pipeline with
Checkpoint Resume", 2026-02). Key lessons applied here:

- Explicit `VALID_TRANSITIONS` — no illegal or accidental transitions.
- Progress model persisted per document; interrupted processing resumes, not restarts.
- `FAILED → retryable` resets to the start of the pipeline preserving progress data.
- Terminal states (`DONE`) are not re-processed accidentally; re-ingest requires an
  explicit change signal.
- Zombie `processing` states (crashed server) must be reconciled.

## Goal

One authoritative, queryable per-file status record that works across **all** ingest paths
(sync, CLI ingest, MCP ingest), driven by a typed state machine, with working retry
orchestration, surfaced through colored CLI output and a detailed MCP status tool — and
every failure typed with an actionable hint.

## Architecture

### New module: `memex/engine/ingestion/status.py`

`FileStatusStore` — the single source of truth.

```
Qdrant collection: memex_file_status          (auto-created lazily)
Point per file, deterministic point ID = sha1(source) → UUID

Payload fields:
  source, source_name,
  status,            # coarse lifecycle: pending|processing|done|skipped|failed|retry
  stage,             # fine position: converting|chunking|context|metadata|embedding|storing|deleting
  chunks, attempts, next_retry_at,
  error, error_type, hint,
  created_at, updated_at, completed_at
```

Two-axis model:
- `status` = coarse lifecycle state machine (drives retry/resume).
- `stage` = fine live position shown in UI, updated within `processing` via self-loop.

### Status model & state machine

`IngestionStatus` enum (coarse lifecycle):

```
PENDING → PROCESSING → DONE
              │   └────→ SKIPPED
              └───────→ FAILED
FAILED → RETRY (scheduled) → PROCESSING   # after backoff window
DONE/SKIPPED → PROCESSING                 # re-ingest after source change
```

Explicit `VALID_TRANSITIONS`:

```python
VALID_TRANSITIONS = {
    PENDING:    {PROCESSING, SKIPPED, FAILED},
    PROCESSING: {PROCESSING, DONE, SKIPPED, FAILED},   # PROCESSING self-loop = stage update
    FAILED:     {RETRY, PROCESSING},                    # manual retry or scheduled
    RETRY:      {PROCESSING, FAILED},                   # retry attempt or backoff exhausted
    DONE:       {PROCESSING},                           # re-ingest on change
    SKIPPED:    {PROCESSING},                           # re-ingest on change
}
```

Behavior:
- `store.transition(source, to_status, stage=..., ...)` validates against the map.
  Illegal transitions raise `StorageError` (never a silent no-op).
- Any non-terminal state → `FAILED` is allowed from anywhere (covers crash/interrupt).
- `FAILED → RETRY` sets `next_retry_at = now + backoff`, increments `attempts`.
- Stage updates are `status=PROCESSING` self-loops carrying a new `stage` value.

`PipelineStage` enum in `progress.py` remains the fine `stage` axis.

### Integration points & data flow

Single uniform update call site — all writers go through `FileStatusStore`:

```python
store.mark_pending(source, source_name)          # before work starts
store.update_stage(source, stage, ...)           # during work (PROCESSING self-loop)
store.mark_done(source, chunks=...)              # success
store.mark_skipped(source, reason=...)           # dedup/unchanged
store.mark_failed(source, error, exc, stage)     # from ANY stage
store.mark_deleted(source)                       # chunks removed by sync
store.schedule_retry(source, error, attempts)    # → RETRY + next_retry_at
store.get_due_retries(now)                       # status=RETRY AND next_retry_at <= now
store.cleanup_stale(stale_after=7d)              # zombie PROCESSING → FAILED
```

| Call site | Method |
|-----------|--------|
| `pipeline.py:_ingest_chunks._progress(msg, pct)` | `update_stage` (pct→stage mapping) |
| `sync.py:_emit()` | `update_stage` / `mark_done` / `mark_failed` / `mark_skipped` |
| `sync.py` reconciliation | `mark_pending` / `mark_deleted` |
| `ingestion.py` orchestrator (per item) | `mark_pending` → `update_stage` → `mark_done/failed` |
| `cli.py ingest` loop | same as orchestrator |
| `mcp/server.py` ingest tools | same |

MCP status tool data flow:

```
rag_processing_status
  → store.get_summary()      # counts by status (one cheap query)
  → store.list_records()     # per-file: source, status, stage, chunks, attempts,
                             #          next_retry_at, error, error_type, hint, updated_at
  → JSON output (keeps existing aggregate fields for backward compat)
```

Qdrant access: the store opens its own lazy client and ensures the collection exists
before first write — same pattern as the main collection.

Thread-safety: writes are atomic single-point upserts (idempotent by deterministic point
ID), safe under sync's thread pool and the orchestrator's asyncio tasks.

### Retry orchestration

```
File fails → store.mark_failed(source, error, exc)
           → RetryQueue.should_retry(error, attempts)
                ├─ yes → store.schedule_retry(source, error, attempts+1)   # RETRY + next_retry_at
                └─ no  → stays FAILED (terminal for this run)
```

- `should_retry` recognizes typed `MemexError` subclasses in addition to the substring
  list: `ConversionTimeoutError`, `ServiceUnavailableError`, `RetrievalError`,
  `EmbeddingError` are retryable; `ConfigError`, `CorruptedDocumentError` are not.
- Backoff uses existing `BACKOFF_SCHEDULE = [60s, 5m, 30m, 2h]`, capped at `MAX_RETRIES = 4`.
- `RetryQueue.process_retries()` is rewritten to re-drive through the real pipeline
  (parse → ingest) rather than submitting a bogus empty payload to Docling.
- Sync automatically picks up due retries each run (`get_due_retries(now)` → re-enter
  reconciliation as pending → PROCESSING).
- `store.cleanup_stale(stale_after=7d)` marks long-unfinished `PROCESSING` records as
  `FAILED` with `hint="process likely died"`.

### New commands / tools

| Surface | Name | Behavior |
|---------|------|----------|
| MCP tool | `rag_retry_failed` | Reset `FAILED` files → `PROCESSING` immediately (bypass backoff). Optional `source_filter`. |
| MCP tool | `rag_processing_status` | Gains optional `filter` (status value); returns per-file detail + aggregate counts |
| CLI | `memex retry [--source-name] [--all]` | Retry failed files now |
| CLI | `memex status [--status foo] [--limit N]` | Colored per-file status table (uses `_STAGE_STYLE` colors) |

### Error handling — everything typed

| Layer | Handling |
|-------|----------|
| `FileStatusStore` | Invalid transitions → `StorageError` (never silent). Qdrant down → `ServiceUnavailableError` with hint |
| `pipeline.py` | Stage-level failures wrapped: `IngestionError(source, detail, stage=...)`; partial stages (e.g. metadata) degrade gracefully with a warning |
| `ingestion.py` | Replaces `f"Failed: {exc}"` strings → catches typed `MemexError`, records `mark_failed(source, error, exc, stage)`, returns structured summary with `error_type` + `hint` |
| `sync.py` | `_log_file_error` writes typed context: `source`, `stage`, `error_type`, `hint` via logging_setup extras |
| `mcp/server.py:_friendly_error` | Rewritten to dispatch on exception type (`ConversionTimeoutError`, `ServiceUnavailableError`, ...) → returns `hint` directly; falls back to `error_context(exc)` — no string-matching |
| CLI `ingest`/`sync`/`retry` | Maps `MemexError` to colored `[red]Error: msg (hint)[/red]`; non-Memex bugs logged with `exc_info` |

Every failed file record always has: `error`, `error_type`, `stage` where it failed,
`hint` when available.

## Files to Create / Modify

**New:**
- `memex/engine/ingestion/status.py` — `FileStatusStore` + `IngestionStatus` + `VALID_TRANSITIONS`
- `tests/unit/test_status.py`
- `tests/unit/test_retry_queue.py`

**Modify:**
- `memex/engine/sources/retry_queue.py` — typed `should_retry`, real `process_retries`
- `memex/engine/sources/status_tracker.py` — superseded; kept as thin shim or removed
- `memex/engine/core/pipeline.py` — `_progress` hook writes status
- `memex/engine/sources/sync.py` — `_emit` writes status; due-retry fold-in
- `memex/engine/ingestion/ingestion.py` — typed errors + status marks
- `memex/cli.py` — `ingest` stages; `status` + `retry` commands
- `memex/mcp/server.py` — `rag_processing_status` detail + filter; `rag_retry_failed`; `_friendly_error` dispatch
- `tests/unit/test_sync.py`, `tests/unit/test_server.py`, `tests/unit/test_cli.py`, `tests/unit/test_pipeline_chunking.py`

## Testing

| Test file | Covers |
|-----------|--------|
| `tests/unit/test_status.py` *(new)* | Valid + invalid transitions raise `StorageError`; stage self-loops; retry scheduling sets `next_retry_at` + increments attempts; cleanup_stale marks zombies FAILED; idempotent upserts with fake Qdrant |
| `tests/unit/test_retry_queue.py` *(new)* | `should_retry` on typed errors (timeout→true, config→false) + backoff schedule |
| `tests/unit/test_sync.py` | failed files → `RETRY` → picked up next run; `_log_file_error` extras |
| `tests/unit/test_server.py` | `_friendly_error` dispatches on types; `rag_retry_failed` tool; `rag_processing_status` returns per-file detail + filter |
| `tests/unit/test_cli.py` | `memex status` / `memex retry` commands |
| `tests/unit/test_pipeline_chunking.py` | `_progress` hook writes status; stage-map correct |

Validation: `make lint`, `make typecheck`, `make test` must stay green (currently 634 passing).

## Success Criteria

1. `memex status` shows accurate per-file stages/status with colors.
2. `rag_processing_status` returns per-file detail + aggregate counts (no silent no-ops).
3. Failed files get a `RETRY` status with backoff; due retries are picked up by the next sync.
4. A file failing at conversion (before chunking) still has a visible FAILED record (F6 fixed).
5. `_friendly_error` returns `hint` for typed errors without string-matching.
6. All tests pass; lint + typecheck clean.

## Out of Scope

- Async Docling client migration (covered by `2026-08-14-production-reliability-design.md`).
- Per-sub-stage counters (pages_extracted, chunks_embedded) — future enhancement.
- Multi-node status aggregation.
