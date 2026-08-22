# OCR Queue Architecture — Design Spec

**Date:** 2026-08-22
**Status:** Approved (user-confirmed all sections)
**Scope:** Memex RAG — OCR fallback re-architected as a producer-consumer queue

## Problem Statement

The OCR fallback currently fails in production for architectural reasons:

1. **Timeout as flow control.** Each OCR HTTP request carries `ocr_timeout` (600s) that counts
   server-side queue wait. Under CPU contention, two concurrent OCR jobs each take ~596s
   (measured), so the second job sits at the timeout edge — any variance causes
   `ReadTimeout` → file fails.
2. **Parallel OCR is slower than serial on this hardware.** Measured: 1 job ≈ 210s;
   2 concurrent jobs ≈ 596s each. RapidOCR is CPU-bound; concurrency causes thread
   oversubscription, not throughput.
3. **No real queue.** The current "OCR lane" is an implicit asyncio semaphore gate, not a
   queue. Queued files wait inside HTTP requests, invisible and timeout-prone.
4. **Wrong fallback semantics.** MarkItDown service outage routes *all* files (including
   digital PDFs) to OCR. OCR must be used **only for genuinely scanned/unreadable files**.

## Goals

- MarkItDown conversion lane never idles: while OCR works on a scanned file, MarkItDown
  keeps converting subsequent files.
- Queue wait never hits a timeout. Timeouts apply only as a per-job dead-man switch.
- Queued files survive process interruption and resume on next start (via status store).
- `rag_ingest_file` (MCP) returns immediately for scanned files; status is polled.
- OCR triggers only for scanned/poor-quality output — never for MarkItDown outages.

## Non-Goals

- No Redis-backed durable queue (status store is the recovery mechanism).
- No job-based OCR server (queue lives client-side).
- No multi-process queue coordination (single-user local box).

## Architecture

### New component: `OcrQueue` (`memex/engine/ingestion/ocr_queue.py`)

```
class OcrJob:                     # dataclass
    source_identifier: str        # logical path (Qdrant source / S3 key)
    local_path: str               # where bytes live for OCR
    source_name: str
    file_idx: int                 # progress context
    total_files: int

class OcrQueue:
    def __init__(self, engine, status_store, progress_cb, workers: int = 1)
    async def enqueue(job)        # non-blocking, producer never waits
    async def drain()             # await until queue empty AND all jobs finished
    async def stop()              # cancel workers; ocr_queued flag guarantees recovery
```

- `asyncio.Queue[OcrJob]`, FIFO discovery order.
- N consumer workers (config `converter.ocr_workers`, **default 1**).
- Consumer per job:
  1. `status_store.update_stage(source, PipelineStage.OCR)` (+`ocr_queued: false` once started)
  2. `convert_with_ocr(bytes(local_path))` — dead-man timeout only (default 900s)
  3. `chunk_markdown_aware(...)` chunking
  4. ingest via engine (reuse sync's `_ingest_markdown`)
  5. `mark_done(chunks)` or `mark_failed` + auto-retry queue
  6. progress events through `progress_cb` so CLI rows animate `◎ OCR`
- Any per-job exception is caught; the worker survives and continues with the next job.

### Status store

- Files entering the queue: `status=processing`, `stage=OCR`, payload
  `ocr_queued: true`, `local_path`, `source_name`.
- Recovery query: `list_records` filter `ocr_queued=true` (or `stage=OCR` with
  `status=processing`) → re-enqueue on next start.
- `memex status` shows queued files with `◎ OCR` stage.

### Producers

| Producer | Behavior |
|----------|----------|
| `sync` | On `needs_ocr`: mark `ocr_queued`, `enqueue`, continue with next file. SyncStats for that file complete only when the consumer finishes (completion callback). Sync ends with `await queue.drain()`. |
| CLI `ingest` | Same queue for the run; enqueue scanned files; `drain()` before exit. |
| MCP server | One long-lived `OcrQueue` started at server startup. `rag_ingest_file` returns `"OCR queued — check rag_processing_status"` immediately. |

### OCR trigger rules (scanned-only semantics)

| MarkItDown outcome | Action |
|--------------------|--------|
| Success, good text | Ingest directly. No OCR. |
| Success, poor quality (empty / tiny text vs file size) | → OCR queue |
| Per-file error (400 / empty parse / `CorruptedDocumentError`) | → OCR queue |
| **Service outage (`ServiceUnavailableError`)** | **No OCR.** Mark failed + auto-retry queue; wait for MarkItDown to return. |

This reverses the previous `ServiceUnavailableError → OCR` routing added in commit 1954dc1.

## Data Flow

```
Sync run (66 files, 4 MarkItDown workers):

for each file:
    download → MarkItDown convert
        ├─ text OK      → ingest → Done         (worker takes next file immediately)
        └─ scanned      → status: ◎ OCR queued
                          queue.enqueue(job)     (no waiting)

OCR consumer (1 worker, FIFO):
    queued → ◎ OCR → convert_with_ocr (dead-man 900s)
    → chunk → ingest → Done / Error (+ auto-retry)

end of run: await queue.drain() → stats include OCR completions
```

MCP flow:

```
rag_ingest_file(scanned.pdf)
  → MarkItDown poor quality → enqueue → return "OCR queued"
  → background consumer OCRs + ingests → rag_processing_status shows Done
```

Recovery flow (crash / Ctrl+C mid-queue):

```
next start: query status store for ocr_queued=true
  → re-read local_path (or re-download from source)
  → re-enqueue → resume
```

## Config Changes

- `converter.ocr_max_concurrent` (2) → **`converter.ocr_workers`** (default **1**).
- `converter.ocr_timeout`: 600 → **900** (per-job dead-man cap only; queue wait excluded).

## Error Handling

- OCR service down / returns no text → `mark_failed` + auto-retry (backoff 300s, max 5 attempts).
- Dead-man timeout → `mark_failed` + auto-retry; worker continues.
- Consumer exception → logged; job failed + retry; worker survives.
- Re-enqueued job whose local file vanished → re-download from source; unresolvable → failed + retry.
- `stop()` on shutdown → current job finishes or is abandoned; `ocr_queued` flag guarantees pickup.

## Testing

**Unit (no Docker):**
- `test_ocr_queue.py`: FIFO drain order; error isolation (bad job doesn't stop worker);
  dead-man timeout → failed+retry; `stop()` semantics; drain waits for in-flight jobs.
- Status store: `ocr_queued` set/cleared; recovery query returns queued files.
- Sync: `needs_ocr` → enqueued not awaited; stats complete only after drain.
- Loader: trigger rules — poor quality → OCR; per-file error → OCR; service outage → **no OCR** (raised for retry). Update the outage test added in 1954dc1 to expect retry instead of needs_ocr.

**Integration (Docker up):**
- Scanned PDF via `memex ingest` → `◎ OCR queued` → Done with chunks.
- MCP `rag_ingest_file` returns immediately; `rag_processing_status` flips to Done.
- Crash recovery: kill sync mid-queue → restart → queued files resume.

**Existing tests** must stay green: sync, cli, ocr_fallback, ocr_client, status.

## Out of Scope

- Parallel OCR workers > 1 remains configurable but not default.
- No progress percentage within a single OCR job (pages processed) — future work.
- No queue depth UI in the CLI beyond `memex status` (future work).
