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

## GPU-Accelerated OCR

Current OCR runs `rapidocr-onnxruntime` on **CPU** (ONNX Runtime CPU provider) — PP-OCRv6-small
models work on CPU but text-detection (DBNet) on high-res page images is the bottleneck
(~210s per 14-page PDF). Move to GPU:

### Container changes

- Base image: `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` → **`nvidia/cuda:12.x-runtime-ubuntu22.04`**
  with uv installed (nvidia-container-toolkit is already configured on the host — Marker uses it).
- `rapidocr-onnxruntime` → keep, but add **`onnxruntime-gpu`** (CUDA + cuDNN ExecutionProvider).
- Runtime provider selection: `providers=["CUDAExecutionProvider", "CPUExecutionProvider"]`
  — automatic CPU fallback if the GPU is unavailable or out of memory. No hard failure.

### docker-compose changes

- OCR service gets `deploy.resources.reservations.devices: [{capabilities: ["gpu"], device_ids: ["0"]}]`.
- VRAM budget: cap detection input side length via `OCR_LIMIT_SIDE_LEN` env (default 1280)
  → ~500-700MB VRAM footprint, fits alongside Ollama (~4GB) on the 8GB card.

### GpuLock coordination

- Extend `gpu_lock.py` to know about OCR (like Marker): before an OCR job starts, check
  VRAM headroom; if Ollama models are resident and VRAM is tight, either evict idle Ollama
  models or wait. OCR's ingest step uses Ollama embeddings, so the two overlap in the
  queue architecture — the lock prevents the same OOM class Marker had.

### Expected effect

- Detection per page: tens of seconds → ~1-2s; a 14-page scanned PDF ≈ 210s → **~30-60s**.
- Dead-man timeout stays 900s (generous on GPU).
- CPU quick win included regardless: PDF render scale 2.0 → **1.5** + side-length cap —
  halves CPU-fallback cost with negligible quality loss on deed-type documents.
- `ocr_workers` stays default 1; with GPU, raising to 2 becomes viable — config knob, not default.

## Config Changes

- `converter.ocr_max_concurrent` (2) → **`converter.ocr_workers`** (default **1**).
- `converter.ocr_timeout`: 600 → **900** (per-job dead-man cap only; queue wait excluded).
- `converter.ocr_render_scale`: 2.0 → **1.5**.
- New: `converter.ocr_limit_side_len` (**1280**) — max page-image side in px (VRAM/CPU cap).

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
- GPU: verify `CUDAExecutionProvider` active in OCR container logs (`nvidia-smi` shows the
  process); verify CPU fallback still works with the GPU device removed.
- Timing check: 14-page scanned PDF completes in <90s with GPU.

**Existing tests** must stay green: sync, cli, ocr_fallback, ocr_client, status.

## Out of Scope

- Parallel OCR workers > 1 remains configurable but not default.
- No progress percentage within a single OCR job (pages processed) — future work.
- No queue depth UI in the CLI beyond `memex status` (future work).
