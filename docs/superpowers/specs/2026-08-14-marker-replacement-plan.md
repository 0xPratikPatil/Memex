# Replace Docling with Marker — Implementation Plan

**Date**: 2026-08-14
**Status**: Implemented
**Supersedes**: Docling conversion in `docker-compose.yml`, `loader.py`, `splitter.py`

## Why

Docling consistently failed in production: 504 timeouts, container disconnects,
CPU-bound OCR, and structure-aware chunking that could not survive large legal
PDFs. Research (Marker's olmocr-bench, 1,403 PDFs) shows Marker beats Docling
on **both quality (76.0 vs 50.3) and throughput (2.9 vs 2.1 pg/s GPU)**; `fast`
mode is 7.4 pg/s (3.5x Docling). It runs in Docker (vllm backend), outputs
Markdown/JSON/HTML/**chunks** for RAG, has optional LLM boost, and handles
scanned PDFs via Surya OCR.

## Target architecture

```
MCP server (host)
  └── HTTP ──► Docker: marker (:5001)  [NEW — replaces docling]
                 marker-pdf + marker_server (FastAPI, /marker/upload)
                 GPU via NVIDIA toolkit, models preloaded at startup
  └── chunking: local recursive/fixed (structure-aware hybrid chunking
                was Docling-specific; recursive is the RAG default anyway)
```

Marker replaces Docling **as the converter only**. Chunking strategy switches
from `hybrid` to `recursive` (the codebase already supports it — `create_chunks`
falls back to it today).

## Official Marker server API (from marker/scripts/server.py, v1.x)

- `POST /marker/upload` — multipart form:
  - `file` (required), `output_format` (`markdown`|`json`|`html`|`chunks`),
    `force_ocr` (bool), `mode` (`balanced`|`fast`), `page_range`, `paginate_output`
- Response (HTTP 200 always):
  ```json
  {"format": "markdown", "output": "<markdown>", "images": {}, "metadata": {}, "success": true}
  {"success": false, "error": "<message>"}   // conversion failure — MUST check "success" field
  ```
- `GET /` and `/docs` (OpenAPI). Health = `GET /health` (add to our wrapper).

## Work packages

### WP1 — Docker: `marker` service (replaces `docling` in docker-compose.yml)

- **Image**: build from source via the existing multi-stage Dockerfile pattern
  (uv, pytorch/pytorch CUDA base, non-root user, GPU via NVIDIA toolkit).
- **New `marker.Dockerfile`** in repo root (mirrors existing Dockerfile style):
  - Stage: `ghcr.io/astral-sh/uv` tool → `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime`
  - `pip install marker-pdf` + `uvicorn[standard] fastapi python-multipart`
  - Preload models at build time (`marker.models.load_all_models()` smoke) to
    avoid cold-start conversion failures.
  - Entry: `python -m uvicorn marker.scripts.server:app --host 0.0.0.0 --port 5001`
  - Add a tiny `GET /health` wrapper (FastAPI route added via an app module
    that imports marker's app and adds the route).
- **compose service** `marker` (name `memex-marker`, port `127.0.0.1:5001:5001`):
  - env: `TORCH_DEVICE=cuda`, `SURYA_INFERENCE_BACKEND=vllm`, `NVIDIA_VISIBLE_DEVICES=all`
  - deploy: cpus 4.0, memory 6G, nvidia GPU reservation (same as docling had)
  - healthcheck `curl -sf http://localhost:5001/health`, tmpfs /tmp, json-file logs,
    restart unless-stopped, labels `com.memex.service=marker`
- **Remove** the `docling` service entirely (no stale code).

### WP2 — Config keys (config.py + config.example.yaml + config.yaml)

- `converter.engine` → `marker` (values: `marker` | `docling` legacy)
- `converter.marker_url` → `http://localhost:5001`
- `converter.marker_mode` → `fast` | `balanced` (default `fast` — legal corpus is
  mostly digital; balanced reserved for scanned-heavy sets)
- `converter.marker_force_ocr` → false (auto-detect)
- `converter.marker_timeout` → 300.0
- `chunking.strategy` → `recursive` (was `hybrid`)
- Keep `DOCLING_*` keys only as legacy (removed from config.example.yaml docs
  but harmless if present).

### WP3 — `memex/engine/ingestion/marker_client.py` (NEW)

Async-capable client, mirrors loader.py's production patterns:

- `convert_markdown(file_bytes, filename) -> ConversionResult` — POST multipart
  to `/marker/upload`, checks `success` field (marker returns errors as HTTP 200
  JSON — the #1 error-handling gotcha).
- `is_marker_available() -> bool` — GET /health with 5s timeout.
- Layered tenacity retries (reuse the dynamic config-read pattern from the
  docling client):
  - transport layer: `HTTP_TRANSPORT_MAX_RETRIES` / `HTTP_TRANSPORT_RETRY_BACKOFF`
    (server restart window)
  - status layer: `HTTP_MAX_RETRIES` for 429/502/503/504
- Typed errors: `ConversionError`, `ConversionTimeoutError`, `ServiceUnavailableError`,
  `CorruptedDocumentError` (empty output / success=false).
- Concurrency cap: reuse `converter.docling_max_concurrent` value via a shared
  module-level semaphore (rename semantics: `converter.marker_max_concurrent`,
  keep the key name generic: `converter.max_concurrent`). New key
  `converter.max_concurrent` (default 2) replaces `docling_max_concurrent`.
- Uses `httpx.AsyncClient`? — No: existing loader/splitter are sync (thread
  pool). Marker client is sync `httpx.Client` (like loader.py) so sync/CLI/MCP
  paths all work unchanged. Async orchestration already happens at the
  `asyncio.to_thread` level.

### WP4 — Wire into `loader.py` + `splitter.py` + `pipeline.py`

- `loader.py`: `parse_file/parse_url/parse_local_file` route through
  `marker_client.convert_markdown()` when `converter.engine == "marker"`;
  Docling path kept behind `engine == "docling"` for rollback.
- `splitter.py`: `chunk_file` returns `{"chunks": [], "markdown": markdown}`
  when engine is marker (chunking happens locally). Simplest: `chunk_file`
  calls the marker client and returns no chunks + full markdown; `_ingest_file`
  in sync.py already handles `chunks=None` → local `create_chunks()`.
- `pipeline.py` `create_chunks()`: `strategy == "hybrid"` now delegates to the
  **local recursive chunker** (already the fallback path) — the Docling hybrid
  chunker call is removed. `chunking.strategy=recursive` is the config default.
- `get_chunker_status()` / `is_hybrid_chunker_available()`: report
  `active_chunker: recursive|fixed` based on config; marker availability shown
  as the converter, not a chunker.

### WP5 — CLI + MCP + sync stages (NO regression)

- `memex status`: `FileStatusStore` unchanged — still shows per-file stages.
- `sync` command live display: unchanged — stages flow through `_on_progress`
  already (fixed earlier). Marker conversion is slower per file than docling
  chunking was; stage `Converting` remains the same.
- MCP `rag_ingest_file/url/batch`, `rag_sync`, `rag_processing_status`,
  `rag_retry_failed`: no interface changes. Errors now typed from marker client.
- `_friendly_error` already dispatches on typed errors (ConversionTimeoutError
  etc.) — marker client raises the same hierarchy, no change needed.

### WP6 — Tests

- `tests/unit/test_marker_client.py` (NEW): success response parse; `success:false`
  → `ConversionError`; empty output → `CorruptedDocumentError`; transport retry
  recovers; health check true/false.
- `tests/unit/test_loader.py` additions: engine=marker routes to marker client;
  engine=docling keeps legacy path.
- Update `tests/unit/test_chunking.py`: chunk_file with engine=marker returns
  empty chunks + markdown; `create_chunks` uses recursive when strategy=hybrid.
- `tests/unit/test_sync.py`, `test_cli.py`, `test_server.py`: assert no
  docling-specific imports remain; stage display unchanged.
- `make lint`, `make typecheck`, `make test` all green.

### WP7 — E2E verification

- `docker compose up -d marker` → healthy.
- `memex ingest /tmp/test.pdf` → success with stage display, dedup on re-ingest.
- `memex sync --dry-run` → per-file stage display, no traceback spam.
- `memex status` → records with statuses.
- Kill marker container mid-sync → files wait through restart (transport retry),
  no RemoteProtocolError storm.

## Rollback

`converter.engine: docling` + `docker compose up -d docling` restores the old
path. All Docling client code is preserved behind the engine flag.

## Out of scope

- Async/streaming marker server (Celery/Redis distributed) — sync server +
  thread pool is sufficient at this scale; can be added later.
- LLM-boost mode (`--use_llm`) — optional, adds API cost; not enabled.
- Visual/ColPali retrieval — separate feature.
