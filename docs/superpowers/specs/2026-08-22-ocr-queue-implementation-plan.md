# Implementation Plan: OCR Queue Architecture + Multi-Model GPU Backends

**Spec:** `docs/superpowers/specs/2026-08-22-ocr-queue-architecture-design.md`
**Estimated effort:** 5 phases, ~4-6 hours total

---

## Phase 1: OCR Server — Multi-Model Backends + GPU (do first, everything depends on it)

### Step 1.1: Backend registry in `ocr_server.py`
- Define a `Backend` protocol: `load() -> None`, `ocr_pil_image(pil) -> dict` (text/confidence/lines), `unload() -> None`, `vram_mb` property
- `RapidOcrBackend`: `RapidOCR()` with `providers=["CUDAExecutionProvider", "CPUExecutionProvider"]` when `onnxruntime-gpu` is importable, else CPU; log the active provider at load
- `GraniteDoclingBackend`: transformers VLM (granite-docling-258m) in page-image → text mode; try `device_map="cuda:0"`, fall back `"cpu"`
- `LightOnOcrBackend`: `lightonocr`/transformers 2-1B, fp16 on CUDA, fp32 CPU fallback
- `ACTIVE_MODEL` env `OCR_MODEL` ∈ {pp-ocrv6-small, granite-docling-258m, lightonocr-2-1b}; one loaded at a time; no cascade

### Step 1.2: Endpoints
- `/health` → `{status, model, provider, loaded, vram_mb}`
- `/model/swap` accepts all three names: `unload()` current (free VRAM) → `load()` new; 400 on unknown/load-failure
- `/convert` unchanged contract; PDF render scale 1.5 + `OCR_LIMIT_SIDE_LEN` (1280) cap applied in `_pdf_to_pil_pages`
- Idle unload: track last-request time; background task unloads active model after `OCR_IDLE_UNLOAD_S` (300) — skips pp-ocrv6-small (resident ≤700MB)

### Step 1.3: `ocr.Dockerfile` + compose
- Base → `nvidia/cuda:12.x-runtime-ubuntu22.04` + uv; install `onnxruntime-gpu`, `transformers`, `accelerate`, `torch`, `lightonocr`, `pypdfium2`, `rapidocr-onnxruntime`
- `docker-compose.yml`: ocr service GPU reservation (`capabilities: ["gpu"], device_ids: ["0"]`), named volume `ocr_models` mounted at `/root/.cache/huggingface`, env `HF_HOME`, `OCR_MODEL`, `OCR_IDLE_UNLOAD_S`

### Step 1.4: Verify (Docker up)
```bash
docker compose build ocr && docker compose up -d ocr
curl -s localhost:5004/health   # → model + provider (CUDA for pp-ocrv6-small)
curl -s -X POST localhost:5004/convert -F "files=@docs/deeds/JET-Trust-deed.pdf"  # text, <90s
curl -s -X POST localhost:5004/model/swap -H 'Content-Type: application/json' -d '{"model":"granite-docling-258m"}'
# health shows granite-docling-258m + provider; swap back
```
- `nvidia-smi` shows the OCR process during a convert; removing GPU device env → CPU provider still works

---

## Phase 2: `OcrQueue` Component + Status-Store Recovery

### Step 2.1: `memex/engine/ingestion/ocr_queue.py` (new)
- `OcrJob` dataclass: `source_identifier`, `local_path`, `source_name`, `file_idx`, `total_files`
- `OcrQueue(engine, status_store, progress_cb, workers=1)`: `asyncio.Queue`, N consumer tasks
- Consumer loop (two phases per job): update stage `◎ OCR` + `ocr_queued: false` → `convert_with_ocr` (dead-man `asyncio.wait_for` with `OCR_TIMEOUT`) → `chunk_markdown_aware` → reuse `_ingest_markdown` from `sync.py` (move it to a shared module, e.g. keep import from sync to avoid churn) → `mark_done` / `mark_failed` + `schedule_retry`
- `enqueue()` non-blocking; `drain()` awaits queue-empty + all consumers idle; `stop()` cancels workers cleanly
- Per-job try/except: worker survives; failed job → auto-retry

### Step 2.2: Status store (`status.py`)
- `mark_ocr_queued(source, local_path, source_name)`: status=processing, stage=OCR, payload `ocr_queued: true`, `local_path`
- `get_ocr_queued() -> list[record]`: recovery query (status=processing AND stage=OCR AND ocr_queued=true)
- `ocr_queued` cleared when the job starts (payload update in `update_stage`)

### Step 2.3: Recovery helper (`ocr_queue.py`)
- `recover_queued(queue, status_store)`: for each record → re-read `local_path`; missing → re-download via source (sync only) or mark failed+retry; else re-enqueue

### Step 2.4: Unit tests
- `tests/unit/test_ocr_queue.py`: FIFO drain; error isolation; dead-man timeout → failed+retry; `stop()`; `drain()` waits in-flight; recovery re-enqueues and handles vanished paths

---

## Phase 3: Producer Integration (sync → CLI → MCP)

### Step 3.1: `sync.py`
- `_convert_and_ingest`: on `needs_ocr` → `status_store.mark_ocr_queued(...)` + `queue.enqueue(job)` + emit `◎ OCR` + **return immediately**; stats/completion callback fires from the consumer
- Create one `OcrQueue` per sync run; after `asyncio.gather` of producers → `await queue.drain()` → then stats
- `_ingest_markdown` stays reusable by the consumer

### Step 3.2: `cli.py` ingest
- Create `OcrQueue` for the run; scanned files enqueue (row shows `◎ OCR`); `drain()` before the summary panel; overall counter completes when consumers finish

### Step 3.3: MCP `server.py`
- Start a long-lived `OcrQueue` at server startup (module-level, lazy)
- `rag_ingest_file`: poor quality → `mark_ocr_queued` + enqueue → return `"OCR queued — poll rag_processing_status"`
- `rag_processing_status` already reads the status store — now shows `◎ OCR` stage

### Step 3.4: Tests
- Sync: `needs_ocr` → enqueued not awaited; stats complete only after drain (extend `test_sync.py`)
- CLI: scanned file shows `◎ OCR` row then Done (extend `test_cli.py`)

---

## Phase 4: GpuLock, Timeout Semantics, Trigger Rules

### Step 4.1: `gpu_lock.py`
- Add `"ocr"` owner with VLM footprint (~4.2GB for lightonocr-2-1b; 0 for pp-ocrv6-small/granite)
- Consumer acquires before OCR phase when model is a VLM, releases before ingest phase (two-phase loop per spec); pp-ocrv6-small/granite skip the lock

### Step 4.2: `ocr_client.py` timeout
- `OCR_TIMEOUT` semantics = per-job dead-man only (900s default); keep ConnectError-only retry; queue wait lives client-side so it never consumes the timeout

### Step 4.3: Loader trigger rules (`loader.py`)
- Revert the `ServiceUnavailableError → OCR` routing (commit 1954dc1): MarkItDown outage now raises → file marked failed + auto-retry queue (no OCR for digital files)
- Keep: poor-quality → `needs_ocr`; per-file `ConversionError`/`CorruptedDocumentError` → `needs_ocr`
- Update `tests/unit/test_ocr_fallback.py::test_markitdown_outage_routes_to_ocr` → expects raise/retry, not needs_ocr

### Step 4.4: Config (`config.py`, `config.yaml`)
- `converter.ocr_workers` (1) replaces `ocr_max_concurrent`; `ocr_timeout` 900; `ocr_render_scale` 1.5; `ocr_limit_side_len` 1280; `ocr_model` default pp-ocrv6-small

### Step 4.5: Tests
- gpu_lock: OCR owner footprint + eviction flow (extend existing lock tests)
- config defaults; loader trigger rules

---

## Phase 5: End-to-End Verification

- `docker compose build ocr markitdown && docker compose up -d`
- Sync 66-file source: digital files flow through MarkItDown lane; scanned files show `◎ OCR`, queue FIFO, no timeout failures; summary matches status store
- Kill sync mid-queue → rerun sync → `ocr_queued` files resume
- MCP: `rag_ingest_file` on a scanned PDF returns "OCR queued" immediately; `rag_processing_status` flips to Done
- Swap models via config: pp-ocrv6-small → granite-docling-258m → lightonocr-2-1b, one scanned page each, correct provider per /health
- Full suite: `uv run pytest tests/unit/ --ignore=tests/unit/test_markitdown_server.py --ignore=tests/unit/test_ocr_server.py` + ruff + mypy

## Final Verification

- [ ] All spec acceptance criteria met (queue never blocks MarkItDown lane; queue wait has no timeout; recovery works; MCP returns immediately; three models each work GPU/CPU)
- [ ] Unit + integration tests green
- [ ] ruff / mypy clean
- [ ] Spec doc committed alongside implementation
