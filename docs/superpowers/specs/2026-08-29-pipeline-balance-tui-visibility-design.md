# Pipeline Balance & TUI Visibility — Design

Date: 2026-08-29
Status: Approved

## Problem

During ingest/sync, the pipeline runs with a persistent stage imbalance:

- The convert pool (MarkItDown/OCR, up to 8 files in flight) finishes faster
  than the LLM phases consume, so converted files pile up in `Queued`.
- The LLM phases (context, metadata, embedding) run for only **2 files at a
  time** (`INGEST_WORKERS=2`, `LLM_WORKERS=2`) while Ollama is configured for
  4 parallel requests (`OLLAMA_NUM_PARALLEL: 4`). Half the serving capacity
  sits idle.
- The GPU lock is acquired and released **twice per file** (once for context,
  once for metadata). Between the two holds, marker can grab the GPU and
  evict Ollama models, forcing a reload (~5–10s) before the next LLM phase.
  Result observed on the remote host: files show `Converting` while GPU
  utilization reads 0% — work stalled, and the TUI gives no explanation.
- Context prefixes are generated in tiny batches (5 chunks per LLM call).
- Small files embed 1–3 texts per Ollama call instead of full 64-text
  batches, under-utilizing GPU throughput.

On top of this, the TUI hides what the LLM is doing. The MarkItDown/OCR queue
rows show converter activity, but there is no row for LLM-phase activity, so
a stalled GPU is indistinguishable from a working one.

## Goals

1. **No stage idle.** Converters and LLM phases both stay busy for the whole
   run; converted files never wait long in `Queued`.
2. **Visible LLM activity.** The TUI shows which files are in context,
   metadata, and embedding right now, plus whether Ollama has models loaded
   (GPU vs CPU).
3. **Honest stages.** When the GPU lock blocks a file, its row shows
   `Waiting GPU` instead of a stale `Converting`.
4. **TUI fits any terminal.** Row pool shrinks on small terminals; the
   Overall progress bar is always visible.
5. **8GB GPU and up.** All parallelism adapts to whatever GPU the host has —
   no hard-coded machine-specific sizing.

## Non-Goals

- No rewrite to asyncio (thread pools stay).
- No phase-parallel pipeline redesign (context/metadata/embedding keep their
  per-file order).
- No changes to query-time features (expansion, reranking).

## Design

### 1. LLM worker parallelism: 2 → 4

- `memex/engine/sources/sync.py`: `INGEST_WORKERS = 2 → 4`.
- `memex/cli.py`: `LLM_WORKERS = 2 → 4`.
- Rationale: 4 matches `OLLAMA_NUM_PARALLEL: 4`; more would queue
  server-side without extra throughput.
- The convert-side look-ahead (`CONVERT_AHEAD=8`, `MAX_AHEAD=8`) stays — the
  converter queue must never starve 4 LLM workers.

### 2. GPU lock consolidation

- `memex/engine/core/pipeline.py` `_ingest_chunks`: wrap the **whole LLM
  phase** (document summary + context enrichment + metadata extraction) in
  one `gpu_lock.acquire("llm")` / `finally: release("llm")` pair. Currently
  context and metadata each acquire/release separately.
- Embedding phase does not hold the LLM lock (Ollama serves embed + chat
  concurrently; embed batches are the payload, not a model reload).
- When acquisition waits (another owner holds the GPU), emit
  `PipelineStage.WAITING_GPU` through the progress callback for that file.

### 3. Cross-file embedding batching

- New module-level batch accumulator in
  `memex/engine/ingestion/embedding.py` (or a new `embed_batcher.py` under
  `engine/ingestion/`):
  - Workers submit `(text, source_identifier, chunk_index)` tuples.
  - A flusher thread submits when **64 texts accumulate or 300 ms elapse**,
    whichever comes first (64 = `embedding.batch_size`, 300ms bound keeps
    latency acceptable for small corpora).
  - Each submission gets a per-request future; results fan back by index.
  - Batching collapses the per-source separation: chunks from multiple small
    files share one Ollama call.
  - Failure fallback: on any batch error, retry the affected texts with the
    existing direct per-file path. The accumulator is drained on process
    shutdown (atexit or explicit `flush()` from the pipeline).

### 4. Context batch size: 5 → 10

- `config.yaml` `contextual_retrieval.batch_size: 10` (default). 10 chunks ×
  ~500 chars fits the 4k context window with room for the prompt.
- `max_batches: 8` stays as the cap.

### 5. TUI: fixed pool, terminal-aware, always-visible progress

- Row pool sizing in `memex/cli.py` `_ProgressTracker._row_pool_size()`:
  `max(12, min(20, height - 4))` — 12 covers 8 converts + 4 LLM workers;
  cap 20 prevents overflow on tall terminals; `height - 4` keeps the Overall
  bar + queue rows + pool inside any terminal.
- Fixed pool invariant stays: every row is pre-created before `Live` starts;
  only in-place updates afterward (Rich issue #1144).
- New file entering the pipeline claims a free slot immediately; terminal
  stages (Done/Skipped/Error) clear the row and recycle the slot. Unused
  slots render as empty lines (existing `unused` field).
- The **Overall progress bar row is permanent** — first row of the layout,
  never recycled, never hidden.

### 6. TUI: LLM activity row

- New `_LlmActivityDisplay` (or extension of `_QueueDisplay`) adds one row
  labeled `LLM` next to the MarkItDown/OCR queue rows.
- Data sources, merged every 0.5 s:
  - A thread-safe `ActivityRegistry` singleton that the pipeline reports
    into: `(source_identifier, phase)` on every phase transition, with
    removal on terminal events. Phases: context, metadata, embedding,
    waiting-gpu.
  - Ollama `GET /api/ps` (already reachable; no new endpoint) for loaded
    models + processor (GPU/CPU).
- Rendering:
  - Active: `◎ context: a.pdf · metadata: b.pdf · embed: c.pdf`
  - All LLM phases empty but files `Queued`: `waiting gpu` (with the number
    of queued files)
  - No pipeline activity: `· idle · qwen2.5:1.5b (GPU)` — model name +
    processor, mirroring the OCR row's idle hint, so a host running CPU-only
    Ollama is immediately visible.

### 7. WAITING_GPU stage

- Add `WAITING_GPU = "Waiting GPU"` to `PipelineStage` in
  `memex/engine/core/progress.py`.
- Emitted when the GPU lock acquisition blocks (and while marker holds it),
  replacing the stale `Converting` on that file's row.
- Non-terminal: once the lock is acquired the file proceeds to `Context`.
- The existing `gpu.max_wait_s=120` timeout still applies — after timeout
  the pipeline proceeds anyway with a warning.

## Error Handling

- GPU lock release in `finally` — exceptions never leave marker deadlocked.
- Batch accumulator failure: fall back to direct per-file embedding for the
  affected texts; log a warning; never crash the run.
- ActivityRegistry is bounded (dict keyed by source id; entries pruned on
  terminal events and when absent > 60 s).
- `/api/ps` poll failure: row degrades to `· idle` — polling must never
  block or crash the display thread.

## Testing

- **Unit**
  - Batch accumulator: fill-to-64 flush, 300 ms timeout flush, result
    ordering, error fallback, shutdown drain.
  - GPU-lock consolidation: single acquire/release per file (mock gpu_lock,
    assert call counts), release on exception.
  - Stage transitions incl. `WAITING_GPU` (non-terminal).
  - `_row_pool_size` math for heights 15/24/40/100.
  - ActivityRegistry: report/remove/prune semantics.
- **Integration**
  - Real ingest of a mixed corpus (small md files + large docx): assert no
    stage is idle for > 30 s, LLM row shows activity, embedding batches
    reach the 64-text target on the large doc.
  - TUI smoke test in a 24-line PTY: no scroll corruption, Overall bar
    visible for the whole run, done rows clear immediately.
- **Regression**: existing 705 unit tests stay green.

## Files Touched

- `memex/engine/sources/sync.py` — `INGEST_WORKERS` 2 → 4
- `memex/cli.py` — `LLM_WORKERS` 2 → 4; pool sizing; LLM activity row;
  WAITING_GPU rendering
- `memex/engine/core/pipeline.py` — consolidated GPU lock; WAITING_GPU
  emission; ActivityRegistry reporting
- `memex/engine/core/progress.py` — `PipelineStage.WAITING_GPU`
- `memex/engine/ingestion/embedding.py` (or new `embed_batcher.py`) —
  cross-file batch accumulator
- `config.yaml` / `config.example.yaml` — `contextual_retrieval.batch_size: 10`
