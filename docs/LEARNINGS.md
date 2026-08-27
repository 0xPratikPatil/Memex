# Learnings — Memex RAG Project

Hard-won lessons from building this RAG pipeline. Read before touching the
CLI display, the sync/ingest pipeline, or anything threaded.

## 1. Rich / TUI live displays

The single biggest debugging saga in this project was duplicated/appended
rows in the progress display. Three independent root causes, all subtle:

- **Never `add_task`/`remove_task` while a `Live` is running.** Rich's
  `LiveRender.position_cursor()` uses the *previous* frame's height to move
  the cursor up; when the render height changes between refreshes the math
  goes stale and rows duplicate (Textualize/rich issue #1144). The pip-style
  fix: **pre-allocate every row before the Live starts, update in place,
  recycle slots**. Constant height from frame 1.
- **`overflow="ellipsis"` does NOT prevent wrapping.** Rich's default column
  overflow is `fold` — long stage text (e.g. `Generating embeddings (3
  chunks)...`) wraps one logical row into 2-3 terminal lines, changing height
  mid-display. You need `Column(no_wrap=True)` — ellipsis then truncates with
  `…` on a single line.
- **Never write log output while a Live is active.** The logger uses its own
  `Console(stderr=True)`, bypassing `Live`'s stdout interception — log lines
  print *inside* the live region and shift the cursor (issue #1052). Buffer
  WARNING+ records during the run and replay them after (see
  `_suspend_live_logs` in `cli.py`).

Other display gotchas:

- `Progress(transient=True)` erases the display on exit — the summary prints
  cleanly afterwards. Non-TTY output skips the live render entirely.
- `progress.update(task_id, total=None)` **ignores None** (None means "don't
  change"). You cannot reset a determinate task to indeterminate via update.
  Create slots indeterminate from the start and hide them with a task field
  (`unused=True`) + slot-aware column subclasses.
- Row-count cap: keep total live rows ≤ terminal height minus whatever was
  printed above (notes panel etc.), or the region scrolls and cursor-up
  erases the wrong lines.
- `script` captures show every Live frame stacked (escape codes unprocessed)
  — that is *not* a display bug. Inspect raw frames by counting
  `CURSOR_UP + ERASE_IN_LINE` pairs; a constant count = constant height.

## 2. Threading and async

- **`httpx.AsyncClient` is not thread-safe.** A shared client across worker
  threads deadlocks (a second thread's client swap orphans the first
  thread's in-flight request). Fix: thread-local client + per-thread event
  loop (`ollama.py`, `llm/base.py`).
- **Don't call `asyncio.run()` per call** — it creates a fresh loop and
  clients bound to the old loop hang forever. One persistent loop per
  thread.
- **Global serialization locks throttle whole pipelines.** A global
  `_llm_sync_lock` (added for safety) ended up serializing the ENTIRE LLM
  phase across files, so converters finished in a burst and then idled.
  Remove locks once the underlying race (thread-local clients) is fixed;
  Ollama queues concurrent requests server-side anyway.
- **Counters mutated from pool threads race.** `stats.skipped += 1` inside
  worker threads needs a lock; the main thread should accumulate results
  returned by workers (`ingested += 1` only in the reaper loop).
- **Two-stage pipeline beats one inline stage.** Convert (fast: MarkItDown
  ~2s) and LLM-ingest (slow: ~10s+) must be decoupled: a convert pool fed in
  bounded just-in-time waves (`CONVERT_AHEAD` in flight, topped up before
  each file's LLM phases) + a parallel ingest pool. Never submit everything
  upfront — that converts in a burst then idles for the rest of the run.
  Count *in-flight* (not-yet-completed) futures for the wave bound.

## 3. Caching

- **Cache keys must be content-based.** Keying the parse cache by file path
  meant re-ingests (even after deleting from Qdrant) never hit the converter
  — the queue "always idle" mystery was 50% this. Key by
  `sha256(file_bytes)`.
- Redis keys are cheap to flush wholesale (`keys rag:parse:*`) when the key
  scheme changes.

## 4. Docker / operations

- **Never run `sudo ./setup.sh`** — it poisons the venv with root-owned
  files; `uv sync` then fails with permission errors. Fix:
  `docker run --rm -v .venv:/v alpine chown -R 1000:1000 /v`.
- **COPY everything the code imports into the image.** `convert_one.py`
  imports `converter_helpers` but the marker Dockerfile never copied it —
  a latent ImportError that only shows at runtime. After moving files,
  grep every `import` and cross-check the Dockerfile COPY lines.
- Docker build context is the repo root even when the Dockerfile lives in a
  subdirectory — COPY paths are relative to the context, not the Dockerfile.
- Router DNS breaks containers intermittently — static `/etc/resolv.conf`
  (1.1.1.1 + 8.8.8.8) + a `resolved.conf.d` drop-in.

## 5. TDD discipline

- Write the failing test **first**, watch it fail for the right reason, then
  implement. Tests written after pass immediately and prove nothing.
- Concurrency tests: track a `max_concurrent` counter under a lock inside
  the mocked slow function (sleep + increment/decrement). Deterministic,
  no wall-time assertions.
- Submission-count tests need spy executors — and filter out non-target
  submits (an ingest pool's submits are tuples, not paths).
- When moving modules, every `patch("module.attr")` target and every
  `import` in tests must move with them — grep is your friend.

## 6. Testing / capture traps

- `script -qec` captures raw terminal bytes; ANSI-stripped analysis that
  does `line.strip()` drops all-whitespace rows (unused pool slots vanish).
- `'0%' in line` matches `100%` — check for false positives.
- A PTY (24×80) is much narrower than a dev terminal — always reproduce
  display bugs at 80 columns.
