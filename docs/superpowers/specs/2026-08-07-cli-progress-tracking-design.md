# CLI Progress Tracking Design

**Date:** 2026-08-07
**Status:** Approved
**Scope:** Add rich progress bars with per-file stages to sync, ingest, and eval CLI commands. Wire MCP `rag_sync` tool to report progress via `ctx.report_progress()`.

## Problem

The CLI commands (`sync`, `ingest`, `eval`) give no feedback during execution. For large collections, the user stares at a blank terminal with no idea how long it will take or what's happening. The MCP `rag_sync` tool also has zero progress reporting — AI clients get no intermediate feedback.

## Current State

| Command | Per-file progress | Final summary |
|---------|------------------|---------------|
| `memex sync` | None (only debug logs) | One-line stats |
| `memex ingest` | Log per file with `--verbose` | `Ingested: N, Errors: M` |
| `memex eval` | N/A (stub) | N/A |
| MCP `rag_sync` | None | JSON summary |

## Design Decisions

1. **Progress callback pattern** — Add `progress_cb` parameter to `sync()`, matching the existing pattern in `ingest_text()`. CLI creates Rich UI, engine stays display-agnostic.
2. **Rich as required dependency** — `rich>=13` added to `pyproject.toml`. Lightweight (~200KB), core to CLI UX.

## Architecture

### Progress Data Model

**New file: `memex/engine/core/progress.py`**

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class FileProgress:
    """Progress state for a single file operation."""
    path: str              # file path or identifier
    total: int             # total files to process
    current: int           # 1-indexed file number
    stage: str             # stage label
    chunks: int = 0        # chunks produced (on success)
    error: str = ""        # error message (on failure)

ProgressCallback = Callable[[FileProgress], None]
```

### Stages per file

| Stage | Meaning |
|-------|---------|
| `Scanning` | Listing files from source |
| `Reconciling` | Comparing current vs stored state |
| `Hashing` | Computing content hash for change detection |
| `Parsing` | Docling document conversion |
| `Ingesting` | Writing chunks + embeddings to Qdrant |
| `Done` | Successfully processed |
| `Error` | Failed (error field populated) |
| `Deleting` | Removing stale chunks |

## Changes

### 1. `memex/engine/core/progress.py` (new)

- `FileProgress` dataclass
- `ProgressCallback` type alias

### 2. `memex/engine/sources/sync.py`

Add `progress_cb: ProgressCallback | None = None` to `sync()`.

Call `progress_cb` at these points:
- Before listing files from each source → `Scanning`
- After listing, before processing → `Reconciling` (with total count)
- For each common file, before hash check → `Hashing`
- For each new/changed file, before parse → `Parsing`
- After parse, before ingest → `Ingesting`
- After successful ingest → `Done` (with chunk count)
- On failure → `Error` (with error message)
- For each deleted file → `Deleting`

### 3. `memex/cli.py` — sync command

Replace the current bare `asyncio.run(_run())` with:

1. Create `rich.progress.Progress` with columns: Spinner, Task description, BarColumn, TimeElapsedColumn
2. Create a progress task for the overall sync
3. Wrap the progress callback to update the Rich task on each `FileProgress`
4. On completion, render a Rich `Table` panel with stats

### 4. `memex/cli.py` — ingest command

The ingest loop is inline in cli.py (calls `parse_file()` and `engine.ingest_text()` directly), so progress display goes directly into the loop — no callback needed:

1. Count total files first (glob the directory, or 1 for single file)
2. Create `rich.progress.Progress` with per-file task
3. In the per-file loop, update Rich task at stages: `Parsing` → `Ingesting` → `Done` (with chunk count from `ingest_text` return)
4. On error, show `Error` stage and continue to next file
5. Final summary panel: `Ingested: N, Errors: M, Total chunks: C`

Note: Unlike sync, ingest doesn't need a `progress_cb` parameter on a separate function — the progress is built directly into the CLI loop.

### 5. `memex/cli.py` — eval command

Currently a stub that prints "(evaluation framework not yet integrated)". Replace with real implementation:

1. Load golden set via `GoldenSet(golden_set_path)` (from `memex.engine.evaluation.golden`)
2. Create `RAGEngine` instance
3. For each query in golden set, call `engine.search(query, top_k=top_k)` and compare results against expected sources
4. Compute aggregate metrics via `keyword_coverage()` and `match_source()` (from `memex.engine.evaluation.metrics`)
5. Rich spinner during execution, then render results as a Rich `Table` with per-query and aggregate metrics
6. If `--compare-rerank` flag, run twice (with and without reranking) and show delta table

### 6. `memex/mcp/server.py` — rag_sync tool

Add `ctx: Context` parameter to `rag_sync`. Wire `progress_cb` to `ctx.report_progress()`:

```python
async def rag_sync(input: SyncInput, ctx: Context) -> str:
    def on_progress(p: FileProgress):
        msg = f"{p.stage} {os.path.basename(p.path)}"
        ctx.report_progress(p.current, p.total, msg)
    
    result = await sync(config, progress_cb=on_progress, ...)
```

### 7. `pyproject.toml`

Add `rich>=13,<14` to `[project.dependencies]`.

## Files Modified

| File | Change |
|------|--------|
| `memex/engine/core/progress.py` | New file — `FileProgress` dataclass |
| `memex/engine/sources/sync.py` | Add `progress_cb` param, call at stages |
| `memex/cli.py` | Rich progress bars for sync, ingest, eval |
| `memex/mcp/server.py` | Wire `ctx.report_progress()` in `rag_sync` |
| `pyproject.toml` | Add `rich>=13,<14` dependency |

## Testing

- **Unit**: `FileProgress` construction, sync engine calls `progress_cb` expected N times
- **CLI smoke**: `memex sync --dry-run` renders Rich output
- **E2E sync**: `memex sync` against test dir, verify per-file stages + summary
- **E2E ingest**: `memex ingest tests/fixtures/sample.pdf` — progress bar
- **E2E eval**: `memex eval tests/fixtures/evaluation.xml` — spinner + results
- **Existing tests**: Run `make test` to verify no regressions

## Out of Scope

- Progress bars for MCP ingest tools (already have `ctx.report_progress()`)
- Async concurrency in sync engine (stays sequential)
- ETA estimation (Rich provides time elapsed, ETA requires speed tracking)
- `serve` command progress (just starts server, no progress to show)
