# Implementation Plan — CLI Progress Tracking

**Spec:** `docs/superpowers/specs/2026-08-07-cli-progress-tracking-design.md`
**Date:** 2026-08-07

## Phase 1: Foundation (progress data model + dependency)

### Task 1.1: Add `rich` dependency
- Edit `pyproject.toml`: add `rich>=13,<14` to `[project.dependencies]`
- Run `uv sync --extra local` to install

### Task 1.2: Create progress data model
- Create `memex/engine/core/progress.py`
- Define `FileProgress` dataclass (path, total, current, stage, chunks, error)
- Define `ProgressCallback` type alias
- Add `__init__.py` import if needed

### Task 1.3: Unit test for FileProgress
- Create `tests/unit/test_progress.py`
- Test `FileProgress` construction with defaults
- Test `ProgressCallback` type works with a lambda

**Checkpoint:** `make lint && make test` — all pass, no regressions

---

## Phase 2: Sync engine progress callbacks

### Task 2.1: Add `progress_cb` to `sync()`
- Edit `memex/engine/sources/sync.py`
- Add `progress_cb: ProgressCallback | None = None` parameter to `sync()`
- Import `FileProgress` from `progress.py`
- Add callback calls at each stage:
  - `Scanning` before listing files from each source
  - `Reconciling` after listing, with total count
  - `Hashing` for each common file before hash check
  - `Parsing` for each new/changed file before Docling conversion
  - `Ingesting` after parse, before Qdrant write
  - `Done` after successful ingest (with chunk count)
  - `Error` on failure (with error message)
  - `Deleting` for each deleted file

### Task 2.2: Unit test for sync progress callbacks
- Create or extend `tests/unit/test_sync.py`
- Mock the sync internals, verify `progress_cb` is called with expected stages
- Test that `progress_cb=None` (default) doesn't break anything

**Checkpoint:** `make lint && make test` — all pass

---

## Phase 3: CLI sync command with Rich progress

### Task 3.1: Add Rich progress to sync CLI
- Edit `memex/cli.py` — sync command
- Import `rich.progress`, `rich.panel`, `rich.table`
- Create `Progress` with columns: SpinnerColumn, Text, BarColumn, TimeElapsedColumn
- Wire `progress_cb` to update the Rich task
- On completion, render a `Panel` with `Table` showing stats (added, changed, deleted, unchanged, errors)

### Task 3.2: CLI smoke test
- Run `memex sync --dry-run` and verify Rich output renders
- Run `memex sync` against a test directory and verify per-file stages + summary

**Checkpoint:** `make lint && make test` — all pass

---

## Phase 4: CLI ingest command with Rich progress

### Task 4.1: Add Rich progress to ingest CLI
- Edit `memex/cli.py` — ingest command
- Count total files before starting (glob or 1 for single file)
- Create `rich.progress.Progress` with per-file task
- In the per-file loop, update task description at stages: Parsing → Ingesting → Done
- On error, show Error and continue
- Final summary panel: Ingested, Errors, Total chunks

### Task 4.2: CLI smoke test
- Run `memex ingest tests/fixtures/sample.pdf` and verify progress bar + summary

**Checkpoint:** `make lint && make test` — all pass

---

## Phase 5: CLI eval command (replace stub)

### Task 5.1: Implement eval CLI
- Edit `memex/cli.py` — eval command
- Replace stub with real implementation:
  - Load golden set via `GoldenSet(golden_set_path)`
  - Create `RAGEngine` instance
  - For each query, call `engine.search(query, top_k=top_k)`
  - Compare results against expected sources using `match_source()`
  - Compute aggregate metrics via `keyword_coverage()`
- Rich spinner during execution
- Render results as a Rich `Table` with per-query and aggregate metrics
- If `--compare-rerank`, run twice and show delta

### Task 5.2: CLI smoke test
- Run `memex eval tests/fixtures/evaluation.xml --top-k 3` and verify output

**Checkpoint:** `make lint && make test` — all pass

---

## Phase 6: MCP rag_sync progress wiring

### Task 6.1: Wire ctx.report_progress() in rag_sync
- Edit `memex/mcp/server.py` — rag_sync tool
- Add `ctx: Context` parameter
- Create `on_progress` callback that calls `ctx.report_progress(p.current, p.total, msg)`
- Pass `progress_cb=on_progress` to `sync()`

### Task 6.2: Verify MCP tool works
- Start MCP server: `uv run memex`
- Call `rag_sync` and verify progress is reported

**Checkpoint:** `make lint && make test` — all pass

---

## Phase 7: End-to-end testing

### Task 7.1: Full e2e test
- Create a test directory with 3-5 sample files (PDF, MD, TXT)
- Run `memex ingest <dir> --recursive` — verify per-file progress + summary
- Run `memex sync` — verify per-file stages + reconciliation summary
- Run `memex sync --dry-run` — verify dry-run output
- Run `memex sync --source-name <name>` — verify source filtering
- Run `memex eval tests/fixtures/evaluation.xml` — verify results table
- Run `memex sync` again (no changes) — verify "unchanged" output

### Task 7.2: Regression test
- Run `make test` — all unit tests pass
- Run `make test-all` — unit + integration tests pass
- Run `make lint && make typecheck` — no new warnings

### Task 7.3: Cleanup
- Remove any test artifacts
- Update CHANGELOG.md with the new feature

**Final checkpoint:** All tests pass, no regressions

---

## Files Modified (summary)

| File | Change |
|------|--------|
| `pyproject.toml` | Add `rich>=13,<14` |
| `memex/engine/core/progress.py` | New — FileProgress dataclass |
| `memex/engine/sources/sync.py` | Add `progress_cb` param + stage callbacks |
| `memex/cli.py` | Rich progress for sync, ingest, eval |
| `memex/mcp/server.py` | Wire `ctx.report_progress()` in rag_sync |
| `tests/unit/test_progress.py` | New — FileProgress unit tests |
| `tests/unit/test_sync.py` | Extend — progress callback tests |
| `CHANGELOG.md` | Add entry |

## Estimated Effort

- Phase 1: ~15 min (foundation)
- Phase 2: ~20 min (sync engine)
- Phase 3: ~25 min (sync CLI)
- Phase 4: ~20 min (ingest CLI)
- Phase 5: ~30 min (eval CLI — replacing stub)
- Phase 6: ~15 min (MCP wiring)
- Phase 7: ~30 min (e2e testing)
- **Total: ~2.5 hours**
