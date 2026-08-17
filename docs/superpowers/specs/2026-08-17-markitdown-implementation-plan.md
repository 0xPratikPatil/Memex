# MarkItDown Integration — Implementation Plan

**Spec:** `docs/superpowers/specs/2026-08-17-markitdown-integration-design.md`  
**Date:** 2026-08-17

---

## Phase 1: Docker Container (Foundation)

### Task 1.1: Create `markitdown.Dockerfile`
- Base: `python:3.12-slim`
- Install: `pip install --no-cache-dir 'markitdown[all]'`
- Copy: `markitdown_server.py` to `/app/`
- Expose: port 5003
- CMD: `uvicorn markitdown_server:app --host 0.0.0.0 --port 5003`
- **Verify:** `docker build -t memex-markitdown .` succeeds

### Task 1.2: Create `markitdown_server.py`
- FastAPI app (~80 lines)
- `POST /convert` — accepts multipart form (file bytes + filename), calls `MarkItDown().convert()`, returns JSON
- `GET /health` — returns `{"status": "ok"}`
- Error handling: catches MarkItDown exceptions, returns structured error JSON
- Logging: request/response logging with source, format, processing_time
- **Verify:** Server starts, responds to health check

### Task 1.3: Add to `docker-compose.yml`
- Add `markitdown` service block (port 5003, 1G memory limit, healthcheck)
- **Verify:** `docker compose up markitdown` starts and becomes healthy

---

## Phase 2: Client Module

### Task 2.1: Create `memex/engine/ingestion/markitdown_client.py`
- `MarkItDownResult` dataclass (markdown, metadata, format, processing_time)
- `async def convert_markdown(file_bytes, filename)` — POST to markitdown service
- Timeout from config (`markitdown_timeout`, default 30s)
- Retry on connection errors (3 attempts, exponential backoff)
- Structured logging on success/failure
- **Verify:** Unit tests pass with mocked HTTP

### Task 2.2: Add config constants to `memex/engine/core/config.py`
- `MARKITDOWN_URL = "markitdown_url"`
- `MARKITDOWN_TIMEOUT = "markitdown_timeout"`
- **Verify:** Config loads without error

### Task 2.3: Update `config.yaml` and `config.example.yaml`
- Add `markitdown_url: "http://localhost:5003"`
- Add `markitdown_timeout: 30.0`
- **Verify:** Config validates correctly

---

## Phase 3: Loader Integration

### Task 3.1: Update `memex/engine/ingestion/loader.py`
- Import `markitdown_client`
- Add `elif engine == "markitdown"` branch in `parse_file()`
- Route to `markitdown_client.convert_markdown()`
- Both clients return same shape → rest of pipeline unchanged
- **Verify:** Loader routes correctly for both engines

---

## Phase 4: Tests

### Task 4.1: Create `tests/unit/test_markitdown_client.py`
- Test successful conversion (mock HTTP)
- Test connection error retry
- Test timeout handling
- Test empty output → CorruptedDocumentError
- Test server 500 → ConversionError
- **Verify:** All tests pass

### Task 4.2: Create `tests/unit/test_markitdown_server.py`
- Test `/health` endpoint
- Test `/convert` with valid file
- Test `/convert` with unsupported format
- Test `/convert` with corrupt file
- **Verify:** All tests pass

### Task 4.3: Run full test suite
- `uv run pytest tests/unit/` — all tests pass
- `uv run ruff check .` — lint clean
- `uv run mypy .` — typecheck clean
- **Verify:** No regressions

---

## Phase 5: E2E Verification

### Task 5.1: Build and start MarkItDown container
- `docker compose build markitdown`
- `docker compose up -d markitdown`
- Verify health check passes

### Task 5.2: Test with a DOCX file
- `memex ingest /path/to/file.docx` with `converter.engine: markitdown`
- Verify: conversion succeeds, chunks stored in Qdrant
- Query: `rag_query` returns relevant results

### Task 5.3: Test parallel ingestion
- Ingest multiple files simultaneously
- Verify: all complete, no errors

### Task 5.4: Verify Marker still works
- Set `converter.engine: marker`
- Ingest a PDF
- Verify: Marker path works unchanged

---

## Execution Order

```
Phase 1 (Docker) → Phase 2 (Client) → Phase 3 (Loader) → Phase 4 (Tests) → Phase 5 (E2E)
```

Each phase is independently testable. Phase 1-3 are the core implementation. Phase 4 validates correctness. Phase 5 validates end-to-end.

## Commit Strategy

- **Commit after Phase 1:** "feat: add MarkItDown Docker container and server"
- **Commit after Phase 2:** "feat: add MarkItDown client module and config"
- **Commit after Phase 3:** "feat: integrate MarkItDown into loader.py"
- **Commit after Phase 4:** "test: add MarkItDown client and server unit tests"
- **Commit after Phase 5:** "feat: MarkItDown integration complete — E2E verified"
