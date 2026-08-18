# MarkItDown Chunking, OCR Fallback, CLI UX & Stale Code Cleanup

**Date:** 2026-08-18
**Status:** Approved
**Scope:** 4 workstreams — chunking, OCR fallback, CLI status, stale code audit

---

## 1. MarkItDown Clean Chunking

### Problem
MarkItDown outputs markdown with tables, lists, and irregular headers. The local `_recursive_chunk()` splits naively by headers then token size, breaking tables/lists mid-element.

### Solution: Markdown-aware chunk pre-processing

**File: `memex/engine/ingestion/splitter.py`**

Add `chunk_markdown_aware(markdown: str, chunk_size: int, overlap: int) -> list[dict]`:

1. Split markdown into blocks by double-newline
2. Classify each block: `table`, `list`, `code`, `heading`, `paragraph`
3. Never split a table/list/code block mid-element — treat as atomic units
4. Group blocks into chunks respecting `chunking.size` and `chunking.overlap`
5. Tables larger than `chunking.size` get wrapped in a `<table>` container and kept whole (or split at row boundaries if truly massive)
6. Output: list of chunk dicts with `content`, `section_header`, `chunk_index`

**File: `memex/engine/ingestion/splitter.py` — `chunk_file()`**

Add `"markitdown"` branch:
- When `CONVERTER_ENGINE == "markitdown"`: call `parse_file()` then `chunk_markdown_aware()`
- Return chunks in same format as Docling hybrid chunker

### Config
Uses existing `chunking.size` and `chunking.overlap`. No new config keys.

---

## 2. MarkItDown OCR Fallback for Scanned PDFs

### Problem
MarkItDown is the default converter, but scanned PDFs produce poor output. OCR fallback currently only triggers on Marker OOM.

### Solution: Quality-based fallback

**File: `memex/engine/ingestion/loader.py`**

Add quality gate after MarkItDown conversion in `parse_local_file()`:

```python
if config.OCR_FALLBACK and _is_poor_quality(result, file_bytes):
    try:
        from memex.engine.ingestion.ocr_client import convert_with_ocr

        ocr_result = convert_with_ocr(file_bytes, filename)
        if ocr_result.ok and len(ocr_result.markdown) > len(result.markdown or ""):
            result = _ocr_to_conversion(ocr_result)
    except Exception:
        pass  # Keep MarkItDown result
```

**New helpers in `loader.py`:**

- `_is_poor_quality(result: ConversionResult, file_bytes: bytes) -> bool`:
  - Returns `True` if markdown is empty or < 100 chars for a multi-page PDF
  - Returns `True` if text-to-bytes ratio < 0.001 (very low text content)
  - Returns `True` if PDF page count > 1 but total text < 200 chars

- `_ocr_to_conversion(ocr_result: OcrResult) -> ConversionResult`:
  - Wraps `OcrResult` in `ConversionResult` with `source="ocr_fallback"` metadata

### Config
Uses existing `converter.ocr_fallback`, `converter.ocr_url`, `converter.ocr_timeout`. No new config keys.

---

## 3. Unified CLI Status Display

### Problem
`ingest` uses `Rich.Progress` bar with confusing description cycling. `sync` uses `Rich.Live` with compact status. Different paradigms, inconsistent UX.

### Solution: Unified Rich.Live for both commands

**File: `memex/cli.py` — Refactor `_build_compact_status()`**
- Increase `_MAX_VISIBLE_ROWS` from 4 to 6
- Add per-file stage icon + colored stage name + chunk count
- Show converter engine name in header line
- Format: `[ingest] Converting report.pdf ⚙ Converting (2 pages)`

**File: `memex/cli.py` — Refactor `ingest` command**
- Replace `Rich.Progress` with `Rich.Live` + `_build_compact_status()`
- Pass `progress_cb` to `engine.ingest_text()` for sub-stage visibility
- Track active files in `OrderedDict` (same pattern as `sync`)
- Show summary table after completion (same as current)

**File: `memex/cli.py` — Refactor `sync` command**
- Keep existing `Rich.Live` pattern but use same `_build_compact_status()` rendering
- Ensure stage names match `PipelineStage` enum exactly

**New `_on_ingest_progress()` callback in `cli.py`:**
- Bridges `pipeline.py`'s `(msg, pct)` callback to `FileProgress`
- Maps percentage to stage name (same logic as `sync.py`'s `_pipeline_progress`)

---

## 4. Stale Code Full Audit

### Goal
Remove dead code, fix inconsistencies, consolidate callback systems.

### Changes by file

**`memex/cli.py`:**
- Remove redundant `_get_qdrant()` calls (lines 144, 153) — keep only the one at line 154
- Add "Chunking" to `_STAGE_STYLE` dict

**`memex/engine/ingestion/ingestion.py`:**
- Update module docstring: remove "Docling" exclusivity, describe multi-engine routing
- Replace hardcoded `"Docling status"` error messages with `f"{config.CONVERTER_ENGINE.title()} status"`

**`memex/engine/core/progress.py`:**
- Keep `CHUNKING` stage (now used by MarkItDown chunking path)

**`memex/engine/core/pipeline.py`:**
- Change `progress_cb` signature from `Callable[[str, int], None]` to `Callable[[FileProgress], None]`
- Update all internal `_progress("msg", pct)` calls to emit `FileProgress`
- Remove the adapter layer in `sync.py`

**`memex/engine/sources/sync.py`:**
- Remove `_pipeline_progress()` adapter (no longer needed after pipeline.py change)
- Remove hardcoded `"Docling"` in `_emit_pipeline()` messages (line 188, 237)
- Use `config.CONVERTER_ENGINE.title()` instead

**`memex/engine/ingestion/splitter.py`:**
- Remove `is_hybrid_chunker_available()` — only used for display, not routing
- Update `mcp/startup.py` and `pipeline.py` `collection_stats` to skip the check

**`memex/mcp/server.py`:**
- Update `_progress` callbacks (lines 269, 363) to accept `FileProgress` instead of `(msg, pct)`
- Update line 321 to not hardcode "Docling" in success message

---

## Files Modified

| File | Changes |
|------|---------|
| `memex/engine/ingestion/splitter.py` | Add `chunk_markdown_aware()`, add `"markitdown"` branch in `chunk_file()`, remove `is_hybrid_chunker_available()` |
| `memex/engine/ingestion/loader.py` | Add `_is_poor_quality()`, `_ocr_to_conversion()`, quality gate after MarkItDown conversion |
| `memex/cli.py` | Refactor `_build_compact_status()`, refactor `ingest` to use Rich.Live, remove redundant `_get_qdrant()`, add "Chunking" style |
| `memex/engine/core/pipeline.py` | Change `progress_cb` to `FileProgress`-based, update internal `_progress()` calls |
| `memex/engine/sources/sync.py` | Remove `_pipeline_progress()` adapter, fix hardcoded "Docling" strings |
| `memex/engine/ingestion/ingestion.py` | Update docstring, fix hardcoded "Docling" error messages |
| `memex/mcp/server.py` | Update `_progress` callbacks to `FileProgress`, fix hardcoded "Docling" in success message |
| `memex/mcp/startup.py` | Remove `is_hybrid_chunker_available()` call |

## Files NOT Modified

- `config.yaml` / `config.example.yaml` — no new config keys needed
- `docker-compose.yml` — no new services
- `ocr_server.py` / `ocr_client.py` — already correct
- `tests/unit/test_ocr_*.py` — existing tests still valid

## Testing

- All existing unit tests must pass
- New unit tests for `chunk_markdown_aware()` with table/list/code scenarios
- New unit tests for `_is_poor_quality()` with various PDF types
- Manual verification: `memex ingest` shows Rich.Live with sub-stages
- Manual verification: `memex sync` shows same display format
