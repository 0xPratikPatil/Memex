# Implementation Plan: MarkItDown Chunking, OCR Fallback, CLI UX & Stale Code Cleanup

**Spec:** `docs/superpowers/specs/2026-08-18-chunking-ocr-fallback-cli-cleanup-design.md`
**Estimated effort:** 4 phases, ~2-3 hours total

---

## Phase 1: Stale Code Full Audit (do first — changes callback signature)

**Why first:** Changes `pipeline.py`'s `progress_cb` signature from `(msg, pct)` to `FileProgress`. This affects `sync.py`, `mcp/server.py`, and `cli.py`. Do this before other workstreams to avoid merge conflicts.

### Step 1.1: Fix `memex/engine/core/pipeline.py`
- Change `progress_cb: Callable[[str, int], None] | None = None` to `progress_cb: Callable[[FileProgress], None] | None = None` in `ingest_text()` (line 594) and `ingest_prechunked()` (line 771)
- Import `FileProgress` from `memex.engine.core.progress`
- Update internal `_progress(msg, pct)` calls to emit `FileProgress(path="", total=0, current=pct, stage=msg, chunks=0)`
- Remove the `progress_cb(msg, pct)` pattern — replace with `progress_cb(FileProgress(...))`

### Step 1.2: Fix `memex/engine/sources/sync.py`
- Remove `_pipeline_progress()` adapter function (lines 266-285)
- In `_ingest_file()`, pass `progress_cb` directly to `engine.ingest_text()` (line 302) and `engine.ingest_prechunked()` (line 315) — no adapter needed
- Replace hardcoded `"Docling"` in `_emit_pipeline()` messages (lines 188, 237) with `config.CONVERTER_ENGINE.title()`

### Step 1.3: Fix `memex/mcp/server.py`
- Update `_progress` callbacks (lines 269, 363) to accept `FileProgress` instead of `(msg, pct)`
- Update line 321: replace `f"(Docling: {result.processing_time:.1f}s, "` with `f"({config.CONVERTER_ENGINE.title()}: {result.processing_time:.1f}s, "`

### Step 1.4: Fix `memex/cli.py`
- Remove redundant `_get_qdrant()` calls at lines 144 and 153 — keep only line 154
- Add `"Chunking"` entry to `_STAGE_STYLE` dict: `("⚙", "cyan", "Chunking")`

### Step 1.5: Fix `memex/engine/ingestion/ingestion.py`
- Update module docstring: replace "Docling parsing" with "document conversion (marker/markitdown/docling)"
- Replace hardcoded `"Docling status"` error messages (lines 100, 276) with `f"{config.CONVERTER_ENGINE.title()} status"`

### Step 1.6: Fix `memex/engine/ingestion/splitter.py`
- Remove `is_hybrid_chunker_available()` function (lines 344-354)

### Step 1.7: Fix `memex/mcp/startup.py`
- Remove `is_hybrid_chunker_available()` import and call (line 53)
- Update `collection_stats` display to skip the check

### Step 1.8: Run tests
```bash
uv run pytest tests/unit/ -x -q
uv run ruff check .
uv run mypy memex/
```

---

## Phase 2: MarkItDown Clean Chunking

### Step 2.1: Add `chunk_markdown_aware()` to `memex/engine/ingestion/splitter.py`
```python
def chunk_markdown_aware(
    markdown: str,
    chunk_size: int = 1024,
    overlap: int = 128,
    filename: str = "",
) -> list[dict]:
```

Implementation:
1. Split markdown into blocks by `\n\n` (double newline)
2. Classify each block by detecting patterns:
   - `table`: starts with `|` or contains `<table>`
   - `list`: starts with `- `, `* `, `1. `, etc.
   - `code`: starts with ``` or `    `
   - `heading`: starts with `#`
   - `paragraph`: everything else
3. Never split a table/list/code block — treat as atomic
4. Group blocks into chunks: accumulate blocks until adding the next would exceed `chunk_size`, then start a new chunk
5. Apply overlap by including the last `overlap` chars of the previous chunk at the start of the next
6. Return list of `{"content": ..., "section_header": ..., "chunk_index": ...}` dicts

### Step 2.2: Add `"markitdown"` branch in `chunk_file()`
In `chunk_file()` (line 255), add:
```python
elif CONVERTER_ENGINE == "markitdown":
    result = parse_file(file_path_or_url, source_identifier=source_identifier)
    chunks = chunk_markdown_aware(
        result.markdown,
        chunk_size=config.CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP,
        filename=filename,
    )
    return {"chunks": chunks, "markdown": result.markdown}
```

### Step 2.3: Write unit tests
- `tests/unit/test_splitter.py` — test `chunk_markdown_aware()` with:
  - Table-heavy markdown (should not split tables)
  - List-heavy markdown (should not split lists)
  - Code blocks (should not split code)
  - Mixed content
  - Empty input
  - Very large table (> chunk_size)

### Step 2.4: Run tests
```bash
uv run pytest tests/unit/test_splitter.py -v
```

---

## Phase 3: MarkItDown OCR Fallback

### Step 3.1: Add `_is_poor_quality()` to `memex/engine/ingestion/loader.py`
```python
def _is_poor_quality(result: ConversionResult, file_bytes: bytes) -> bool:
    """Detect poor conversion quality (e.g., scanned PDF via MarkItDown)."""
    text = (result.markdown or "").strip()
    if not text:
        return True
    if len(text) < 100 and len(file_bytes) > 10_000:  # short text, large file
        return True
    if len(text) / max(len(file_bytes), 1) < 0.001:  # very low text-to-bytes ratio
        return True
    return False
```

### Step 3.2: Add `_ocr_to_conversion()` to `memex/engine/ingestion/loader.py`
```python
def _ocr_to_conversion(ocr_result: OcrResult) -> ConversionResult:
    """Wrap OcrResult in ConversionResult."""
    return ConversionResult(
        markdown=ocr_result.markdown,
        status="success" if ocr_result.ok else "error",
        processing_time=ocr_result.processing_time,
        errors=[] if ocr_result.ok else ["OCR fallback failed"],
    )
```

### Step 3.3: Add quality gate in `parse_local_file()` after MarkItDown conversion
After the `CONVERTER_ENGINE == "markitdown"` block (around line 420), add:
```python
if config.OCR_FALLBACK and _is_poor_quality(result, file_bytes):
    try:
        from memex.engine.ingestion.ocr_client import convert_with_ocr
        ocr_result = convert_with_ocr(file_bytes, filename)
        if ocr_result.ok and len(ocr_result.markdown or "") > len(result.markdown or ""):
            result = _ocr_to_conversion(ocr_result)
    except Exception:
        pass  # Keep MarkItDown result
```

### Step 3.4: Write unit tests
- `tests/unit/test_loader.py` — test `_is_poor_quality()` with:
  - Empty markdown → True
  - Short text + large file → True
  - Low text-to-bytes ratio → True
  - Normal text → False
- Test `parse_local_file()` with mocked MarkItDown returning poor quality → falls back to OCR

### Step 3.5: Run tests
```bash
uv run pytest tests/unit/test_loader.py -v
```

---

## Phase 4: Unified CLI Status Display

### Step 4.1: Refactor `_build_compact_status()` in `memex/cli.py`
- Increase `_MAX_VISIBLE_ROWS` from 4 to 6
- Add converter engine name in header: `[ingest] marker · 3 files active`
- Per-file format: `  ⚙ Converting report.pdf (2 pages)`
- Chunk count when available: `  ✓ Done report.pdf (12 chunks)`

### Step 4.2: Refactor `ingest` command to use Rich.Live
- Replace `Rich.Progress` with `Rich.Live` + `_build_compact_status()`
- Create `_ingest_files()` async helper that:
  - Processes files sequentially
  - Updates `active_files` OrderedDict per stage
  - Calls `live.update(_build_compact_status(active_files, ...))`
- Pass `progress_cb` to `engine.ingest_text()` for sub-stage visibility

### Step 4.3: Add `_on_ingest_progress()` callback
- Bridges `pipeline.py`'s `FileProgress` to the `active_files` OrderedDict
- Maps stage names to display format

### Step 4.4: Update `sync` command
- Ensure same `_build_compact_status()` rendering
- Stage names match `PipelineStage` enum exactly

### Step 4.5: Write unit tests
- Test `_build_compact_status()` with various file states
- Test stage icon/color mapping

### Step 4.6: Run full test suite
```bash
uv run pytest tests/unit/ -x -q
uv run ruff check .
uv run ruff format .
uv run mypy memex/
```

---

## Verification Checklist

- [ ] All existing 699 unit tests pass
- [ ] New tests pass for chunking, OCR fallback, CLI display
- [ ] `ruff check .` clean
- [ ] `ruff format .` clean
- [ ] `mypy memex/` clean
- [ ] No hardcoded "Docling" strings remain (grep for "Docling" in *.py)
- [ ] No redundant `_get_qdrant()` calls
- [ ] `CHUNKING` stage is used in MarkItDown chunking path
- [ ] `is_hybrid_chunker_available()` removed
- [ ] `pipeline.py` callback signature is `FileProgress`-based
- [ ] `_pipeline_progress` adapter removed from `sync.py`
