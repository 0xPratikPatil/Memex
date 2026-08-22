# CLI Status Display & OCR Fallback Hardening

**Date:** 2026-08-22
**Status:** Draft

## Problem

1. **CLI output** — Rich Table shows all files (including completed), has table borders/headers, and doesn't feel like a live progress display. User wants: colors, icons, only in-pipeline files, per-file stage + percentage, overall progress bar.

2. **OCR fallback not triggering** — MarkItDown converts scanned PDFs but produces poor/empty markdown. The quality gate (`_is_poor_quality`) is too lenient (needs <100 chars AND >10KB), and exceptions are silently swallowed (`except Exception: pass`). OCR service may not even be running.

## Design

### Part 1: CLI Status Display

**Format:** Plain text lines with Rich markup, no table.

```
  ✓ report.pdf      Done         12 chunks
  ↷ notes.md        Skipped
  ⚙  scan.pdf       Converting   3/10 pages
  ✗ bad.pdf         Error        timeout
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4/10 40.0%
```

**Rules:**
- Only show files currently in pipeline (active stages: Converting, Hashing, Parsing, Converting, Chunking, Embedding, Storing)
- Completed files (Done/Skipped/Error) stay visible until the command finishes
- Each line: `icon filename  Stage  detail` with per-stage colors
- Bottom: overall progress bar with percentage and count
- Use `rich.group.Group` + `rich.text.Text` (not Table) — `live.update()` replaces the whole renderable, no appending

**Stage styles** (reuse existing `_STAGE_STYLE`):
| Stage | Icon | Color |
|-------|------|-------|
| Done | ✓ | green |
| Skipped | ↷ | cyan |
| Converting | ⚙ | cyan |
| Hashing | # | blue |
| Parsing | p | cyan |
| Embedding | emb | yellow |
| Storing | ··· | green |
| Error | ✗ | red |

**Implementation:**
- Replace `_build_live_table()` with `_build_live_display()` returning `rich.group.Group`
- Each file = one `Text` line with icon + name + stage + detail
- Bottom = `rich.progress.BarColumn` style progress line
- `ingest` and `sync` commands both use this

### Part 2: OCR Fallback Hardening

**Changes in `loader.py`:**

1. **Lower quality thresholds** — `_is_poor_quality()`:
   - Empty text → triggers OCR (unchanged)
   - `< 500 chars AND > 5KB` → triggers OCR (was: <100 AND >10KB)
   - Text-to-bytes ratio `< 0.005` → triggers OCR (was: <0.001)

2. **Pre-flight OCR check** — before calling `convert_with_ocr()`, call `is_ocr_available()`. If unreachable, log warning and skip (don't hang).

3. **Replace silent `except Exception: pass`** with:
   ```python
   except Exception as e:
       logger.warning("OCR fallback failed for %s: %s", filename, e)
   ```

4. **Verify OCR output** — after OCR, check `ocr_result.ok` AND `len(ocr_result.markdown) > 0`. Only replace MarkItDown result if OCR actually produced output.

**Flow:**
```
MarkItDown converts → _is_poor_quality()?
  ├─ No  → use MarkItDown result
  └─ Yes → is_ocr_available()?
            ├─ No  → log warning, use MarkItDown result
            └─ Yes → convert_with_ocr()
                      ├─ ok AND has output → use OCR result
                      └─ fail/empty → log warning, use MarkItDown result
```

## Files to modify

- `memex/cli.py` — replace `_build_live_table` with `_build_live_display`, update `ingest` and `sync`
- `memex/engine/ingestion/loader.py` — fix `_is_poor_quality` thresholds, add logging to OCR fallback, add pre-flight check

## Testing

- Unit test `_is_poor_quality` with new thresholds
- Unit test OCR fallback flow (mock: available/unavailable, success/failure)
- Manual: run `memex sync` and verify clean output
