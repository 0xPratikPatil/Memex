# Design: Sync Tool — Eliminate Temp Dir + Fix Source Matching

**Date**: 2026-08-06
**Status**: Draft
**Scope**: Fix sync tool to use real source paths (no temp copies) and fix a latent source-matching bug that causes full re-ingestion every run.

---

## Problem Statement

### 1. Pointless temp directory usage for LocalSource

Every sync run creates `tempfile.TemporaryDirectory(prefix="memex_sync_")` in `/tmp`. For local sources, `LocalSource.download()` does `shutil.copy2(file.path, dest)` — copying an already-local file into the temp dir, then parsing the copy and deleting it. If sync fails, the temp folder lingers. Every run generates a new folder.

### 2. Latent source-matching bug (root cause of repeated re-ingestion)

`_ingest_file` stores `source = file_path` in the Qdrant payload, where `file_path` is the **temp path** (`/tmp/memex_sync_XXX/report.pdf`).

`_get_stored_hashes` queries Qdrant filtering `source == source_name` (e.g. `"docs"`).

These never match:
- Stored: `source = "/tmp/memex_sync_abc123/report.pdf"`
- Queried: `source == "docs"`

Result: `_get_stored_hashes` always returns empty → every file appears "new" → re-ingested every sync → sync never converges and generates a new temp folder each run.

---

## Design

### Change 1 — LocalSource uses the real file directly

**File**: `memex/engine/sources/local.py`

`download()` returns `file.path` itself instead of copying to a temp dir:

```python
def download(self, file: SourceFile, dest: Path) -> Path:
    """Local files are already on disk — return the path directly."""
    return Path(file.path)
```

### Change 2 — S3Source uses a stable cache dir

**File**: `memex/engine/sources/s3.py`

S3Source already has a `cache_dir` (`~/.cache/rag/s3` by default, configurable). Its `download()` already uses it correctly. **No change needed** to the method itself, but the sync caller must pass the right target dir (see Change 3). The `cache_dir` persists across runs, so no throwaway temp folder is created.

### Change 3 — Decouple "logical source id" from "file to read"

**File**: `memex/engine/sources/sync.py`

`_ingest_file` currently uses the file path as the Qdrant `source` identifier. Change it to accept two args:

```python
def _ingest_file(engine, source_identifier: str, file_path: str, source_name: str) -> int:
    """Convert and ingest a single file.

    source_identifier: logical path stored in Qdrant (real local path or S3 key).
    file_path: where the file actually lives for reading/parsing.
    """
    result = parse_file(file_path)
    ...
    count = engine.ingest_text(
        result.markdown,
        source_identifier=source_identifier,  # logical path, not temp path
        ...
    )
```

Callers pass:
- **LocalSource**: `source_identifier = sf.path` (real path), `file_path = sf.path` (same, since no copy)
- **S3Source**: `source_identifier = sf.path` (S3 key), `file_path = str(source.download(sf, cache_dir))` (downloaded copy)

### Change 4 — Fix `_get_stored_hashes` filter

**File**: `memex/engine/sources/sync.py`

Filter on the `source_name` metadata field instead of `source`:

```python
scroll_filter = Filter(must=[FieldCondition(key="source_name", match=MatchValue(value=source_name))])
```

Verified: `_ingest_file` passes `metadata={"source_name": source_name}` (sync.py:89), and `ingest_text` spreads `base_meta` (the metadata dict) into `point_meta` at the top level (pipeline.py:691). So `source_name` is a top-level payload key, and the filter matches what's stored.

### Change 5 — Remove the global temp dir wrapper

**File**: `memex/engine/sources/sync.py`

Remove `with tempfile.TemporaryDirectory(prefix="memex_sync_") as tmp_dir:` around the reconcile loop.

For each source, compute a stable download destination:
- **LocalSource**: no download needed — use `sf.path` directly.
- **S3Source**: use `source.cache_dir` (the persistent `~/.cache/rag/s3`).

---

## Files Modified

| File | Change |
|------|--------|
| `memex/engine/sources/sync.py` | Remove temp wrapper, fix `_ingest_file` signature, fix `_get_stored_hashes` filter, use real paths |
| `memex/engine/sources/local.py` | `download()` returns `file.path` directly (no copy) |

---

## Testing

1. **Unit tests**: Update `tests/unit/test_sync.py`:
   - `test_adds_new_files`: mock source returns path, verify `_ingest_file` called with real path
   - `test_detects_changed_files`: verify filter uses `source_name`
   - New test: `LocalSource.download()` returns original path, no copy made
2. **Manual**: Run `memex sync` twice — second run should report all files `unchanged` (converges).
3. **Manual**: Verify no `/tmp/memex_sync_*` folders created after sync.

---

## Risks

- **Existing indexed data**: Files previously ingested with temp-path `source` values won't match the new real-path `source` values. First sync after this change will re-ingest those (they appear "new"), then converge. Acceptable one-time cost; old temp-path chunks become orphans unless cleaned.
- **S3 `cache_dir`**: If two sources share a filename, the S3 cache dir could collide. `download()` returns `dest / file.name` — dedupe by keeping the source-prefix in the path if needed. Low risk for typical usage.
