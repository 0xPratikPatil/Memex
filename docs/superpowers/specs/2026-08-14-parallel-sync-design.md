# Parallel Sync Design

**Date:** 2026-08-14
**Status:** Approved
**Scope:** Sync engine parallelism, double Docling elimination, async Qdrant

## Problem

The sync pipeline processes files **100% sequentially** — one file at a time through download → parse → chunk → embed → upsert. With millions of files, this makes sync unusable even on high-end GPUs (A4000, RTX4000).

Additionally, when `chunking.strategy=hybrid`, each file hits Docling **twice**: once for markdown conversion, once for structure-aware chunking. This doubles the Docling load.

## Bottlenecks Identified

| # | Bottleneck | Severity | Location |
|---|-----------|----------|----------|
| 1 | Sequential file processing | Critical | `sync.py:216-294` |
| 2 | Double Docling API calls (hybrid mode) | High | `sync.py:84` → `pipeline.py:602` → `splitter.py` |
| 3 | Sync bypasses IngestionOrchestrator | High | `sync.py` vs `ingestion.py` |
| 4 | Blocking Qdrant calls in async context | Medium | `sync.py:53-71`, `sync.py:263-294` |
| 5 | O(N) hash loading per source | Medium | `sync.py:45-71` |
| 6 | No streaming for large file sets | Medium | `sync.py:156-189` |

## Design

### 1. Eliminate Double Docling Calls

**When `chunking.strategy=hybrid`:** Use `splitter.chunk_file(include_doc=True)` which returns both chunks and converted markdown in a single API call.

**Current:**
```
parse_file(file_path)  →  Docling /v1/convert/source  →  markdown
create_chunks(markdown) →  splitter.chunk_file()      →  Docling /v1/chunk/hybrid/source  →  chunks
```

**Proposed:**
```
chunk_file(file_path, include_doc=True)  →  Docling /v1/chunk/hybrid/source  →  chunks + markdown
```

**Changes to `_ingest_file()` in `sync.py`:**

The key change is passing a progress callback through the entire chain so ALL pipeline stages are visible in the sync progress display:

```python
def _ingest_file(engine, source_identifier, file_path, source_name, progress_cb=None):
    """Convert and ingest a single file with full stage reporting.
    
    Stages surfaced via progress_cb:
      - "Converting" (70%) — Docling conversion / chunking
      - "Context" (73%) — Context enrichment per chunk  
      - "Metadata" (74%) — Entity/topic extraction
      - "Embedding" (75-89%) — Dense + sparse embedding
      - "Storing" (90%) — Qdrant upsert
    """
    from memex.engine.core import config
    
    strategy = config.CHUNK_STRATEGY.lower()
    
    def _pipeline_progress(msg, pct):
        if progress_cb:
            progress_cb(msg, pct)
    
    if strategy == "hybrid":
        from memex.engine.ingestion.splitter import chunk_file
        _pipeline_progress("Converting + chunking (Docling)...", 70)
        result = chunk_file(file_path, include_doc=True)
        markdown = result.get("markdown", "")
        chunks = result.get("chunks", [])
        if not markdown:
            from memex.engine.ingestion.loader import parse_file
            parse_result = parse_file(file_path)
            if not parse_result.ok:
                raise RuntimeError(f"Docling conversion failed: {parse_result.status}")
            markdown = parse_result.markdown
    else:
        from memex.engine.ingestion.loader import parse_file
        _pipeline_progress("Converting (Docling)...", 70)
        result = parse_file(file_path)
        if not result.ok:
            raise RuntimeError(f"Docling conversion failed: {result.status}")
        markdown = result.markdown
        chunks = None
    
    content_hash = engine.compute_file_hash(markdown.encode())
    
    if chunks is not None:
        count = engine.ingest_prechunked(
            chunks=chunks,
            markdown=markdown,
            source_identifier=source_identifier,
            metadata={...},
            content_hash=content_hash,
            progress_cb=_pipeline_progress,
        )
    else:
        count = engine.ingest_text(
            markdown,
            source_identifier=source_identifier,
            metadata={...},
            content_hash=content_hash,
            progress_cb=_pipeline_progress,
        )
    return count
```

**New method `RAGEngine.ingest_prechunked()`** in `pipeline.py`: Takes pre-chunked data (from HybridChunker), skips the `create_chunks()` call, and goes directly to context enrichment → metadata extraction → embedding → upsert.

```python
def ingest_prechunked(
    self,
    chunks: list[dict[str, Any]],
    markdown: str,
    source_identifier: str,
    metadata: dict[str, Any] | None = None,
    content_hash: str = "",
    progress_cb: Callable[[str, int], None] | None = None,
) -> int:
    """Ingest pre-chunked data, skipping the create_chunks() call.

    Used when chunking was done externally (e.g., by HybridChunker in a
    single Docling API call). Skips the double Docling call problem.
    
    Embeddings are stored to Qdrant IMMEDIATELY at the end of this call —
    no batching across files. Each file's vectors are persisted as soon as
    all stages complete for that file.
    """
    def _progress(msg: str, pct: int) -> None:
        if progress_cb:
            progress_cb(msg, pct)
        logger.info("ingest [%d%%] %s", pct, msg)

    if not chunks:
        raise ValueError("No chunks to ingest.")

    raw_chunks = [c for c in chunks if len(c.get("content", "").strip()) >= config.MIN_CHUNK_LEN]
    if not raw_chunks:
        raise ValueError("No chunks above MIN_CHUNK_LEN after filtering.")

    from memex.engine.ingestion.hashing import dedup_chunks
    raw_chunks = dedup_chunks(raw_chunks)
    if not raw_chunks:
        raise ValueError("No chunks after deduplication.")

    # Context enrichment — stage: "Context" (73%)
    if config.ENABLE_CONTEXTUAL_RETRIEVAL:
        ctx_gen = ContextGenerator(self._llm)
        document_summary = ""
        if config.CONTEXT_STRATEGY == "summary":
            _progress("Generating document summary...", 71)
            try:
                document_summary = ctx_gen.generate_document_summary(markdown)
            except Exception:
                logger.warning("Document summary generation failed", exc_info=True)
        _progress("Adding context to chunks...", 73)
        raw_chunks = ctx_gen.enrich_chunks(raw_chunks, document_summary=document_summary)

    # Metadata extraction — stage: "Metadata" (74%)
    if config.ENABLE_METADATA_EXTRACTION:
        extractor = MetadataExtractor(self._llm)
        _progress("Extracting metadata...", 74)
        batch_meta = extractor.extract_batch(
            chunks=raw_chunks,
            document_text=markdown,
            source_identifier=source_identifier,
        )
        for chunk, meta in zip(raw_chunks, batch_meta, strict=True):
            chunk["metadata"] = meta

    # Embedding — stage: "Embedding" (75-89%)
    _progress(f"Generating embeddings ({len(raw_chunks)} chunks)...", 75)
    chunk_texts = [c["content"] for c in raw_chunks]
    raw_texts = [strip_context_prefix(c["content"]) for c in raw_chunks]

    import concurrent.futures
    contextual_vecs = None
    max_workers = 3 if config.ENABLE_CONTEXTUAL_RETRIEVAL else 2
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        dense_fut = pool.submit(self._dense_embed_batch, raw_texts)
        sparse_fut = pool.submit(self._sparse_embed, chunk_texts)
        ctx_fut = pool.submit(self._dense_embed_batch, chunk_texts) if config.ENABLE_CONTEXTUAL_RETRIEVAL else None

        dense_vecs = dense_fut.result()
        sparse_vecs = sparse_fut.result()
        if ctx_fut:
            contextual_vecs = ctx_fut.result()

    # Store in Qdrant — stage: "Storing" (90%)
    # EMBEDDINGS STORED IMMEDIATELY PER FILE — no cross-file batching
    _progress("Storing in Qdrant...", 90)
    # ... build points and upsert (same as ingest_text lines 663-750)
    
    logger.info("Ingested %d chunks for '%s' — embeddings persisted", len(points), source_identifier)
    return len(points)
```

**Key differences from `ingest_text()`:**
1. No `create_chunks()` call — chunks come from the caller
2. Progress callback is threaded through — all stages are surfaced
3. Embeddings stored immediately per file — confirmed at line "EMBEDDINGS STORED IMMEDIATELY PER FILE"

**Fallback:** If `chunk_file()` fails or returns empty, fall back to `parse_file()` + `create_chunks()` (current behavior).

### 2. Parallel File Processing in sync()

Replace sequential `for` loops with bounded `asyncio.gather()`:

```python
async def sync(config_module, source_name=None, dry_run=False, progress_cb=None):
    stats = SyncStats()
    
    # Phase 1: List sources (unchanged)
    # Phase 2: Get stored hashes (unchanged)
    # Phase 3: Reconcile to determine new/changed/deleted (unchanged)
    
    # Phase 4: Process files concurrently
    max_concurrent = config.MAX_CONCURRENT_SYNC  # NEW config key
    
    for src_cfg in source_configs:
        src_name = src_cfg.get("name", src_cfg.get("type", "local"))
        source = get_source(src_cfg.get("type", "local"), src_cfg)
        
        # ... reconcile logic unchanged ...
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _process_file(path, sf, action, file_idx, total):
            """Process a single file with bounded concurrency."""
            async with semaphore:
                try:
                    if action == "add":
                        _emit(path, "Parsing", file_idx, total)
                        local = await asyncio.to_thread(source.download, sf, download_dir)
                        _emit(path, "Ingesting", file_idx, total)
                        chunks = await asyncio.to_thread(
                            _ingest_file, engine, sf.path, str(local), src_name
                        )
                        _emit(path, "Done", file_idx, total, chunks=chunks)
                        return ("added", path)
                    
                    elif action == "update":
                        await asyncio.to_thread(engine.delete_by_source, path)
                        _emit(path, "Parsing", file_idx, total)
                        local = await asyncio.to_thread(source.download, sf, download_dir)
                        _emit(path, "Ingesting", file_idx, total)
                        chunks = await asyncio.to_thread(
                            _ingest_file, engine, sf.path, str(local), src_name
                        )
                        _emit(path, "Done", file_idx, total, chunks=chunks)
                        return ("changed", path)
                    
                    elif action == "delete":
                        _emit(path, "Deleting", file_idx, total)
                        await asyncio.to_thread(engine.delete_by_source, path)
                        _emit(path, "Done", file_idx, total)
                        return ("deleted", path)
                
                except Exception as exc:
                    _emit(path, "Error", file_idx, total, error=str(exc))
                    return ("error", path, str(exc))
        
        # Build and run tasks
        # NOTE: file_idx is passed explicitly to avoid closure capture issues
        tasks = []
        file_idx = 0
        total_files = len(new_paths) + len(common_paths) + len(deleted_paths)
        
        for path in new_paths:
            file_idx += 1
            tasks.append(_process_file(path, current_map[path], "add", file_idx, total_files))
        
        for path in changed_paths:
            file_idx += 1
            tasks.append(_process_file(path, current_map[path], "update", file_idx, total_files))
        
        for path in deleted_paths:
            file_idx += 1
            tasks.append(_process_file(path, current_map[path], "delete", file_idx, total_files))
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Tally results
        for r in results:
            if isinstance(r, Exception):
                stats.errors.append(str(r))
            elif r[0] == "added":
                stats.added += 1
            elif r[0] == "changed":
                stats.changed += 1
            elif r[0] == "deleted":
                stats.deleted += 1
            elif r[0] == "error":
                stats.errors.append(r[2])
    
    return stats
```

### 3. Async Qdrant Wrappers

Add async wrappers for blocking Qdrant calls:

```python
# In sync.py or a new utils module

async def _get_stored_hashes_async(engine, source_name: str) -> dict[str, str]:
    """Async wrapper for _get_stored_hashes using to_thread."""
    return await asyncio.to_thread(_get_stored_hashes, engine, source_name)

async def _delete_by_source_async(engine, path: str) -> None:
    """Async wrapper for delete_by_source using to_thread."""
    await asyncio.to_thread(engine.delete_by_source, path)
```

### 4. New Config Key

Add to `config.py`:
```python
MAX_CONCURRENT_SYNC: int = _cfg_int("ingestion.max_concurrent_sync", 8)
```

Add to `config.yaml`:
```yaml
ingestion:
  max_concurrent_sync: 8        # Files processed in parallel during sync
  max_concurrent_parses: 3      # (existing) Docling parse concurrency for batch ingest
```

### 5. Progress Callback Updates — All Stages Visible

**Problem:** Currently `_ingest_file()` doesn't pass a progress callback to `ingest_text()`, so the fine-grained pipeline stages (chunking, context, metadata, embedding, storing) are silent. User only sees "Parsing" → "Ingesting" → "Done".

**Solution:** Thread the progress callback through the entire chain:
```
sync._emit() → _ingest_file(progress_cb=...) → ingest_text(progress_cb=...) / ingest_prechunked(progress_cb=...)
```

The sync `_emit()` function now shows the current pipeline stage for each file:

```python
# Stage names shown in CLI progress display:
# "Converting"    — Docling conversion / chunking
# "Context"       — Context enrichment per chunk
# "Metadata"      — Entity/topic extraction  
# "Embedding"     — Dense + sparse embedding generation
# "Storing"       — Qdrant upsert
# "Done"          — Complete with chunk count
# "Error"         — Failed with error message
```

**CLI display example with concurrent files:**
```
⠋ report.pdf    — Embedding (42 chunks)
⠙ data.csv      — Storing (18 chunks)  
⠸ notes.md      — Context (7 chunks)
⠹ analysis.pdf  — Done (56 chunks)
 3/20 files done
```

**Concurrent progress tracking:**
With parallel processing, multiple files report progress simultaneously. Use an `AtomicCounter` for the completed file count:

```python
import threading

class AtomicCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
            return self._value
    
    @property
    def value(self):
        with self._lock:
            return self._value

# In sync():
completed = AtomicCounter()

def _emit(path, stage, file_idx, total, chunks=0, error=""):
    if progress_cb is not None:
        progress_cb(FileProgress(
            path=path,
            total=total,
            current=completed.increment() if stage in ("Done", "Error") else completed.value,
            stage=stage,
            chunks=chunks,
            error=error,
        ))
```

**Immediate per-file embedding storage:** Each file's embeddings are stored to Qdrant as soon as that file completes all pipeline stages — no cross-file batching. This is already the case in `ingest_text()` (line 724-730) and is confirmed in `ingest_prechunked()`. The "Storing" stage visible in the progress display confirms when embeddings are persisted.

## Architecture Communication Patterns

### Component Interaction Diagram

```
sync.py (orchestrator)
  │
  ├── Phase 1: List Sources
  │   └── Source.list_files() → [SourceFile]
  │
  ├── Phase 2: Hash Reconciliation
  │   └── _get_stored_hashes() → {path: hash}
  │
  ├── Phase 3: Concurrent Processing
  │   └── asyncio.gather(*tasks, return_exceptions=True)
  │       │
  │       ├── Task 1: _process_file(path, "add", progress_cb)
  │       │   ├── source.download() → local_path
  │       │   │   └── _emit("Converting", 70%)
  │       │   ├── _ingest_file(progress_cb=...) 
  │       │   │   ├── chunk_file(include_doc=True) → Docling API (1 call)
  │       │   │   │   └── _emit("Converting", 70%)
  │       │   │   │   └── OR fallback: parse_file() + create_chunks()
  │       │   │   └── engine.ingest_prechunked(progress_cb=...)
  │       │   │       ├── dedup_chunks()
  │       │   │       ├── ContextGenerator.enrich_chunks()
  │       │   │       │   └── _emit("Context", 73%)
  │       │   │       ├── MetadataExtractor.extract_batch()
  │       │   │       │   └── _emit("Metadata", 74%)
  │       │   │       ├── EmbeddingService.embed() → dense + sparse
  │       │   │       │   └── _emit("Embedding", 75%)
  │       │   │       └── Qdrant.upsert()  ← IMMEDIATE PER-FILE
  │       │   │           └── _emit("Storing", 90%)
  │       │   └── _emit("Done", chunks=N)
  │       │
  │       ├── Task 2: _process_file(path, "update", progress_cb)
  │       │   ├── engine.delete_by_source()
  │       │   │   └── _emit("Deleting")
  │       │   └── (same as add)
  │       │
  │       └── Task N: _process_file(path, "delete", progress_cb)
  │           └── engine.delete_by_source()
  │
  └── Phase 4: Tally Results → SyncStats
```

### Data Flow

1. **Source → Sync:** `SourceFile` objects (path, size, modified_at)
2. **Sync → Docling:** File path or bytes → chunks + markdown (single API call)
3. **Docling → Pipeline:** Chunks + markdown → context enrichment → metadata → embeddings
4. **Pipeline → Qdrant:** Points (id, vectors, payload) → upsert
5. **Qdrant → Sync:** Stored hashes for reconciliation

### Error Propagation

```
Docling fails → fallback to parse_file() + create_chunks()
  └── Both fail → log error, skip file, continue

Qdrant fails → retry with backoff (built-in qdrant_client)
  └── Still fails → log error, skip file, continue

Embedding fails → retry with backoff (tenacity)
  └── Still fails → log error, skip file, continue

Any exception in _process_file → caught, logged, returned as ("error", path, msg)
  └── SyncStats.errors accumulates all errors
  └── Sync continues processing remaining files
```

## Files to Modify

| File | Changes |
|------|---------|
| `memex/engine/sources/sync.py` | Rewrite `sync()` for parallel processing; `_ingest_file()` gets `progress_cb` param and hybrid-aware path; add `AtomicCounter` |
| `memex/engine/core/pipeline.py` | Add `ingest_prechunked()` with `progress_cb` param; confirm `ingest_text()` already stores immediately |
| `memex/engine/core/config.py` | Add `MAX_CONCURRENT_SYNC` constant |
| `config.yaml` / `config.example.yaml` | Add `ingestion.max_concurrent_sync` |
| `memex/cli.py` | Update `_on_progress` to display fine-grained pipeline stages (Context, Metadata, Embedding, Storing) |

## Edge Cases

### 1. Docling Chunker Fails or Returns Empty
- If `chunk_file()` raises an exception: fall back to `parse_file()` + `create_chunks()`
- If `chunk_file()` returns empty chunks: fall back to `parse_file()` + `create_chunks()`
- If both fail: log error, skip file, continue with next file

### 2. Concurrent Qdrant Writes
- Qdrant handles concurrent upserts internally (per-point locking)
- Multiple workers can safely write to the same collection
- Point IDs are deterministic (`uuid5(source_identifier, chunk_index)`), so re-ingesting the same file overwrites cleanly

### 3. Memory Pressure with Millions of Files
- Process one source at a time (sources are typically independent)
- Within a source, semaphore limits concurrent tasks to `max_concurrent_sync`
- Tasks are lightweight (coroutines doing I/O), not heavy objects
- If a single source has millions of files, the task list itself could be large. Consider batching: process files in chunks of 10K, run batch, then next batch.

### 4. Partial Failure
- Some files fail, others succeed: track per-file errors, report in `SyncStats.errors`
- Do not abort the entire sync on individual file failures
- Return partial results: `stats.added`, `stats.changed`, etc. reflect successful operations only

### 5. Crash Recovery
- Current sync has no checkpointing (unlike `IngestionOrchestrator`)
- On crash, partially processed files may have:
  - Old chunks deleted but new chunks not yet upserted
  - New chunks partially upserted
- **Mitigation:** Delete-then-upsert is atomic per-file. If crash happens between delete and upsert, the file's chunks are missing. Re-running sync will re-ingest them (content hash check will detect they're missing).
- **Future improvement:** Add checkpointing to sync (out of scope for this change).

### 6. Source Directory Modified During Sync
- Files could be added/removed between listing phase and processing phase
- **Acceptable:** Next sync catches the changes
- **Not acceptable:** Crash or data corruption — neither occurs with current design

### 7. Docling Service Overloaded
- With parallel requests, Docling could get overwhelmed
- **Mitigation:** `max_concurrent_sync` limits concurrent files. Each file makes at most 1 Docling call (down from 2). Docling's own concurrency handling applies.
- **Tuning:** Start with `max_concurrent_sync=8`, increase based on Docling GPU utilization

### 8. Qdrant Connection Drops
- Wrap Qdrant calls in try/except with retry logic
- `qdrant_client` has built-in retry via `QdrantClient(timeout=..., max_retries=...)`
- If Qdrant is unreachable: log error, skip file, continue

### 9. Embedding Service Overloaded
- Parallel files all call Ollama for embedding simultaneously
- **Mitigation:** `EMBED_BATCH_SIZE` (default 64) limits per-file batch size. Ollama handles concurrent requests.
- **Future improvement:** Add a global embedding semaphore across all concurrent files

### 10. File Modified Between Hash Check and Ingest
- Race window: file changes after content hash is computed but before ingest completes
- **Acceptable:** The ingested version will have the old content hash. Next sync will detect the change and re-ingest.
- **Not acceptable:** Corrupted data — does not occur because hash is computed before ingest.

### 11. Same File Processed by Two Sync Operations
- If user runs `memex sync` twice in parallel, the same file could be processed by both
- **Mitigation:** Point IDs are deterministic (`uuid5(source, idx)`), so both will upsert to the same points. No corruption, just wasted work.
- **Future improvement:** File-level locking via Redis (out of scope).

## Testing

1. **Unit test:** Mock Docling/Qdrant, verify concurrent processing with `asyncio.Semaphore`
2. **Integration test:** Small corpus (10-20 files), verify all files ingested correctly with parallel sync
3. **Performance test:** Compare sequential vs parallel sync time on same corpus
4. **Regression test:** Verify sync with `chunking.strategy=recursive` still works (no double-call issue)
5. **Dry-run test:** Verify `--dry-run` mode works with parallel processing

## Migration

- No migration needed — this is a performance optimization, not a data format change
- Existing collections are compatible
- New config key `ingestion.max_concurrent_sync` defaults to 8, backward-compatible
