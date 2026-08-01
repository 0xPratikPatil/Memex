# Contextual Retrieval Fix + Search Parallelism

Date: 2026-08-01
Status: approved

## Problem

`rag_query` returns empty `context_prefix` for most chunks even though
`contextual_retrieval.enabled` is `true` and strategy is `summary`.

Additionally, hybrid search runs Qdrant queries sequentially despite
dense, sparse, HyDE, and multi-query searches being independent.

## Root Cause Analysis

### Bug 1: Inverted vector assignment (`pipeline.py:ingest_text`)

`enrich_chunks` prepends `[Context: ...]` to `content`, so after enrichment:

- `chunk["content"]` = `"[Context: Q3 results] Revenue grew 20%."` (enriched)
- `strip_context_prefix(chunk["content"])` = `"Revenue grew 20%."` (raw)

Current assignment (lines 623–638):

```
dense_vecs     = embed(chunk["content"])            # enriched → "dense"
contextual_vecs = embed(strip_context_prefix(...))   # raw      → "contextual_dense"
```

At search time (line 778):
```
dense_vector_name = "contextual_dense" if effective_contextual else "dense"
```

So when contextual search is ON → uses `contextual_dense` → embeds raw text (wrong).
When OFF → uses `dense` → embeds enriched text (wrong).

Complete inversion.

### Bug 2: Single-batch summary bypass (`context.py:enrich_chunks`)

When strategy is "summary" and the document has ≤ `batch_size` chunks
(default 10), only ONE batch is created. The concurrent path at line 140
is skipped (`len(all_batches) > 1` is false). Code falls through to the
sequential path (line 165) which checks:

```python
if strategy == "surrounding":
    contexts = _batch_context_from_surrounding(...)
else:
    contexts = [_context_from_header(...)]   # ← "summary" lands here
```

Result: all chunks get `_context_from_header()` instead of
`_batch_context_from_summary()`. Since most chunks lack section headers,
`context_prefix` is empty and `content` is unmodified.

### Bug 3: No fallback chain in summary strategy

`_batch_context_from_summary` has three silent failure modes:

1. **Empty summary**: if `generate_document_summary` fails or returns empty,
   the method returns `[""] * len(batch)` immediately (line 188–189).

2. **Parse failure**: if the LLM response does not contain numbered lines
   (common with small models like qwen2.5:1.5b), `re.findall` matches
   nothing. Lines array is empty, padded with empty strings.

3. **LLM call failure**: if `self._chat()` raises, the exception propagates
   to `_process_summary_batch` and kills the ThreadPoolExecutor task.
   `pool.map()` re-raises and the entire ingest fails.

No per-chunk fallback attempt exists. `_context_from_summary` (per-chunk
LLM call) exists at line 66 but is never invoked by `enrich_chunks`.

### Bug 4: No startup check for vector compatibility

Existing collections from before contextual retrieval was enabled lack
the `contextual_dense` vector. Searches using the `contextual_dense`
vector name against such collections silently return no hits or degraded
results with zero context_prefix.

## Design

### 1A: Fix inverted vector assignment

File: `memex/engine/core/pipeline.py`, `ingest_text` method

```python
chunk_texts    = [c["content"] for c in raw_chunks]          # enriched (has prefix)
raw_texts      = [strip_context_prefix(c["content"]) for c]  # raw (no prefix)

# In thread pool: embed raw as "dense", embed enriched as "contextual_dense"
dense_vecs     = pool.submit(embed, raw_texts)                # raw → "dense"
contextual_vecs = pool.submit(embed, chunk_texts)             # enriched → "contextual_dense"
```

The `dense_vector_name` selection at lines 778/943 stays unchanged.

### 1B: Fix enrich_chunks routing

File: `memex/engine/ingestion/context.py`, `enrich_chunks` method

Restructure so "summary" is always handled regardless of batch count:

```python
if strategy == "summary":
    if len(all_batches) > 1:
        # concurrent ThreadPoolExecutor for multiple batches
        ...
    else:
        # single batch, no threading overhead
        contexts = self._batch_context_from_summary(all_batches[0][0], document_summary)
        # apply to chunks
    return enriched
```

### 1C: Add resilience chain

File: `memex/engine/ingestion/context.py`, `_batch_context_from_summary` method

Each chunk's context goes through a fallback chain:

```
1. Batch LLM → re.findall numbered lines
2. Parse failed? → per-chunk _context_from_summary(chunk, summary)
3. Per-chunk failed or summary empty? → _context_from_header(section_header)
4. No header? → "" (empty, no prefix added)
```

Implementation: return a list of (context, fallback_info) from
`_batch_context_from_summary`, then iterate in `enrich_chunks` to fill
gaps with per-chunk fallbacks.

```python
def _batch_context_from_summary(self, batch, summary):
    if not summary:
        return [("", "empty_summary")] * len(batch)
    try:
        response = self._chat(prompt)
        lines = re.findall(...)
        if len(lines) < len(batch):
            logger.debug("Parse mismatch: got %d lines for %d chunks", len(lines), len(batch))
        contexts = []
        for i in range(len(batch)):
            if i < len(lines) and lines[i].strip():
                contexts.append((f"[Context: {lines[i].strip()}]", "batch"))
            else:
                contexts.append(("", "parse_gap"))
        return contexts
    except Exception:
        logger.warning("Batch context generation failed", exc_info=True)
        return [("", "llm_error")] * len(batch)
```

Then `enrich_chunks` fills gaps:

```python
for chunk, (ctx, source) in zip(batch, contexts, strict=True):
    if not ctx:
        # Per-chunk fallback
        try:
            ctx = self._context_from_summary(chunk["content"], document_summary)
        except Exception:
            ctx = self._context_from_header(chunk.get("section_header", ""))
    # ... apply ctx to chunk
```

### 1D: Startup vector compatibility check

File: `memex/engine/core/pipeline.py`, `_ensure_collection` method

After confirming collection exists and contextual retrieval is enabled,
check that `contextual_dense` is present in the collection's vector config:

```python
if config.ENABLE_CONTEXTUAL_RETRIEVAL:
    try:
        info = qdrant.get_collection(config.COLLECTION_NAME)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict) and "contextual_dense" not in vectors:
            logger.warning(
                "contextual_retrieval is ON but collection '%s' has no "
                "contextual_dense vector. Existing chunks will not benefit "
                "from contextual search. Re-ingest affected documents.",
                config.COLLECTION_NAME,
            )
    except Exception:
        pass  # Don't block startup on metadata checks
```

### 1E: Robust parse + fallback for batch context

The batch prompt already asks for numbered output. Add a second parser
that handles non-numbered responses (e.g., line-by-line output) as a
secondary attempt before the per-chunk fallback.

### 2A: Parallel dense + sparse search

File: `memex/engine/core/pipeline.py`, `hybrid_search` method

Replace sequential Qdrant calls (lines 794–812):

```python
# Before: sequential
dense_hits = qdrant.query_points(using=dense_vector_name, ...)
sparse_hits = qdrant.query_points(using="sparse", ...)

# After: parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    dense_future  = pool.submit(qdrant.query_points, ..., using=dense_vector_name, ...)
    sparse_future = pool.submit(qdrant.query_points, ..., using="sparse", ...)
    dense_hits  = dense_future.result()
    sparse_hits = sparse_future.result()
```

Expected latency drop: ~40% (from sum to max) for the Qdrant fetch phase.

### 2B: Parallel HyDE + multi-query with main search

File: `memex/engine/core/pipeline.py`, `hybrid_search` method

HyDE dense search and multi-query paraphrase searches are independent of
each other and of the main dense/sparse searches. Start them in the same
thread pool after embedding:

```python
# All search tasks in one pool (max 3 parallel Qdrant calls)
# Task 1: dense search
# Task 2: sparse search
# Task 3+ (if expansion): HyDE + paraphrase searches (submitted after embed)

# After RRF fusion of main dense+sparse:
if expansion:
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = []
        if expanded_query.hyde_vector:
            futures.append(("hyde", pool.submit(qdrant.query_points, ...)))
        for para_vec in para_vecs:
            futures.append(("para", pool.submit(qdrant.query_points, ...)))
        for name, future in futures:
            hits = future.result()
            _merge_hits(...)
```

### 2C: Contextual embedding in same pool as dense/sparse

File: `memex/engine/core/pipeline.py`, `ingest_text` method

Instead of sequential embedding (dense+sparse → wait → contextual):

```python
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    dense_future       = pool.submit(embed, raw_texts)      # "dense"
    sparse_future      = pool.submit(sparse_embed, chunk_texts)
    contextual_future  = pool.submit(embed, chunk_texts)    # "contextual_dense"
    dense_vecs         = dense_future.result()
    sparse_vecs        = sparse_future.result()
    contextual_vecs    = contextual_future.result()
```

### 2D: MMR search — same dense/sparse parallelism

File: `memex/engine/core/pipeline.py`, `mmr_search` method

MMR only does dense search currently. Keep as-is since there's only one
Qdrant call; nothing to parallelize (MMR inherently needs the dense
results first for candidate selection).

## Impact

- **Backward compat**: Existing collections need re-ingest for correct
  contextual retrieval (vectors were inverted). The startup warning
  (1D) communicates this.
- **Tests affected**: `test_contextual_retrieval.py`, `test_contextual_ingest.py`,
  `test_expansion_search.py`, `test_server.py`
- **No config changes**: Same yaml keys, same schema.
- **No new dependencies**: `concurrent.futures` already used elsewhere.

## Testing

1. Unit: `_batch_context_from_summary` fallback chain — mock LLM failures
2. Unit: `enrich_chunks` single-batch summary path — verify it routes correctly
3. Integration: ingest small doc (<10 chunks), verify all chunks have non-empty `context_prefix`
4. Integration: search with `use_contextual_search=true`, verify results differ from `false`
5. Performance: search latency before/after parallelism
