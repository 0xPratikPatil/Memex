# Qdrant Payload Fixes — Enterprise-Grade Design

**Date:** 2026-07-29
**Status:** Approved
**Scope:** Bug fixes, metadata quality, testing, monitoring

---

## Problem Statement

Analysis of Qdrant payloads at `/tmp/payload` revealed multiple issues:

1. **`structural.total_chunks` always 0** — nested metadata has incorrect value
2. **`_fallback_per_chunk` wrong chunk_index** — uses batch-local index instead of global
3. **Field duplication** — `chunk_index`, `total_chunks`, `section_header` stored twice
4. **No metadata validation** — extracted metadata not validated before storage
5. **No testing** — no unit/integration tests for metadata extraction
6. **No monitoring** — no metrics for extraction quality

## Research Findings

Based on analysis of production RAG systems for legal/financial documents:

- **Contextual Retrieval** (Anthropic): Prepending document-level summaries to chunks reduces retrieval failures by 49%. Our implementation aligns with this approach.
- **Metadata Enriched RAG** (arxiv 2603.19251): Metadata-enriched chunks improve span recall by 320% on M&A contracts.
- **Structure-Aware Chunking**: Section/subsection-based retrieval achieves highest recall (0.47).
- **Testing Strategy** (3-layer): Unit tests (every save), Integration tests (every PR), Eval tests (nightly).

## Design

### Part 1: Bug Fixes

#### Bug 1: `structural.total_chunks` always 0

**File:** `rag/services/metadata_extractor.py`

**Root cause:** `extract_structural()` reads `chunk.get("total_chunks", 0)` but the chunk dict doesn't have `total_chunks` yet — it's only added later in `ingest_text()`.

**Fix:** Add `total_chunks` to chunk dict before calling `extract_structural()`:

```python
# In _extract_batch_metadata() line 446:
chunk_with_index = {**chunk, "chunk_index": batch_start + i, "total_chunks": len(chunks)}

# In _fallback_per_chunk() line 470:
chunk_with_index = {**chunk, "chunk_index": batch_start + i, "total_chunks": len(chunks)}
```

#### Bug 2: `_fallback_per_chunk` wrong chunk_index

**File:** `rag/services/metadata_extractor.py`

**Root cause:** Line 470 uses `i` (index within batch) instead of global index. The method doesn't receive `batch_start`.

**Fix:** Pass `batch_start` to `_fallback_per_chunk()` and use it:

```python
def _fallback_per_chunk(self, batch, document_text, source_identifier, doc_type, batch_start=0):
    ...
    chunk_with_index = {**chunk, "chunk_index": batch_start + i, "total_chunks": len(chunks)}
```

#### Bug 3: Field duplication

**File:** `rag/services/metadata_extractor.py`

**Root cause:** `extract_structural()` returns `chunk_index`, `total_chunks`, `section_header` which duplicate top-level payload fields.

**Fix:** Remove these 3 fields from `extract_structural()` return dict:

```python
def extract_structural(self, chunk, source_identifier):
    content = chunk.get("content", "")
    header = chunk.get("section_header", "")
    # ... regex logic ...
    return {
        # REMOVED: chunk_index, total_chunks, section_header
        "heading_level": heading_level,
        "is_list": is_list,
        "is_code": is_code,
        "is_table": is_table,
        "char_count": len(content),
        "word_count": len(content.split()),
        "link_count": len(links),
        "email_count": len(emails),
        "phone_count": len(phones),
    }
```

### Part 2: Metadata Validation Layer

Add validation after extraction to ensure quality:

```python
def validate_metadata(meta: dict) -> dict:
    """Validate and normalize extracted metadata."""
    # Schema validation
    assert isinstance(meta.get("entities", {}), dict)
    assert isinstance(meta.get("topics", []), list)
    assert isinstance(meta.get("keywords", []), list)
    
    # Value validation
    if "doc_type" in meta:
        valid_types = ["report", "article", "policy", "contract", "other"]
        if meta["doc_type"] not in valid_types:
            meta["doc_type"] = "other"
    
    # Completeness checks
    if not meta.get("language"):
        meta["language"] = "en"  # default
    
    return meta
```

### Part 3: Contextual Retrieval (No Changes)

Research confirms our approach is correct:
- Document-level summary prepended to chunks = Summary-Augmented Chunking (SAC)
- Generic summaries outperform expert-guided
- The `context_prefix` + `content` dual storage is by design for dual-vector approach

**No changes needed** — our implementation aligns with Anthropic's Contextual Retrieval.

### Part 4: Testing Strategy

#### Layer 1: Unit Tests (run every save)

- `test_structural_metadata.py` — verify `extract_structural()` returns correct fields
- `test_metadata_validation.py` — verify validation logic
- `test_chunk_index_consistency.py` — verify chunk_index is correct in all paths

#### Layer 2: Integration Tests (run every PR)

- `test_ingestion_pipeline.py` — verify full ingestion with metadata
- `test_contextual_retrieval.py` — verify context prefix is correct
- `test_qdrant_payload.py` — verify payload structure matches schema

#### Layer 3: Eval Tests (run nightly)

- `test_retrieval_quality.py` — verify recall@10 >= 0.85
- `test_metadata_filtering.py` — verify filters work correctly

### Part 5: Production Monitoring

**Metrics to track:**
- Extraction success rate per field
- Field completeness percentage
- Structural metadata consistency (no duplicates)
- Contextual retrieval hit rate
- Retrieval recall@10

**Alerts:**
- Extraction failure rate > 5%
- Missing metadata fields > 10%
- Structural.total_chunks always 0 (current bug)

## Implementation Order

1. Fix `structural.total_chunks` bug
2. Fix `_fallback_per_chunk` chunk_index bug
3. Remove field duplication from `extract_structural()`
4. Add metadata validation layer
5. Add unit tests
6. Add integration tests
7. Run lint and typecheck
8. Commit changes

## Verification

After implementation:
1. Run `make test` — all unit tests pass
2. Run `make lint` — no lint errors
3. Ingest a test document and verify Qdrant payload:
   - `structural.total_chunks` > 0
   - `structural.chunk_index` matches top-level `chunk_index`
   - No duplicate fields
   - Metadata fields are valid

## References

- Anthropic Contextual Retrieval: https://www.anthropic.com/engineering/contextual-retrieval
- Metadata Enriched RAG: https://arxiv.org/abs/2603.19251
- Financial Report Chunking: https://arxiv.org/abs/2402.05131
- RAG Testing Strategy: https://www.learnwithparam.com/blog/testing-evaluating-rag-pipelines
