# Implementation Plan: Metadata Dedup + Redis Cleanup

**Spec**: `docs/superpowers/specs/2026-08-06-metadata-dedup-redis-cleanup-design.md`

## Phase 1: Entity Extraction Prompts + Date Dedup

### Task 1.1: Improve entity extraction prompts in `extractor.py`

**File**: `memex/engine/metadata/extractor.py`

1. Update `extract_entities()` prompt (line 102-106):
   - Add per-entity-type instructions
   - Add date format guidance + dedup instruction
   - Keep `dates` in entity keys (LLM-based)

2. Update batch entity prompt (line 398):
   - Change `orgs` → `organizations` for consistency
   - Add date format guidance

3. Update `extract_topics()` prompt (line 169-172):
   - Add "short noun phrases (2-4 words)"
   - Add "avoid generic labels like 'information'"

### Task 1.2: Remove regex date extraction

**File**: `memex/engine/metadata/extractor.py`

1. Delete `_DATE_PATTERNS` list (lines 22-29)
2. Delete `extract_dates()` method (lines 202-215)
3. Remove calls in `extract_all()` (lines 78-80)
4. Remove calls in `_extract_batch_metadata()` (line 433)
5. Remove calls in `_fallback_per_chunk()` (line 463)

### Task 1.3: Update search result mapping

**File**: `memex/engine/core/pipeline.py`

1. Line 920: Change `"dates": payload.get("dates", [])` → `"dates": payload.get("entities", {}).get("dates", [])`
2. Line 1067: Same change for MMR search

### Task 1.4: Update filter fields

**File**: `memex/engine/retrieval/filter.py`

1. Line 164: Remove `FieldInfo(name="dates", type="list", values=[], count=0)`

### Task 1.5: Update MCP docs and schema

**File**: `memex/mcp/server.py`

1. Line 428: Remove `dates` from the list of filterable fields
2. Line 451: Update filter field examples

**File**: `memex/mcp/schemas.py`

1. Line 78: Update `metadata_filter` description to remove `dates`

## Phase 2: Redis/Caching Clarity

### Task 2.1: Update config defaults

**File**: `config.yaml`

1. Change `redis_url: "redis://localhost:6379/0"` → `redis_url: ""`
2. Add comment: `# uncomment for persistent cross-restart cache`

**File**: `config.example.yaml`

1. Same change

### Task 2.2: Update cache initialization

**File**: `memex/engine/utils/cache.py`

1. In `_get_redis()`: Add early check for empty `redis_url`
2. Log clear message: "Cache backend: in-memory LRU (set caching.redis_url for Redis)"

## Phase 3: Verify + Test

### Task 3.1: Run lint and typecheck

```bash
make lint
make typecheck
```

### Task 3.2: Run unit tests

```bash
make test
```

### Task 3.3: Manual verification

1. Ingest a document → verify `entities.dates` has normalized strings
2. Verify no top-level `dates` in Qdrant payload
3. Start app with `redis_url: ""` → verify log says "in-memory LRU"
4. Search → verify dates appear in results from `entities.dates`
