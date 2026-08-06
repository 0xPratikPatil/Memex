# Design: Metadata Deduplication + Redis/Caching Cleanup

**Date**: 2026-08-06
**Status**: Draft
**Scope**: Two focused fixes — entity date deduplication and Redis/caching clarity

---

## Problem Statement

### 1. `dates` duplication in Qdrant payload

The Qdrant payload stores date information twice:
- `entities.dates` — LLM-extracted plain strings (e.g., `["2026-01-15", "January 2026"]`)
- `dates` (top-level) — regex-extracted structured dicts (e.g., `[{"value": "2026-01-15", "format": "iso"}]`)

This wastes storage, confuses filtering, and the LLM extraction prompt lacks format guidance producing inconsistent date strings.

### 2. Redis/caching confusion

`config.yaml` has `redis_url: "redis://localhost:6379/0"` but Redis is commented out in `docker-compose.yml`. The code silently falls back to in-memory cache, but the config suggests Redis should be running. Users see the `redis_url` and think something is broken.

---

## Design

### Part 1: Consolidate Date Extraction

**Decision**: Keep LLM-based dates inside `entities`, remove regex-based top-level `dates`.

**Rationale**: User prefers LLM extraction. The LLM can understand context ("next Tuesday", "Q3 2025", "the following year") that regex cannot. Removing regex eliminates duplication.

#### Changes to `memex/engine/metadata/extractor.py`

**1. Improve entity extraction prompt (single-chunk, line 102-106)**

Current:
```python
prompt = (
    "Extract named entities from this text. Return JSON with keys: "
    "people, organizations, dates, locations, products. "
    "Each value is a list of unique strings. Only output JSON.\n\n"
    f"Text: {text[:1000]}"
)
```

New:
```python
prompt = (
    "Extract named entities from this text. Return JSON with these keys:\n"
    "- people: full names of people mentioned\n"
    "- organizations: company, agency, or institution names\n"
    "- locations: cities, countries, addresses, or place names\n"
    "- products: product names, model numbers, or brand names\n"
    '- dates: all dates mentioned, as readable strings (e.g. "January 15, 2026", "Q3 2025", "2024"). '
    'Normalize partial dates (e.g. "Jan" -> "January"). Deduplicate: if the same date appears '
    "multiple ways, keep the most complete form.\n\n"
    "Each value is a list of unique strings. Only output JSON.\n\n"
    f"Text: {text[:1000]}"
)
```

Key improvements:
- Explicit instructions per entity type
- Date format guidance: readable strings, not raw regex matches
- Dedup instruction: normalize multiple representations of the same date
- Examples in prompt for dates

**2. Improve batch entity prompt (line 398)**

Current:
```python
tasks.append("entities (JSON with keys: people, orgs, dates, locations, products — each a list)")
```

New:
```python
tasks.append(
    "entities (JSON object with keys: people (full names), organizations (companies/agencies), "
    "locations (places), products (product names), dates (readable date strings like "
    '"January 15, 2026" or "Q3 2025" — normalize and deduplicate). Each value is a list of unique strings.'
)
```

Key improvements:
- Consistent key names (`organizations` not `orgs`)
- Date format guidance in batch prompt
- Dedup instruction

**3. Improve topic extraction prompt (line 169-172)**

Current:
```python
prompt = (
    f"Extract up to {config.MAX_TOPICS_PER_CHUNK} topic labels from this text. "
    "Return as JSON array of strings. Only output JSON.\n\n"
    f"Text: {text[:1000]}"
)
```

New:
```python
prompt = (
    f"Extract up to {config.MAX_TOPICS_PER_CHUNK} topic labels from this text. "
    "Topics should be short noun phrases (2-4 words) that describe the main subjects. "
    'Avoid generic labels like "information" or "details". '
    "Return as JSON array of strings. Only output JSON.\n\n"
    f"Text: {text[:1000]}"
)
```

Key improvements:
- Guidance on topic format (short noun phrases)
- Anti-pattern: avoid generic labels

**4. Remove regex-based date extraction**

- Delete `_DATE_PATTERNS` list (lines 22-29)
- Delete `extract_dates()` method (lines 202-215)
- Remove call in `extract_all()` (lines 78-80): `dates = self.extract_dates(...)` / `metadata["dates"] = dates`
- Remove call in `_extract_batch_metadata()` (line 433): `meta["dates"] = self.extract_dates(...)`
- Remove call in `_fallback_per_chunk()` (line 463): `meta["dates"] = self.extract_dates(...)`

**5. Remove `_EMAIL_RE`, `_URL_RE`, `_PHONE_RE` if only used by `extract_dates`**

Check: these are also used by `extract_structural()` (lines 313-315). Keep them.

#### Changes to `memex/engine/core/pipeline.py`

1. **Search result mapping** (lines 920 and 1067): Change `"dates": payload.get("dates", [])` to `"dates": payload.get("entities", {}).get("dates", [])` so search results read dates from the `entities` dict.

The `**(chunk.get("metadata", {}))` spread (line 690) already puts `entities` into the payload. After removing the regex `dates`, the payload will only have `entities.dates`.

#### Changes to `memex/engine/retrieval/filter.py`

Remove `FieldInfo(name="dates", ...)` from the known fields list (line 164). Dates are now nested inside `entities` and accessible via `entities.dates` filtering.

#### Changes to `memex/mcp/server.py`

Update MCP tool documentation (line 451) to remove `"dates"` from the top-level filter fields list. Dates are now inside `entities`.

---

### Part 2: Redis/Caching Clarity

**Decision**: Make Redis explicitly opt-in. In-memory cache works out of the box.

#### Changes to `config.yaml`

```yaml
caching:
  enabled: true                            # in-memory LRU cache (works out of the box)
  # redis_url: "redis://localhost:6379/0"  # uncomment for persistent cross-restart cache
  redis_url: ""                            # empty = in-memory only (default)
  ttl_embedding: 86400                     # 24h — embedding cache
  ttl_search: 3600                         # 1h — search result cache
  ttl_parse: 604800                        # 7d — parsed document cache
  ttl_expansion: 21600                     # 6h — query expansion cache
  max_memory_mb: 256                       # in-memory LRU cap
  eviction_policy: allkeys-lru             # Redis eviction policy (unused when redis_url empty)
```

Key change: `redis_url: ""` (empty) instead of `"redis://localhost:6379/0"`.

#### Changes to `config.example.yaml`

Same change — `redis_url: ""` with comment explaining Redis is optional.

#### Changes to `memex/engine/utils/cache.py`

**1. Skip Redis entirely when `redis_url` is empty (in `_get_redis()`, line 73-112)**

Current: Always tries to import redis and connect, fails silently.

New: Check `config.REDIS_URL` early:
```python
def _get_redis() -> Any:
    global _redis
    if _redis is not None:
        return _redis
    if not config.ENABLE_CACHE:
        return None
    with _redis_lock:
        if _redis is not None:
            return _redis
        # Skip Redis entirely when URL is empty
        if not config.REDIS_URL:
            logger.info("Cache backend: in-memory LRU (set caching.redis_url for Redis)")
            _redis = True
            return _redis
        try:
            import redis

            pool = redis.ConnectionPool.from_url(
                config.REDIS_URL,
                decode_responses=True,
                max_connections=10,
            )
            _redis = redis.Redis(
                connection_pool=pool,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            _redis.ping()
            logger.info("Cache backend: Redis at %s", config.REDIS_URL)
            return _redis
        except ImportError:
            logger.warning("redis package not installed, using in-memory cache")
            _redis = True
            return _redis
        except Exception as exc:
            logger.warning("Redis unavailable, using in-memory cache: %s", exc)
            _redis = True
            return _redis
```

Key improvement: When `redis_url` is empty, skips the import entirely and logs clearly.

---

## Files Modified

| File | Change |
|------|--------|
| `memex/engine/metadata/extractor.py` | Improve prompts, remove regex date extraction |
| `memex/engine/retrieval/filter.py` | Remove `dates` from top-level filter fields |
| `memex/mcp/server.py` | Update docs to remove `dates` from filter fields |
| `memex/mcp/schemas.py` | Update `metadata_filter` description to remove `dates` |
| `config.yaml` | `redis_url: ""` (empty default) |
| `config.example.yaml` | `redis_url: ""` (empty default) |
| `memex/engine/utils/cache.py` | Skip Redis when URL empty, log clearly |

---

## Testing

1. **Entity extraction**: Run ingestion on a document with dates. Verify `entities.dates` contains normalized, deduplicated date strings. Verify no top-level `dates` field in payload.
2. **Topic extraction**: Verify topics are short noun phrases, not generic labels.
3. **In-memory cache**: Start the app with `redis_url: ""`. Verify cache works. Verify log says "Cache backend: in-memory LRU".
4. **Redis (optional)**: Set `redis_url: "localhost:6379"`. Verify Redis connection. Verify log says "Cache backend: Redis at localhost:6379".

---

## Risks

- **LLM date format**: The LLM may still produce inconsistent date strings. The prompt guidance should reduce this, but monitoring may be needed.
- **Filtering**: Removing `dates` from top-level filtering breaks any code that filters by `dates`. The `entities.dates` nesting should provide equivalent functionality.
