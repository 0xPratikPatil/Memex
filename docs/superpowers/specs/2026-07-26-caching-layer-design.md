# Caching Layer Design

**Date**: 2026-07-26
**Status**: Draft
**Author**: Opencode

---

## Problem Statement

The current RAG pipeline has several redundant computations:

1. **Repeated embeddings**: Identical queries get re-embedded every time (Ollama API calls).
2. **Repeated searches**: Identical queries hit Qdrant and reranker every time.
3. **Repeated document parsing**: The same file gets re-parsed by Docling even if unchanged (partially addressed by content hash check in `ingest_text`).
4. **LLM expansion calls**: Query expansion calls (HyDE, multi-query, rewrite) are expensive and repeat for similar queries.

Each Ollama embedding call takes ~50-200ms, and Qdrant search + reranking takes ~100-500ms. For repeated queries (common in conversational RAG), this adds unnecessary latency.

---

## Solution Overview

Implement a **cache-aside** caching layer with Redis:

1. **Embedding Cache**: Cache query → embedding mappings.
2. **Search Result Cache**: Cache query hash → search results.
3. **Document Parse Cache**: Cache file hash → Docling result (extends existing hash check).
4. **TTL-based Invalidation**: Automatic expiration.
5. **Manual Invalidation**: Explicit cache clear on document delete/re-ingest.

```
┌──────────┐     ┌────────────┐     ┌──────────┐
│  Client   │────▶│ Cache Layer │────▶│ Pipeline  │
│  Query    │     │ (Redis)     │     │ (Qdrant + │
└──────────┘     └────────────┘     │  Ollama)  │
                      │              └──────────┘
                      │ hit → return cached
                      │ miss → compute + cache
```

---

## Architecture

### Cache Keys

| Cache Type | Key Format | TTL | Value |
|-----------|------------|-----|-------|
| Embedding | `emb:{model}:{text_hash}` | 24h | `list[float]` |
| Search Result | `search:{query_hash}:{top_k}:{source_filter}` | 1h | `list[dict]` |
| Document Parse | `parse:{file_hash}` | 7d | `ConversionResult` (serialized) |
| Query Expansion | `expand:{query_hash}:{strategy}` | 6h | `ExpandedQuery` (serialized) |

### Cache Flow

```
Query arrives
    │
    ▼
┌─────────────────┐
│ Check Embed Cache │
│ key: emb:{hash}   │
└──────┬──────────┘
       │
   hit │         miss
   │           │
   │           ▼
   │    ┌──────────────┐
   │    │ Compute embed  │
   │    │ via Ollama     │
   │    └──────┬───────┘
   │           │
   │           ▼
   │    ┌──────────────┐
   │    │ Store in cache │
   │    └──────┬───────┘
   │           │
   ▼           ▼
┌─────────────────┐
│ Check Search Cache│
│ key: search:{hash}│
└──────┬──────────┘
       │
   hit │         miss
   │           │
   │           ▼
   │    ┌──────────────┐
   │    │ Run Qdrant +   │
   │    │ Rerank         │
   │    └──────┬───────┘
   │           │
   │           ▼
   │    ┌──────────────┐
   │    │ Store in cache │
   │    └──────┬───────┘
   │           │
   ▼           ▼
Return results
```

---

## Files to Modify/Create

| File | Action | Purpose |
|------|--------|---------|
| `src/cache.py` | **Create** | Redis cache client, cache decorators, invalidation |
| `src/config.py` | **Modify** | Add Redis config and cache TTL settings |
| `src/pipeline.py` | **Modify** | Wrap embedding and search with cache |
| `src/docling_client.py` | **Modify** | Cache Docling parse results |
| `docker-compose.yml` | **Modify** | Add Redis service |
| `pyproject.toml` | **Modify** | Add `redis` dependency |
| `tests/unit/test_cache.py` | **Create** | Unit tests for cache operations |
| `tests/integration/test_cache_integration.py` | **Create** | Integration tests with Redis |

---

## Implementation Details

### 1. Configuration (`config.py` additions)

```python
# ── Cache Settings ──────────────────────────────────────────────────────────
ENABLE_CACHE: bool = _env_bool("ENABLE_CACHE", False)
REDIS_URL: str = _env("REDIS_URL", "redis://localhost:6379/0")

CACHE_TTL_EMBEDDING: int = _env_int("CACHE_TTL_EMBEDDING", 86400)     # 24h
CACHE_TTL_SEARCH: int = _env_int("CACHE_TTL_SEARCH", 3600)            # 1h
CACHE_TTL_PARSE: int = _env_int("CACHE_TTL_PARSE", 604800)            # 7d
CACHE_TTL_EXPANSION: int = _env_int("CACHE_TTL_EXPANSION", 21600)     # 6h

CACHE_MAX_MEMORY_MB: int = _env_int("CACHE_MAX_MEMORY_MB", 256)
CACHE_EVICTION_POLICY: str = _env("CACHE_EVICTION_POLICY", "allkeys-lru")
```

### 2. Cache Module (`src/cache.py`)

```python
"""Redis-backed caching layer for RAG pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
from functools import wraps
from typing import Any, Callable
from . import config

logger = logging.getLogger("cache")

_redis = None


def _get_redis():
    """Lazy-init Redis client."""
    global _redis
    if _redis is None:
        try:
            import redis
            _redis = redis.Redis.from_url(
                config.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            _redis.ping()
            logger.info("Connected to Redis at %s", config.REDIS_URL)
        except Exception as exc:
            logger.warning("Redis unavailable, caching disabled: %s", exc)
            _redis = None
    return _redis


def _hash_key(*parts: str) -> str:
    """Create a deterministic cache key hash."""
    combined = ":".join(str(p) for p in parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def get_cached(namespace: str, key_parts: str, ttl: int) -> Any | None:
    """Get a cached value by namespace and key parts."""
    if not config.ENABLE_CACHE:
        return None
    r = _get_redis()
    if r is None:
        return None
    try:
        cache_key = f"rag:{namespace}:{_hash_key(*key_parts.split(':'))}"
        data = r.get(cache_key)
        if data:
            logger.debug("Cache hit: %s", cache_key)
            return json.loads(data)
        logger.debug("Cache miss: %s", cache_key)
        return None
    except Exception as exc:
        logger.warning("Cache get failed: %s", exc)
        return None


def set_cached(namespace: str, key_parts: str, value: Any, ttl: int) -> None:
    """Store a value in cache with TTL."""
    if not config.ENABLE_CACHE:
        return
    r = _get_redis()
    if r is None:
        return
    try:
        cache_key = f"rag:{namespace}:{_hash_key(*key_parts.split(':'))}"
        data = json.dumps(value)
        r.setex(cache_key, ttl, data)
        logger.debug("Cache set: %s (ttl=%ds)", cache_key, ttl)
    except Exception as exc:
        logger.warning("Cache set failed: %s", exc)


def invalidate_namespace(namespace: str) -> int:
    """Invalidate all keys in a namespace. Returns count of deleted keys."""
    if not config.ENABLE_CACHE:
        return 0
    r = _get_redis()
    if r is None:
        return 0
    try:
        pattern = f"rag:{namespace}:*"
        keys = list(r.scan_iter(match=pattern, count=1000))
        if keys:
            r.delete(*keys)
            logger.info("Invalidated %d cache keys in namespace '%s'", len(keys), namespace)
        return len(keys)
    except Exception as exc:
        logger.warning("Cache invalidation failed: %s", exc)
        return 0


def invalidate_for_document(source_identifier: str) -> None:
    """Invalidate all caches related to a document."""
    invalidate_namespace("search")
    invalidate_namespace("parse")
    logger.info("Invalidated caches for document: %s", source_identifier)


def cache_embedding(text: str, embedding: list[float]) -> None:
    """Cache a text → embedding mapping."""
    set_cached("emb", text, embedding, config.CACHE_TTL_EMBEDDING)


def get_cached_embedding(text: str) -> list[float] | None:
    """Get cached embedding for text."""
    return get_cached("emb", text, config.CACHE_TTL_EMBEDDING)


def cache_search_results(query: str, top_k: int, source_filter: str | None, results: list[dict]) -> None:
    """Cache search results."""
    key = f"{query}:{top_k}:{source_filter or ''}"
    set_cached("search", key, results, config.CACHE_TTL_SEARCH)


def get_cached_search_results(query: str, top_k: int, source_filter: str | None) -> list[dict] | None:
    """Get cached search results."""
    key = f"{query}:{top_k}:{source_filter or ''}"
    return get_cached("search", key, config.CACHE_TTL_SEARCH)


def cache_parse_result(file_hash: str, result: dict) -> None:
    """Cache Docling parse result."""
    set_cached("parse", file_hash, result, config.CACHE_TTL_PARSE)


def get_cached_parse_result(file_hash: str) -> dict | None:
    """Get cached Docling parse result."""
    return get_cached("parse", file_hash, config.CACHE_TTL_PARSE)


def close() -> None:
    """Close Redis connection."""
    global _redis
    if _redis is not None:
        _redis.close()
        _redis = None
```

### 3. Pipeline Integration

**Embedding cache** (`pipeline.py`):

```python
def _dense_embed(self, text: str) -> list[float]:
    from src.cache import get_cached_embedding, cache_embedding

    cached = get_cached_embedding(text)
    if cached is not None:
        return cached

    # existing Ollama call
    embedding = self._dense_embed_batch([text])[0]

    cache_embedding(text, embedding)
    return embedding
```

**Search result cache** (`pipeline.py`):

```python
def hybrid_search(self, query, top_k=5, rerank=True, source_filter=None):
    from src.cache import get_cached_search_results, cache_search_results

    cached = get_cached_search_results(query, top_k, source_filter)
    if cached is not None:
        return cached

    # existing search logic
    results = ...  # existing computation

    cache_search_results(query, top_k, source_filter, results)
    return results
```

### 4. Docling Cache (`docling_client.py`)

```python
def parse_file(file_path_or_url: str) -> ConversionResult:
    from src.cache import get_cached_parse_result, cache_parse_result

    # Compute hash from URL/path
    file_hash = hashlib.sha256(file_path_or_url.encode()).hexdigest()[:16]

    cached = get_cached_parse_result(file_hash)
    if cached is not None:
        return ConversionResult(**cached)

    # existing parse logic
    result = parse_url(file_path_or_url) if is_url else parse_local_file(file_path_or_url)

    # Cache serialized result
    cache_parse_result(file_hash, {
        "markdown": result.markdown,
        "status": result.status,
        "processing_time": result.processing_time,
        "errors": result.errors,
    })
    return result
```

### 5. Docker Compose Addition

```yaml
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    networks:
      - backend
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 15s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 256M
    restart: unless-stopped

# Add to mcp service environment:
#   REDIS_URL: "redis://redis:6379/0"
#   ENABLE_CACHE: "true"
```

### 6. Dependency Addition

```toml
dependencies = [
    # ... existing ...
    "redis>=5,<6",
]
```

---

## Testing Strategy

### Unit Tests (`tests/unit/test_cache.py`)

- `test_hash_key_deterministic`: Same input → same hash.
- `test_hash_key_different_inputs`: Different inputs → different hashes.
- `test_cache_miss_returns_none`: Empty cache → None.
- `test_cache_hit_returns_value`: Set then get → same value.
- `test_cache_ttl_expired`: Set with short TTL, wait, get → None.
- `test_invalidate_namespace`: Invalidate specific namespace only.
- `test_invalidate_for_document`: Clears search + parse caches.
- `test_cache_disabled_passthrough`: When `ENABLE_CACHE=False`, all operations are no-ops.
- `test_redis_unavailable_graceful`: When Redis is down, operations fail silently.

### Integration Tests (`tests/integration/test_cache_integration.py`)

- `test_embedding_cache_hit`: Embed same text twice → second call returns cached.
- `test_search_cache_hit`: Search same query twice → second call returns cached.
- `test_search_cache_invalidation`: Delete document → search cache cleared.
- `test_parse_cache_hit`: Parse same file → second call returns cached.
- `test_cache_with_real_redis`: Full flow with actual Redis instance.

### Performance Tests

- Measure latency of cached vs. uncached search (expect 2-5x improvement on cache hits).
- Measure Redis memory usage after 1000 cached queries.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Redis becomes unavailable | Graceful fallback to no-cache | All cache ops wrapped in try/except; pipeline continues without cache |
| Cache staleness (stale results after re-ingest) | User sees old results | Invalidate caches on document delete/re-ingest; short TTL for search results |
| Redis memory exhaustion | OOM crash | Set `maxmemory` + `allkeys-lru` eviction; monitor with `INFO memory` |
| Cache key collisions | Wrong results returned | SHA256 hash with namespace prefix; collision probability negligible |
| JSON serialization overhead | Slight latency on cache writes | JSON is fast enough for these sizes; use msgpack if needed later |
| Cold cache on restart | First queries are slow | Expected behavior; pre-warm with common queries if needed |

---

## Priority & Effort

- **Priority**: Medium (reduces latency but not critical for correctness)
- **Estimated effort**: 1-2 days
- **Dependencies**: Redis (new service in docker-compose)
- **Rollback**: Feature flag `ENABLE_CACHE` defaults to `False`; remove Redis from compose
