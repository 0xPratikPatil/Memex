"""Redis-backed caching layer for RAG pipeline.

Implements cache-aside pattern with TTL-based invalidation.
Redis is optional — all operations gracefully degrade to an in-memory
dict (LRU-bounded) when Redis is unavailable or caching is disabled.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from memex.engine.core import config

logger = logging.getLogger("cache")

_MEMORY_MAX_ENTRIES = 2000

_redis: Any = None
_redis_lock = threading.Lock()
_memory_cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
_memory_lock = threading.Lock()
_metrics: CacheMetrics | None = None
_metrics_lock = threading.Lock()


@dataclass
class CacheMetrics:
    """Collect cache hit/miss/latency metrics (thread-safe via external lock)."""

    hits: int = 0
    misses: int = 0
    errors: int = 0
    sets: int = 0
    invalidations: int = 0
    total_get_latency_ms: float = 0.0
    total_set_latency_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "sets": self.sets,
            "invalidations": self.invalidations,
            "hit_rate": round(self.hit_rate, 4),
            "avg_get_latency_ms": round(self.total_get_latency_ms / max(self.hits + self.misses, 1), 2),
            "avg_set_latency_ms": round(self.total_set_latency_ms / max(self.sets, 1), 2),
        }


def _get_metrics() -> CacheMetrics:
    global _metrics
    if _metrics is not None:
        return _metrics
    with _metrics_lock:
        if _metrics is None:
            _metrics = CacheMetrics()
        return _metrics


def _get_redis() -> Any:
    """Lazy-init Redis client with connection pooling.

    Returns:
        redis.Redis client if Redis is connected,
        True (sentinel) if Redis unavailable but in-memory fallback active,
        None if caching is fully disabled.
    """
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


def _hash_key(*parts: str) -> str:
    """Create a deterministic cache key hash from parts."""
    combined = ":".join(str(p) for p in parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:24]


def _mem_get(key: str) -> Any | None:
    """Get a value from the in-memory LRU cache. Returns None on miss or expiry."""
    with _memory_lock:
        entry = _memory_cache.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at > 0 and time.monotonic() > expires_at:
            del _memory_cache[key]
            return None
        _memory_cache.move_to_end(key)
        return value


def _mem_set(key: str, value: Any, ttl: int) -> None:
    """Store a value in the in-memory LRU cache with TTL."""
    expires_at = time.monotonic() + ttl if ttl > 0 else 0
    with _memory_lock:
        _memory_cache[key] = (value, expires_at)
        _memory_cache.move_to_end(key)
        while len(_memory_cache) > _MEMORY_MAX_ENTRIES:
            _memory_cache.popitem(last=False)


def _mem_scan(match: str) -> list[str]:
    """Find in-memory keys matching a glob-like prefix. Match arg is 'rag:{namespace}:*'."""
    prefix = match.rstrip("*")
    with _memory_lock:
        return [k for k in _memory_cache if k.startswith(prefix)]


def _mem_delete(*keys: str) -> int:
    """Delete keys from in-memory cache. Returns count of deleted keys."""
    count = 0
    with _memory_lock:
        for k in keys:
            if k in _memory_cache:
                del _memory_cache[k]
                count += 1
    return count


def get_cached(namespace: str, key_parts: str) -> Any | None:
    """Get a cached value by namespace and key parts.

    Returns None on cache miss, error, or when caching is disabled.
    """
    if not config.ENABLE_CACHE:
        return None
    r = _get_redis()
    if r is None:
        return None

    metrics = _get_metrics()
    t0 = time.monotonic()
    cache_key = f"rag:{namespace}:{_hash_key(key_parts)}"

    if r is True:
        # In-memory fallback
        try:
            data = _mem_get(cache_key)
            elapsed = (time.monotonic() - t0) * 1000
            with _metrics_lock:
                metrics.total_get_latency_ms += elapsed
                if data is not None:
                    metrics.hits += 1
                else:
                    metrics.misses += 1
            return data
        except Exception:
            return None

    try:
        data = r.get(cache_key)
        elapsed = (time.monotonic() - t0) * 1000
        with _metrics_lock:
            metrics.total_get_latency_ms += elapsed
            if data:
                metrics.hits += 1
            else:
                metrics.misses += 1
        if data:
            logger.debug("Cache hit: %s (%.1fms)", cache_key, elapsed)
            return json.loads(data)
        logger.debug("Cache miss: %s (%.1fms)", cache_key, elapsed)
        return None
    except Exception as exc:
        with _metrics_lock:
            metrics.errors += 1
        logger.warning("Cache get failed: %s", exc)
        return None


def set_cached(namespace: str, key_parts: str, value: Any, ttl: int) -> None:
    """Store a value in cache with TTL (seconds)."""
    if not config.ENABLE_CACHE:
        return
    r = _get_redis()
    if r is None:
        return

    metrics = _get_metrics()
    t0 = time.monotonic()
    cache_key = f"rag:{namespace}:{_hash_key(key_parts)}"

    if r is True:
        # In-memory fallback
        try:
            _mem_set(cache_key, value, ttl)
            elapsed = (time.monotonic() - t0) * 1000
            with _metrics_lock:
                metrics.sets += 1
                metrics.total_set_latency_ms += elapsed
            logger.debug("Cache set (memory): %s (ttl=%ds, %.1fms)", cache_key, ttl, elapsed)
        except Exception as exc:
            with _metrics_lock:
                metrics.errors += 1
            logger.warning("Cache set failed (memory): %s", exc)
        return

    try:
        data = json.dumps(value)
        r.set(cache_key, data, ex=ttl)
        elapsed = (time.monotonic() - t0) * 1000
        with _metrics_lock:
            metrics.sets += 1
            metrics.total_set_latency_ms += elapsed
        logger.debug("Cache set: %s (ttl=%ds, %.1fms)", cache_key, ttl, elapsed)
    except Exception as exc:
        with _metrics_lock:
            metrics.errors += 1
        logger.warning("Cache set failed: %s", exc)


def invalidate_namespace(namespace: str) -> int:
    """Invalidate all keys in a namespace. Returns count of deleted keys."""
    if not config.ENABLE_CACHE:
        return 0
    r = _get_redis()
    if r is None:
        return 0

    if r is True:
        pattern = f"rag:{namespace}:"
        keys = _mem_scan(pattern)
        count = _mem_delete(*keys)
        if count:
            logger.info("Invalidated %d memory cache keys in namespace '%s'", count, namespace)
        metrics = _get_metrics()
        with _metrics_lock:
            metrics.invalidations += count
        return count

    try:
        pattern = f"rag:{namespace}:*"
        keys = list(r.scan_iter(match=pattern, count=1000))
        if keys:
            r.delete(*keys)
            logger.info("Invalidated %d cache keys in namespace '%s'", len(keys), namespace)
        metrics = _get_metrics()
        metrics.invalidations += len(keys)
        return len(keys)
    except Exception as exc:
        logger.warning("Cache invalidation failed: %s", exc)
        return 0


def invalidate_for_document(source_identifier: str) -> int:
    """Invalidate all caches related to a specific document.

    Uses the source_identifier to scope invalidation to only this document's
    cache entries, preserving caches for other documents.
    Returns total number of invalidated keys.
    """
    if not config.ENABLE_CACHE:
        return 0
    r = _get_redis()
    if r is None:
        return 0

    if r is True:
        count = 0
        with _memory_lock:
            to_delete = [
                k
                for k, (v, _) in _memory_cache.items()
                if (isinstance(v, str) and source_identifier in v)
                or (isinstance(v, (list, dict)) and source_identifier in json.dumps(v))
                or source_identifier in k
            ]
            for k in to_delete:
                del _memory_cache[k]
            count = len(to_delete)
        if count:
            logger.info("Invalidated %d memory cache keys for document: %s", count, source_identifier)
        metrics = _get_metrics()
        with _metrics_lock:
            metrics.invalidations += count
        return count

    count = 0
    try:
        # Invalidate search cache entries that contain this document's source
        # Since search cache keys include query params, we scan and filter
        pattern = "rag:search:*"
        keys_to_delete = []
        for key in r.scan_iter(match=pattern, count=1000):
            try:
                data = r.get(key)
                if data and source_identifier.encode() in data.encode():
                    keys_to_delete.append(key)
            except Exception:
                pass
        # Invalidate parse cache entries for this document
        parse_pattern = "rag:parse:*"
        for key in r.scan_iter(match=parse_pattern, count=1000):
            try:
                if source_identifier.encode() in key.encode():
                    keys_to_delete.append(key)
            except Exception:
                pass
        if keys_to_delete:
            r.delete(*keys_to_delete)
            count = len(keys_to_delete)
            logger.info("Invalidated %d cache keys for document: %s", count, source_identifier)
        metrics = _get_metrics()
        with _metrics_lock:
            metrics.invalidations += count
    except Exception as exc:
        logger.warning("Cache invalidation failed: %s", exc)
    return count


# ── Domain-specific helpers ──────────────────────────────────────────────────


def cache_embedding(text: str, embedding: list[float], model: str = "") -> None:
    """Cache a text → embedding mapping, keyed by model name.

    Including the model name prevents cross-model cache poisoning when
    switching embedding models (e.g. bge-m3 → qwen3-embedding).
    """
    model = model or config.EMBED_MODEL
    key = f"{model}:{text}"
    set_cached("emb", key, embedding, config.CACHE_TTL_EMBEDDING)


def get_cached_embedding(text: str, model: str = "") -> list[float] | None:
    """Get cached embedding for text, keyed by model name."""
    model = model or config.EMBED_MODEL
    key = f"{model}:{text}"
    return get_cached("emb", key)


def cache_search_results(
    query: str,
    top_k: int,
    source_filter: str | None,
    results: list[dict[str, Any]],
    metadata_filter: dict[str, Any] | None = None,
) -> None:
    """Cache search results."""
    import json as _json

    filter_key = _json.dumps(metadata_filter, sort_keys=True) if metadata_filter else ""
    key = f"{query}:{top_k}:{source_filter or ''}:{filter_key}"
    set_cached("search", key, results, config.CACHE_TTL_SEARCH)


def get_cached_search_results(
    query: str,
    top_k: int,
    source_filter: str | None,
    metadata_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    """Get cached search results."""
    import json as _json

    filter_key = _json.dumps(metadata_filter, sort_keys=True) if metadata_filter else ""
    key = f"{query}:{top_k}:{source_filter or ''}:{filter_key}"
    return get_cached("search", key)


def cache_parse_result(file_hash: str, result: dict[str, Any]) -> None:
    """Cache Docling parse result."""
    set_cached("parse", file_hash, result, config.CACHE_TTL_PARSE)


def get_cached_parse_result(file_hash: str) -> dict[str, Any] | None:
    """Get cached Docling parse result."""
    return get_cached("parse", file_hash)


def get_cache_stats() -> dict[str, Any]:
    """Return current cache metrics and Redis info."""
    metrics = _get_metrics()
    stats: dict[str, Any] = {"metrics": metrics.as_dict(), "enabled": config.ENABLE_CACHE}

    r = _get_redis()
    if r is not None and r is not True:
        try:
            info = r.info("memory")
            stats["redis_memory_used_mb"] = round(info.get("used_memory", 0) / (1024 * 1024), 2)
            stats["redis_memory_peak_mb"] = round(info.get("peak_memory", 0) / (1024 * 1024), 2)
            key_count = r.dbsize()
            stats["redis_key_count"] = key_count
        except Exception:
            pass
    elif r is True:
        stats["cache_backend"] = "memory"
        stats["memory_entries"] = len(_memory_cache)
        stats["memory_max"] = _MEMORY_MAX_ENTRIES
    return stats


def close() -> None:
    """Close Redis connection and pool."""
    global _redis
    if _redis is not None and _redis is not True:
        import contextlib

        with contextlib.suppress(Exception):
            _redis.close()
            _redis.connection_pool.disconnect()
    _redis = None
