"""Redis-backed caching layer for RAG pipeline.

Implements cache-aside pattern with TTL-based invalidation.
Redis is optional — all operations gracefully degrade to no-ops when
Redis is unavailable or caching is disabled.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from rag import config

logger = logging.getLogger("cache")

_redis: Any = None
_metrics = None


@dataclass
class CacheMetrics:
    """Collect cache hit/miss/latency metrics."""

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
    if _metrics is None:
        _metrics = CacheMetrics()
    return _metrics


def _get_redis() -> Any:
    """Lazy-init Redis client with connection pooling."""
    global _redis
    if _redis is not None:
        return _redis
    if not config.ENABLE_CACHE:
        return None
    try:
        import redis

        _redis = redis.Redis.from_url(
            config.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            max_connections=10,
        )
        _redis.ping()
        logger.info("Connected to Redis at %s", config.REDIS_URL)
        return _redis
    except ImportError:
        logger.warning("redis package not installed, caching disabled")
        return None
    except Exception as exc:
        logger.warning("Redis unavailable, caching disabled: %s", exc)
        _redis = None
        return None


def _hash_key(*parts: str) -> str:
    """Create a deterministic cache key hash from parts."""
    combined = ":".join(str(p) for p in parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:24]


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
    try:
        cache_key = f"rag:{namespace}:{_hash_key(key_parts)}"
        data = r.get(cache_key)
        elapsed = (time.monotonic() - t0) * 1000
        metrics.total_get_latency_ms += elapsed
        if data:
            metrics.hits += 1
            logger.debug("Cache hit: %s (%.1fms)", cache_key, elapsed)
            return json.loads(data)
        metrics.misses += 1
        logger.debug("Cache miss: %s (%.1fms)", cache_key, elapsed)
        return None
    except Exception as exc:
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
    try:
        cache_key = f"rag:{namespace}:{_hash_key(key_parts)}"
        data = json.dumps(value)
        r.set(cache_key, data, ex=ttl)
        elapsed = (time.monotonic() - t0) * 1000
        metrics.sets += 1
        metrics.total_set_latency_ms += elapsed
        logger.debug("Cache set: %s (ttl=%ds, %.1fms)", cache_key, ttl, elapsed)
    except Exception as exc:
        metrics.errors += 1
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
        metrics = _get_metrics()
        metrics.invalidations += len(keys)
        return len(keys)
    except Exception as exc:
        logger.warning("Cache invalidation failed: %s", exc)
        return 0


def invalidate_for_document(source_identifier: str) -> int:
    """Invalidate all caches related to a document.

    Clears search results and parse cache entries.
    Returns total number of invalidated keys.
    """
    count = 0
    count += invalidate_namespace("search")
    count += invalidate_namespace("parse")
    logger.info("Invalidated %d caches for document: %s", count, source_identifier)
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
) -> None:
    """Cache search results."""
    key = f"{query}:{top_k}:{source_filter or ''}"
    set_cached("search", key, results, config.CACHE_TTL_SEARCH)


def get_cached_search_results(
    query: str,
    top_k: int,
    source_filter: str | None,
) -> list[dict[str, Any]] | None:
    """Get cached search results."""
    key = f"{query}:{top_k}:{source_filter or ''}"
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
    if r is not None:
        try:
            info = r.info("memory")
            stats["redis_memory_used_mb"] = round(info.get("used_memory", 0) / (1024 * 1024), 2)
            stats["redis_memory_peak_mb"] = round(info.get("peak_memory", 0) / (1024 * 1024), 2)
            key_count = r.dbsize()
            stats["redis_key_count"] = key_count
        except Exception:
            pass
    return stats


def close() -> None:
    """Close Redis connection."""
    global _redis
    if _redis is not None:
        import contextlib

        with contextlib.suppress(Exception):
            _redis.close()
        _redis = None
