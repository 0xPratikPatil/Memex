"""Unit tests for cache module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from rag import config
from rag.services.cache import (
    CacheMetrics,
    _hash_key,
    cache_embedding,
    cache_parse_result,
    cache_search_results,
    get_cache_stats,
    get_cached,
    get_cached_embedding,
    get_cached_parse_result,
    get_cached_search_results,
    invalidate_for_document,
    invalidate_namespace,
    set_cached,
)

# ── Hash key tests ──────────────────────────────────────────────────────────


class TestHashKey:
    def test_deterministic(self) -> None:
        a = _hash_key("hello", "world")
        b = _hash_key("hello", "world")
        assert a == b

    def test_different_inputs(self) -> None:
        a = _hash_key("hello")
        b = _hash_key("world")
        assert a != b

    def test_length(self) -> None:
        h = _hash_key("test")
        assert len(h) == 16

    def test_hex_only(self) -> None:
        h = _hash_key("test")
        assert all(c in "0123456789abcdef" for c in h)


# ── CacheMetrics tests ──────────────────────────────────────────────────────


class TestCacheMetrics:
    def test_hit_rate_empty(self) -> None:
        m = CacheMetrics()
        assert m.hit_rate == 0.0

    def test_hit_rate计算正确(self) -> None:
        m = CacheMetrics(hits=3, misses=1)
        assert m.hit_rate == 0.75

    def test_as_dict(self) -> None:
        m = CacheMetrics(hits=10, misses=5, sets=7)
        d = m.as_dict()
        assert d["hits"] == 10
        assert d["misses"] == 5
        assert d["sets"] == 7
        assert "hit_rate" in d
        assert "avg_get_latency_ms" in d


# ── Disabled cache tests ────────────────────────────────────────────────────


class TestDisabledCache:
    @patch.object(config, "ENABLE_CACHE", False)
    def test_get_returns_none(self) -> None:
        assert get_cached("test", "key") is None

    @patch.object(config, "ENABLE_CACHE", False)
    def test_set_is_noop(self) -> None:
        set_cached("test", "key", "value", 60)

    @patch.object(config, "ENABLE_CACHE", False)
    def test_invalidate_returns_zero(self) -> None:
        assert invalidate_namespace("test") == 0

    @patch.object(config, "ENABLE_CACHE", False)
    def test_embedding_cache_returns_none(self) -> None:
        assert get_cached_embedding("text") is None

    @patch.object(config, "ENABLE_CACHE", False)
    def test_search_cache_returns_none(self) -> None:
        assert get_cached_search_results("q", 5, None) is None

    @patch.object(config, "ENABLE_CACHE", False)
    def test_parse_cache_returns_none(self) -> None:
        assert get_cached_parse_result("hash") is None

    @patch.object(config, "ENABLE_CACHE", False)
    def test_stats_show_disabled(self) -> None:
        stats = get_cache_stats()
        assert stats["enabled"] is False


# ── Mock Redis tests ────────────────────────────────────────────────────────


@pytest.fixture
def mock_redis() -> MagicMock:
    """Return a mock Redis client."""
    r = MagicMock()
    r.ping = MagicMock(return_value=True)
    r.get = MagicMock(return_value=None)
    r.setex = MagicMock(return_value=True)
    r.delete = MagicMock(return_value=1)
    r.scan_iter = MagicMock(return_value=iter([]))
    r.dbsize = MagicMock(return_value=0)
    r.info = MagicMock(return_value={"used_memory": 0, "peak_memory": 0})
    return r


class TestWithMockRedis:
    @patch.object(config, "ENABLE_CACHE", True)
    @patch.object(config, "REDIS_URL", "redis://localhost:6379/0")
    def test_set_and_get(self, mock_redis: MagicMock) -> None:
        import rag.services.cache as cache_mod

        cache_mod._redis = mock_redis

        set_cached("test", "mykey", {"data": "value"}, 60)
        mock_redis.setex.assert_called_once()

        # Simulate cache hit
        mock_redis.get.return_value = json.dumps({"data": "value"})
        result = get_cached("test", "mykey")
        assert result == {"data": "value"}

        cache_mod._redis = None

    @patch.object(config, "ENABLE_CACHE", True)
    def test_cache_miss_returns_none(self, mock_redis: MagicMock) -> None:
        import rag.services.cache as cache_mod

        cache_mod._redis = mock_redis
        mock_redis.get.return_value = None

        result = get_cached("test", "nonexistent")
        assert result is None

        cache_mod._redis = None

    @patch.object(config, "ENABLE_CACHE", True)
    def test_invalidate_namespace(self, mock_redis: MagicMock) -> None:
        import rag.services.cache as cache_mod

        cache_mod._redis = mock_redis
        mock_redis.scan_iter.return_value = iter(["rag:test:k1", "rag:test:k2"])

        count = invalidate_namespace("test")
        assert count == 2
        mock_redis.delete.assert_called_once_with("rag:test:k1", "rag:test:k2")

        cache_mod._redis = None

    @patch.object(config, "ENABLE_CACHE", True)
    def test_invalidate_for_document(self, mock_redis: MagicMock) -> None:
        import rag.services.cache as cache_mod

        cache_mod._redis = mock_redis
        mock_redis.scan_iter.return_value = iter([])

        count = invalidate_for_document("/docs/test.pdf")
        assert count == 0

        cache_mod._redis = None


# ── Domain helper tests ──────────────────────────────────────────────────────


class TestDomainHelpers:
    @patch.object(config, "ENABLE_CACHE", True)
    def test_cache_embedding_roundtrip(self, mock_redis: MagicMock) -> None:
        import rag.services.cache as cache_mod

        cache_mod._redis = mock_redis

        embedding = [0.1, 0.2, 0.3]
        cache_embedding("hello world", embedding)
        mock_redis.setex.assert_called_once()

        # Simulate hit
        mock_redis.get.return_value = json.dumps(embedding)
        result = get_cached_embedding("hello world")
        assert result == embedding

        cache_mod._redis = None

    @patch.object(config, "ENABLE_CACHE", True)
    def test_cache_search_results_roundtrip(self, mock_redis: MagicMock) -> None:
        import rag.services.cache as cache_mod

        cache_mod._redis = mock_redis

        results = [{"id": "1", "content": "test"}]
        cache_search_results("query", 5, None, results)

        mock_redis.get.return_value = json.dumps(results)
        result = get_cached_search_results("query", 5, None)
        assert result == results

        cache_mod._redis = None

    @patch.object(config, "ENABLE_CACHE", True)
    def test_cache_parse_result_roundtrip(self, mock_redis: MagicMock) -> None:
        import rag.services.cache as cache_mod

        cache_mod._redis = mock_redis

        result = {"markdown": "# Test", "status": "success", "processing_time": 1.5, "errors": []}
        cache_parse_result("abc123", result)

        mock_redis.get.return_value = json.dumps(result)
        cached = get_cached_parse_result("abc123")
        assert cached == result

        cache_mod._redis = None

    @patch.object(config, "ENABLE_CACHE", True)
    def test_search_cache_key_includes_source_filter(self, mock_redis: MagicMock) -> None:
        import rag.services.cache as cache_mod

        cache_mod._redis = mock_redis

        cache_search_results("q", 5, "/docs/a.pdf", [])
        key_with_filter = mock_redis.setex.call_args[0][0]

        cache_search_results("q", 5, None, [])
        key_no_filter = mock_redis.setex.call_args[0][0]

        assert key_with_filter != key_no_filter

        cache_mod._redis = None


# ── Error handling tests ────────────────────────────────────────────────────


class TestErrorHandling:
    @patch.object(config, "ENABLE_CACHE", True)
    def test_redis_unavailable_graceful(self) -> None:
        import rag.services.cache as cache_mod

        cache_mod._redis = None
        # Force _get_redis to attempt connection and fail
        with patch("rag.services.cache._get_redis", return_value=None):
            result = get_cached("test", "key")
            assert result is None

    @patch.object(config, "ENABLE_CACHE", True)
    def test_redis_get_error_returns_none(self, mock_redis: MagicMock) -> None:
        import rag.services.cache as cache_mod

        cache_mod._redis = mock_redis
        mock_redis.get.side_effect = Exception("connection lost")

        result = get_cached("test", "key")
        assert result is None

        cache_mod._redis = None

    @patch.object(config, "ENABLE_CACHE", True)
    def test_redis_set_error_no_raise(self, mock_redis: MagicMock) -> None:
        import rag.services.cache as cache_mod

        cache_mod._redis = mock_redis
        mock_redis.setex.side_effect = Exception("connection lost")

        # Should not raise
        set_cached("test", "key", "value", 60)

        cache_mod._redis = None

    @patch.object(config, "ENABLE_CACHE", True)
    def test_close(self, mock_redis: MagicMock) -> None:
        import rag.services.cache as cache_mod

        cache_mod._redis = mock_redis
        from rag.services.cache import close

        close()
        mock_redis.close.assert_called_once()
        assert cache_mod._redis is None
