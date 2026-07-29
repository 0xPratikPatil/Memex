"""Integration tests for Redis caching layer — requires a live Redis instance."""

from __future__ import annotations

import uuid

import pytest

from rag import config
from rag.services.cache import (
    cache_embedding,
    cache_search_results,
    get_cached,
    get_cached_embedding,
    get_cached_search_results,
    invalidate_for_document,
    invalidate_namespace,
    set_cached,
)


@pytest.mark.integration
class TestCacheIntegration:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        import redis

        import rag.services.cache as cache_mod

        try:
            self._redis_client = redis.Redis(host="localhost", port=6379, socket_connect_timeout=2)
            self._redis_client.ping()
        except Exception:
            pytest.skip("Redis not available")

        monkeypatch.setattr(config, "ENABLE_CACHE", True)
        cache_mod._redis = None

        self._ns = f"itest_{uuid.uuid4().hex[:8]}"

        yield

        try:
            pattern = f"rag:{self._ns}:*"
            keys = list(self._redis_client.scan_iter(match=pattern, count=1000))
            if keys:
                self._redis_client.delete(*keys)
        except Exception:
            pass

        for ns in ("emb", "search"):
            try:
                pattern = f"rag:{ns}:*"
                keys = list(self._redis_client.scan_iter(match=pattern, count=1000))
                if keys:
                    self._redis_client.delete(*keys)
            except Exception:
                pass

        cache_mod._redis = None

    def test_set_and_get(self):
        set_cached(self._ns, "test_key", {"data": "hello"}, 60)
        result = get_cached(self._ns, "test_key")
        assert result == {"data": "hello"}

    def test_get_miss_returns_none(self):
        result = get_cached(self._ns, "nonexistent")
        assert result is None

    def test_cache_embedding_and_retrieve(self):
        emb = [0.1, 0.2, 0.3]
        cache_embedding("hello world", emb)
        result = get_cached_embedding("hello world")
        assert result == emb

    def test_cache_embedding_miss(self):
        result = get_cached_embedding("no_such_text")
        assert result is None

    def test_cache_search_results_and_retrieve(self):
        results = [{"id": "1", "content": "test chunk"}]
        cache_search_results("integration query", 5, self._ns, results)
        cached = get_cached_search_results("integration query", 5, self._ns)
        assert cached == results

    def test_cache_search_results_none_source_filter(self):
        results = [{"id": "2", "content": "another chunk"}]
        cache_search_results("some query", 10, None, results)
        cached = get_cached_search_results("some query", 10, None)
        assert cached == results

    def test_invalidate_namespace_clears_keys(self):
        set_cached(self._ns, "k1", "val1", 60)
        set_cached(self._ns, "k2", "val2", 60)

        assert get_cached(self._ns, "k1") == "val1"

        invalidate_namespace(self._ns)

        assert get_cached(self._ns, "k1") is None
        assert get_cached(self._ns, "k2") is None

    def test_invalidate_for_document_clears_search_with_source_in_results(self):
        results = [{"id": "99", "text": "doc test", "source": "/docs/test.pdf"}]
        cache_search_results("doc query", 3, None, results)
        cached = get_cached_search_results("doc query", 3, None)
        assert cached == results

        invalidate_for_document("/docs/test.pdf")

        assert get_cached_search_results("doc query", 3, None) is None

    def test_invalidate_for_document_preserves_unrelated_search_cache(self):
        results = [{"id": "99", "text": "doc test"}]
        cache_search_results("doc query", 3, None, results)
        cached = get_cached_search_results("doc query", 3, None)
        assert cached == results

        invalidate_for_document("/docs/test.pdf")

        assert get_cached_search_results("doc query", 3, None) == results

    def test_overwrite_updates_value(self):
        set_cached(self._ns, "overwrite_key", "original", 60)
        set_cached(self._ns, "overwrite_key", "updated", 60)
        assert get_cached(self._ns, "overwrite_key") == "updated"
