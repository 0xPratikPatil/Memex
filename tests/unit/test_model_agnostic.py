"""Unit tests for model-agnostic pipeline: fallback, cache, API detection."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

# ── Shared fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def mock_redis() -> MagicMock:
    """Return a mock Redis client."""
    r = MagicMock()
    r.ping = MagicMock(return_value=True)
    r.get = MagicMock(return_value=None)
    r.set = MagicMock(return_value=True)
    r.delete = MagicMock(return_value=1)
    r.scan_iter = MagicMock(return_value=iter([]))
    r.dbsize = MagicMock(return_value=0)
    r.info = MagicMock(return_value={"used_memory": 0, "peak_memory": 0})
    return r


# ── Embedding fallback tests ─────────────────────────────────────────────


class TestEmbeddingFallback:
    """Pipeline._dense_embed_batch delegates to EmbeddingService with fallback."""

    @patch("memex.engine.core.pipeline.config.EMBED_MODEL", "primary-model")
    @patch("memex.engine.core.pipeline.config.EMBED_MODEL_FALLBACK", "fallback-model")
    @patch("memex.engine.core.pipeline.config.ENABLE_CACHE", False)
    def test_uses_primary_model_by_default(self) -> None:
        from memex.engine.ingestion.embedding import EmbeddingService
        from memex.engine.core.pipeline import RAGEngine

        engine = RAGEngine()

        with patch.object(EmbeddingService, "_post_batch", return_value=[[0.1, 0.2]]):
            result = engine._dense_embed_batch(["test text"], model="primary-model")
            assert result == [[0.1, 0.2]]

    @patch("memex.engine.core.pipeline.config.EMBED_MODEL", "primary-model")
    @patch("memex.engine.core.pipeline.config.EMBED_MODEL_FALLBACK", "fallback-model")
    @patch("memex.engine.core.pipeline.config.ENABLE_CACHE", False)
    def test_falls_back_on_primary_failure(self) -> None:
        from memex.engine.ingestion.embedding import EmbeddingService
        from memex.engine.core.pipeline import RAGEngine

        engine = RAGEngine()
        call_models = []

        def post_side_effect(texts, model):
            call_models.append(model)
            if model == "primary-model":
                raise RuntimeError("primary model unavailable")
            return [[0.3, 0.4] for _ in texts]

        with patch.object(EmbeddingService, "_post_batch", side_effect=post_side_effect):
            result = engine._dense_embed_batch(["test text"])
            assert "primary-model" in call_models
            assert "fallback-model" in call_models
            assert result == [[0.3, 0.4]]

    @patch("memex.engine.core.pipeline.config.EMBED_MODEL", "fallback-model")
    @patch("memex.engine.core.pipeline.config.EMBED_MODEL_FALLBACK", "fallback-model")
    @patch("memex.engine.core.pipeline.config.ENABLE_CACHE", False)
    def test_raises_when_fallback_also_fails(self) -> None:
        from memex.engine.ingestion.embedding import EmbeddingService
        from memex.engine.core.pipeline import RAGEngine

        engine = RAGEngine()

        with (
            patch.object(EmbeddingService, "_post_batch", side_effect=RuntimeError("both models failed")),
            pytest.raises(RuntimeError, match="both models failed"),
        ):
            engine._dense_embed_batch(["test text"])


# ── Cache model-awareness tests ──────────────────────────────────────────


class TestCacheModelAwareness:
    """Embedding cache keys include model name to prevent cross-model poisoning."""

    @patch("memex.engine.utils.cache.config.ENABLE_CACHE", True)
    @patch("memex.engine.utils.cache.config.EMBED_MODEL", "model-a")
    def test_cache_key_includes_model(self, mock_redis: MagicMock) -> None:
        import memex.engine.utils.cache as cache_mod

        cache_mod._redis = mock_redis

        from memex.engine.utils.cache import cache_embedding

        cache_embedding("hello", [0.1, 0.2], model="model-a")
        key_a = mock_redis.set.call_args[0][0]

        cache_embedding("hello", [0.3, 0.4], model="model-b")
        key_b = mock_redis.set.call_args[0][0]

        # Keys must differ even though text is the same
        assert key_a != key_b

        cache_mod._redis = None

    @patch("memex.engine.utils.cache.config.ENABLE_CACHE", True)
    @patch("memex.engine.utils.cache.config.EMBED_MODEL", "model-a")
    def test_get_uses_model_in_key(self, mock_redis: MagicMock) -> None:
        import memex.engine.utils.cache as cache_mod

        cache_mod._redis = mock_redis

        from memex.engine.utils.cache import cache_embedding, get_cached_embedding

        # Cache for model-a
        cache_embedding("hello", [0.1, 0.2], model="model-a")

        # Get for model-a should hit
        mock_redis.get.return_value = json.dumps([0.1, 0.2])
        result_a = get_cached_embedding("hello", model="model-a")
        assert result_a == [0.1, 0.2]

        # Get for model-b should miss (different key)
        mock_redis.get.return_value = None
        result_b = get_cached_embedding("hello", model="model-b")
        assert result_b is None

        cache_mod._redis = None

    @patch("memex.engine.utils.cache.config.ENABLE_CACHE", True)
    @patch("memex.engine.utils.cache.config.EMBED_MODEL", "model-a")
    def test_default_model_used_when_not_specified(self, mock_redis: MagicMock) -> None:
        import memex.engine.utils.cache as cache_mod

        cache_mod._redis = mock_redis

        from memex.engine.utils.cache import cache_embedding

        # Cache without specifying model
        cache_embedding("hello", [0.1, 0.2])
        key_default = mock_redis.set.call_args[0][0]

        # Cache with explicit model-a (same as default)
        cache_embedding("hello", [0.3, 0.4], model="model-a")
        key_explicit = mock_redis.set.call_args[0][0]

        # Should produce same key
        assert key_default == key_explicit

        cache_mod._redis = None


# ── Ollama API detection tests ───────────────────────────────────────────


class TestEmbeddingServiceAPI:
    """EmbeddingService uses batched /api/embed endpoint."""

    def _make_svc(self):
        from memex.engine.ingestion.embedding import EmbeddingService

        mock_client = MagicMock()
        svc = EmbeddingService(mock_client)
        return svc, mock_client

    @patch("memex.engine.ingestion.embedding.config.EMBED_MODEL", "test-model")
    @patch("memex.engine.ingestion.embedding.config.DENSE_DIM", 2)
    @patch("memex.engine.ingestion.embedding.config.ENABLE_CACHE", False)
    def test_embed_sends_batched_input(self) -> None:
        svc, mock_client = self._make_svc()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        with patch.object(svc, "_embed_url", "http://localhost:11434/api/embed"):
            result = svc.embed(["hello", "world"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        call_json = mock_client.post.call_args[1]["json"]
        assert call_json["input"] == ["hello", "world"]

    @patch("memex.engine.ingestion.embedding.config.EMBED_MODEL", "test-model")
    @patch("memex.engine.ingestion.embedding.config.DENSE_DIM", 2)
    @patch("memex.engine.ingestion.embedding.config.ENABLE_CACHE", False)
    def test_embed_caches_results(self) -> None:
        svc, mock_client = self._make_svc()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[0.1, 0.2]]}
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        with (
            patch.object(svc, "_embed_url", "http://localhost:11434/api/embed"),
            patch("memex.engine.utils.cache.cache_embedding") as mock_cache,
        ):
            svc.embed(["hello"])

        mock_cache.assert_called_once_with("hello", [0.1, 0.2], model="test-model")

    @patch("memex.engine.ingestion.embedding.config.EMBED_MODEL", "test-model")
    @patch("memex.engine.ingestion.embedding.config.ENABLE_CACHE", False)
    def test_embed_respects_batch_size(self) -> None:
        from memex.engine.core import config

        svc, _mock_client = self._make_svc()

        call_counts = []

        def post_side_effect(texts, model):
            call_counts.append(len(texts))
            return [[0.1] for _ in texts]

        with (
            patch.object(svc, "_embed_url", "http://localhost:11434/api/embed"),
            patch.object(config, "EMBED_BATCH_SIZE", 2),
            patch.object(svc, "_post_batch", side_effect=post_side_effect),
        ):
            svc.embed(["a", "b", "c"])

        # Two sub-batches: [2] then [1]
        assert call_counts == [2, 1]

    def test_resolve_url_switches_default_to_embed(self) -> None:
        from memex.engine.ingestion.embedding import EmbeddingService

        with patch("memex.engine.ingestion.embedding.config.OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings"):
            url = EmbeddingService._resolve_embed_url()
        assert "/api/embed" in url
        assert "embeddings" not in url


# ── Query expansion fallback tests ───────────────────────────────────────


class TestQueryExpansionFallback:
    """QueryExpander._embed delegates to EmbeddingService with fallback."""

    def _make_expander(self):
        from memex.engine.retrieval.expansion import QueryExpander

        mock_llm = MagicMock()
        expander = QueryExpander(mock_llm)
        return expander, mock_llm

    @patch("memex.engine.retrieval.expansion.config.EMBED_MODEL", "primary")
    @patch("memex.engine.retrieval.expansion.config.EMBED_MODEL_FALLBACK", "fallback")
    def test_uses_primary_model(self) -> None:
        from memex.engine.ingestion.embedding import EmbeddingService

        expander, _mock_llm = self._make_expander()

        with patch.object(EmbeddingService, "embed", return_value=[[0.1]]):
            result = expander._embed("test")

        assert result == [0.1]

    @patch("memex.engine.retrieval.expansion.config.EMBED_MODEL", "primary")
    @patch("memex.engine.retrieval.expansion.config.EMBED_MODEL_FALLBACK", "fallback")
    def test_falls_back_on_failure(self) -> None:
        from memex.engine.ingestion.embedding import EmbeddingService

        expander, _mock_llm = self._make_expander()

        with patch.object(EmbeddingService, "embed", return_value=[[0.3]]):
            result = expander._embed("test")

        assert result == [0.3]

    @patch("memex.engine.retrieval.expansion.config.EMBED_MODEL", "fallback")
    @patch("memex.engine.retrieval.expansion.config.EMBED_MODEL_FALLBACK", "fallback")
    def test_raises_when_fallback_also_fails(self) -> None:
        from memex.engine.ingestion.embedding import EmbeddingService

        expander, _mock_llm = self._make_expander()

        with (
            patch.object(EmbeddingService, "embed", side_effect=RuntimeError("all failed")),
            pytest.raises(RuntimeError, match="all failed"),
        ):
            expander._embed("test")


# ── Config fallback defaults tests ───────────────────────────────────────


class TestConfigFallbackDefaults:
    """Config has fallback model env vars."""

    def test_embed_model_fallback_exists(self) -> None:
        from memex.engine.core import config

        assert hasattr(config, "EMBED_MODEL_FALLBACK")

    def test_rerank_model_fallback_exists(self) -> None:
        from memex.engine.core import config

        assert hasattr(config, "RERANK_MODEL_FALLBACK")

    def test_rerank_type_exists(self) -> None:
        from memex.engine.core import config

        assert hasattr(config, "RERANK_TYPE")
        assert config.RERANK_TYPE in ("cross-encoder", "causal-lm", "auto")
