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
    r.setex = MagicMock(return_value=True)
    r.delete = MagicMock(return_value=1)
    r.scan_iter = MagicMock(return_value=iter([]))
    r.dbsize = MagicMock(return_value=0)
    r.info = MagicMock(return_value={"used_memory": 0, "peak_memory": 0})
    return r


# ── Embedding fallback tests ─────────────────────────────────────────────


class TestEmbeddingFallback:
    """Pipeline._dense_embed_batch falls back to EMBED_MODEL_FALLBACK on failure."""

    @patch("rag.pipeline.config.EMBED_MODEL", "primary-model")
    @patch("rag.pipeline.config.EMBED_MODEL_FALLBACK", "fallback-model")
    @patch("rag.pipeline.config.ENABLE_CACHE", False)
    def test_uses_primary_model_by_default(self) -> None:
        from rag.pipeline import RAGEngine

        engine = RAGEngine()

        def embed_side_effect(client, uncached_texts, cached_map, model):
            for idx, _text in uncached_texts:
                cached_map[idx] = [0.1, 0.2]

        with patch.object(engine, "_embed_via_ollama", side_effect=embed_side_effect):
            result = engine._dense_embed_batch(["test text"], model="primary-model")
            assert result == [[0.1, 0.2]]

    @patch("rag.pipeline.config.EMBED_MODEL", "primary-model")
    @patch("rag.pipeline.config.EMBED_MODEL_FALLBACK", "fallback-model")
    @patch("rag.pipeline.config.ENABLE_CACHE", False)
    def test_falls_back_on_primary_failure(self) -> None:
        from rag.pipeline import RAGEngine

        engine = RAGEngine()
        mock_client = MagicMock()
        engine._ollama = mock_client

        call_count = 0

        def side_effect(client, uncached_texts, cached_map, model):
            nonlocal call_count
            call_count += 1
            if model == "primary-model":
                raise RuntimeError("primary model unavailable")
            # Fallback succeeds
            for idx, _text in uncached_texts:
                cached_map[idx] = [0.3, 0.4]

        with patch.object(engine, "_embed_via_ollama", side_effect=side_effect):
            result = engine._dense_embed_batch(["test text"])
            assert call_count == 2  # primary + fallback
            assert result == [[0.3, 0.4]]

    @patch("rag.pipeline.config.EMBED_MODEL", "fallback-model")
    @patch("rag.pipeline.config.EMBED_MODEL_FALLBACK", "fallback-model")
    @patch("rag.pipeline.config.ENABLE_CACHE", False)
    def test_raises_when_fallback_also_fails(self) -> None:
        from rag.pipeline import RAGEngine

        engine = RAGEngine()
        mock_client = MagicMock()
        engine._ollama = mock_client

        def side_effect(client, uncached_texts, cached_map, model):
            raise RuntimeError("both models failed")

        with (
            patch.object(engine, "_embed_via_ollama", side_effect=side_effect),
            pytest.raises(RuntimeError, match="both models failed"),
        ):
            engine._dense_embed_batch(["test text"])


# ── Cache model-awareness tests ──────────────────────────────────────────


class TestCacheModelAwareness:
    """Embedding cache keys include model name to prevent cross-model poisoning."""

    @patch("rag.services.cache.config.ENABLE_CACHE", True)
    @patch("rag.services.cache.config.EMBED_MODEL", "model-a")
    def test_cache_key_includes_model(self, mock_redis: MagicMock) -> None:
        import rag.services.cache as cache_mod

        cache_mod._redis = mock_redis

        from rag.services.cache import cache_embedding

        cache_embedding("hello", [0.1, 0.2], model="model-a")
        key_a = mock_redis.setex.call_args[0][0]

        cache_embedding("hello", [0.3, 0.4], model="model-b")
        key_b = mock_redis.setex.call_args[0][0]

        # Keys must differ even though text is the same
        assert key_a != key_b

        cache_mod._redis = None

    @patch("rag.services.cache.config.ENABLE_CACHE", True)
    @patch("rag.services.cache.config.EMBED_MODEL", "model-a")
    def test_get_uses_model_in_key(self, mock_redis: MagicMock) -> None:
        import rag.services.cache as cache_mod

        cache_mod._redis = mock_redis

        from rag.services.cache import cache_embedding, get_cached_embedding

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

    @patch("rag.services.cache.config.ENABLE_CACHE", True)
    @patch("rag.services.cache.config.EMBED_MODEL", "model-a")
    def test_default_model_used_when_not_specified(self, mock_redis: MagicMock) -> None:
        import rag.services.cache as cache_mod

        cache_mod._redis = mock_redis

        from rag.services.cache import cache_embedding

        # Cache without specifying model
        cache_embedding("hello", [0.1, 0.2])
        key_default = mock_redis.setex.call_args[0][0]

        # Cache with explicit model-a (same as default)
        cache_embedding("hello", [0.3, 0.4], model="model-a")
        key_explicit = mock_redis.setex.call_args[0][0]

        # Should produce same key
        assert key_default == key_explicit

        cache_mod._redis = None


# ── Ollama API detection tests ───────────────────────────────────────────


class TestOllamaAPIDetection:
    """_embed_via_ollama detects /api/embed vs /api/embeddings and uses correct format."""

    def _make_engine(self):
        from rag.pipeline import RAGEngine

        engine = RAGEngine()
        mock_client = MagicMock()
        engine._ollama = mock_client
        return engine, mock_client

    @patch("rag.pipeline.config.EMBED_MODEL", "test-model")
    @patch("rag.pipeline.config.ENABLE_CACHE", False)
    def test_new_api_uses_input_field(self) -> None:
        engine, mock_client = self._make_engine()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[0.1, 0.2]]}
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        cached_map: dict[int, list[float]] = {}
        with patch("rag.pipeline.config.OLLAMA_EMBED_URL", "http://localhost:11434/api/embed"):
            engine._embed_via_ollama(mock_client, [(0, "hello")], cached_map, "test-model")

        call_json = mock_client.post.call_args[1]["json"]
        assert "input" in call_json
        assert call_json["input"] == "hello"
        assert "prompt" not in call_json

    @patch("rag.pipeline.config.EMBED_MODEL", "test-model")
    @patch("rag.pipeline.config.ENABLE_CACHE", False)
    def test_legacy_api_uses_prompt_field(self) -> None:
        engine, mock_client = self._make_engine()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embedding": [0.3, 0.4]}
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        cached_map: dict[int, list[float]] = {}
        with patch("rag.pipeline.config.OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings"):
            engine._embed_via_ollama(mock_client, [(0, "hello")], cached_map, "test-model")

        call_json = mock_client.post.call_args[1]["json"]
        assert "prompt" in call_json
        assert call_json["prompt"] == "hello"
        assert "input" not in call_json

    @patch("rag.pipeline.config.EMBED_MODEL", "test-model")
    @patch("rag.pipeline.config.ENABLE_CACHE", False)
    def test_new_api_caches_with_model(self) -> None:
        engine, mock_client = self._make_engine()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[0.1, 0.2]]}
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        cached_map: dict[int, list[float]] = {}
        with (
            patch("rag.pipeline.config.OLLAMA_EMBED_URL", "http://localhost:11434/api/embed"),
            patch("rag.services.cache.cache_embedding") as mock_cache,
        ):
            engine._embed_via_ollama(mock_client, [(0, "hello")], cached_map, "my-model")
            mock_cache.assert_called_once_with("hello", [0.1, 0.2], model="my-model")

    @patch("rag.pipeline.config.EMBED_MODEL", "test-model")
    @patch("rag.pipeline.config.ENABLE_CACHE", False)
    def test_populates_cached_map(self) -> None:
        engine, mock_client = self._make_engine()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[0.5, 0.6]]}
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        cached_map: dict[int, list[float]] = {}
        with patch("rag.pipeline.config.OLLAMA_EMBED_URL", "http://localhost:11434/api/embed"):
            engine._embed_via_ollama(mock_client, [(3, "world")], cached_map, "test-model")

        assert cached_map[3] == [0.5, 0.6]


# ── Query expansion fallback tests ───────────────────────────────────────


class TestQueryExpansionFallback:
    """QueryExpander._embed falls back to EMBED_MODEL_FALLBACK on failure."""

    def _make_expander(self):
        from rag.services.query_expansion import QueryExpander

        mock_ollama = MagicMock()
        expander = QueryExpander(mock_ollama)
        return expander, mock_ollama

    @patch("rag.services.query_expansion.config.EMBED_MODEL", "primary")
    @patch("rag.services.query_expansion.config.EMBED_MODEL_FALLBACK", "fallback")
    def test_uses_primary_model(self) -> None:
        expander, mock_ollama = self._make_expander()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[0.1]]}
        mock_resp.raise_for_status = MagicMock()
        mock_ollama.post.return_value = mock_resp

        with patch("rag.services.query_expansion.config.OLLAMA_EMBED_URL", "http://localhost:11434/api/embed"):
            result = expander._embed("test")

        assert result == [0.1]
        call_json = mock_ollama.post.call_args[1]["json"]
        assert call_json["model"] == "primary"

    @patch("rag.services.query_expansion.config.EMBED_MODEL", "primary")
    @patch("rag.services.query_expansion.config.EMBED_MODEL_FALLBACK", "fallback")
    def test_falls_back_on_failure(self) -> None:
        expander, mock_ollama = self._make_expander()

        call_count = 0

        def side_effect(url, json=None):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if json.get("model") == "primary":
                raise RuntimeError("primary unavailable")
            resp.json.return_value = {"embeddings": [[0.3]]}
            return resp

        mock_ollama.post.side_effect = side_effect

        with patch("rag.services.query_expansion.config.OLLAMA_EMBED_URL", "http://localhost:11434/api/embed"):
            result = expander._embed("test")

        assert call_count == 2
        assert result == [0.3]

    @patch("rag.services.query_expansion.config.EMBED_MODEL", "fallback")
    @patch("rag.services.query_expansion.config.EMBED_MODEL_FALLBACK", "fallback")
    def test_raises_when_fallback_also_fails(self) -> None:
        expander, mock_ollama = self._make_expander()
        mock_ollama.post.side_effect = RuntimeError("all failed")

        with (
            patch("rag.services.query_expansion.config.OLLAMA_EMBED_URL", "http://localhost:11434/api/embed"),
            pytest.raises(RuntimeError, match="all failed"),
        ):
            expander._embed("test")

    @patch("rag.services.query_expansion.config.EMBED_MODEL", "primary")
    @patch("rag.services.query_expansion.config.EMBED_MODEL_FALLBACK", "fallback")
    def test_embed_single_uses_new_api(self) -> None:
        expander, mock_ollama = self._make_expander()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[0.7]]}
        mock_resp.raise_for_status = MagicMock()
        mock_ollama.post.return_value = mock_resp

        with patch("rag.services.query_expansion.config.OLLAMA_EMBED_URL", "http://localhost:11434/api/embed"):
            result = expander._embed_single("test", "mymodel")

        assert result == [0.7]
        call_json = mock_ollama.post.call_args[1]["json"]
        assert call_json == {"model": "mymodel", "input": "test"}

    @patch("rag.services.query_expansion.config.EMBED_MODEL", "primary")
    @patch("rag.services.query_expansion.config.EMBED_MODEL_FALLBACK", "fallback")
    def test_embed_single_uses_legacy_api(self) -> None:
        expander, mock_ollama = self._make_expander()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embedding": [0.8]}
        mock_resp.raise_for_status = MagicMock()
        mock_ollama.post.return_value = mock_resp

        with patch("rag.services.query_expansion.config.OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings"):
            result = expander._embed_single("test", "mymodel")

        assert result == [0.8]
        call_json = mock_ollama.post.call_args[1]["json"]
        assert call_json == {"model": "mymodel", "prompt": "test"}


# ── Config fallback defaults tests ───────────────────────────────────────


class TestConfigFallbackDefaults:
    """Config has fallback model env vars."""

    def test_embed_model_fallback_exists(self) -> None:
        from rag import config

        assert hasattr(config, "EMBED_MODEL_FALLBACK")

    def test_rerank_model_fallback_exists(self) -> None:
        from rag import config

        assert hasattr(config, "RERANK_MODEL_FALLBACK")

    def test_rerank_type_exists(self) -> None:
        from rag import config

        assert hasattr(config, "RERANK_TYPE")
        assert config.RERANK_TYPE in ("cross-encoder", "causal-lm", "auto")
