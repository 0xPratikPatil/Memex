"""Integration tests for query expansion with live Ollama."""

from __future__ import annotations

import httpx
import pytest

from memex.engine.core import config
from memex.engine.retrieval.expansion import ExpandedQuery, QueryExpander


def _check_ollama(host: str = "localhost", port: int = 11434) -> bool:
    """Return True if Ollama HTTP API is reachable."""
    try:
        with httpx.Client(timeout=httpx.Timeout(5.0)) as client:
            resp = client.get(f"http://{host}:{port}/api/version")
            return resp.is_success
    except Exception:
        return False


@pytest.mark.integration
class TestQueryExpanderIntegration:
    @pytest.fixture(autouse=True)
    def _skip_if_no_ollama(self) -> None:
        if not _check_ollama():
            pytest.skip("Ollama not reachable at localhost:11434")

    @pytest.fixture
    def ollama_client(self) -> httpx.Client:
        return httpx.Client(timeout=60.0)

    @pytest.fixture
    def expander(self, ollama_client: httpx.Client, monkeypatch: pytest.MonkeyPatch) -> QueryExpander:
        monkeypatch.setattr(config, "ENABLE_QUERY_EXPANSION", True)
        monkeypatch.setattr(config, "ENABLE_QUERY_REWRITE", True)
        monkeypatch.setattr(config, "ENABLE_MULTI_QUERY", True)
        monkeypatch.setattr(config, "CHAT_MODEL", "qwen3.5:0.8b")
        return QueryExpander(ollama_client)

    def test_rewrite_returns_string(self, expander: QueryExpander) -> None:
        rewritten = expander._rewrite("what is machine learning")
        assert isinstance(rewritten, str)
        assert len(rewritten) > 0

    def test_multi_query_returns_paraphrases(self, expander: QueryExpander) -> None:
        paraphrases = expander._multi_query("what is machine learning")
        assert isinstance(paraphrases, list)
        assert len(paraphrases) > 0
        assert all(isinstance(p, str) and len(p) > 0 for p in paraphrases)

    def test_expand_returns_expanded_query_with_all_fields(
        self, expander: QueryExpander, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(config, "ENABLE_HYDE", True)
        result = expander.expand("what is machine learning")
        assert isinstance(result, ExpandedQuery)
        assert result.original == "what is machine learning"
        assert result.rewritten is not None
        assert isinstance(result.rewritten, str)
        assert result.paraphrases is not None
        assert isinstance(result.paraphrases, list)

    def test_chat_error_returns_none_gracefully(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bad_client = httpx.Client(timeout=httpx.Timeout(1.0))
        monkeypatch.setattr(config, "ENABLE_QUERY_REWRITE", True)
        monkeypatch.setattr(config, "ENABLE_MULTI_QUERY", True)
        monkeypatch.setattr(config, "CHAT_MODEL", "qwen3.5:0.8b")
        monkeypatch.setattr(config, "OLLAMA_EMBED_URL", "http://localhost:19999/api/embeddings")
        expander = QueryExpander(bad_client)
        result = expander.expand("test query")
        assert result.rewritten is None
        assert result.original == "test query"
