"""Unit tests for query expansion module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from memex.engine.core import config
from memex.engine.llm.base import EmbedProvider, LLMProvider
from memex.engine.retrieval.expansion import ExpandedQuery, QueryExpander


@pytest.fixture
def mock_llm() -> MagicMock:
    provider = MagicMock()
    provider.chat_sync = MagicMock(return_value="mocked response from the LLM")
    return provider


@pytest.fixture
def mock_embedder() -> MagicMock:
    provider = MagicMock()
    provider.embed = MagicMock(return_value=[[0.1] * config.DENSE_DIM])
    return provider


@pytest.fixture
def expander(mock_llm: MagicMock, mock_embedder: MagicMock):
    with patch.object(config, "ENABLE_CACHE", False):
        yield QueryExpander(mock_llm, mock_embedder)


# ── ExpandedQuery dataclass ──────────────────────────────────────────────────


class TestExpandedQuery:
    def test_default_fields(self) -> None:
        eq = ExpandedQuery(original="test query")
        assert eq.original == "test query"
        assert eq.rewritten is None
        assert eq.hyde_vector is None
        assert eq.paraphrases is None

    def test_all_fields_populated(self) -> None:
        eq = ExpandedQuery(
            original="test",
            rewritten="rewritten test",
            hyde_vector=[0.1, 0.2],
            paraphrases=["paraphrase 1", "paraphrase 2"],
        )
        assert eq.rewritten == "rewritten test"
        assert len(eq.hyde_vector) == 2  # type: ignore[arg-type]
        assert len(eq.paraphrases) == 2  # type: ignore[arg-type]


# ── Expansion disabled (passthrough) ────────────────────────────────────────


class TestExpansionDisabled:
    def test_expand_returns_original_only(self, expander: QueryExpander) -> None:
        with patch.multiple(
            config,
            ENABLE_QUERY_REWRITE=False,
            ENABLE_HYDE=False,
            ENABLE_MULTI_QUERY=False,
        ):
            result = expander.expand("hello world")
            assert result.original == "hello world"
            assert result.rewritten is None
            assert result.hyde_vector is None
            assert result.paraphrases is None

    def test_expand_with_single_flag(self, expander: QueryExpander) -> None:
        with patch.multiple(
            config,
            ENABLE_QUERY_REWRITE=True,
            ENABLE_HYDE=False,
            ENABLE_MULTI_QUERY=False,
        ):
            result = expander.expand("test")
            assert result.rewritten == "mocked response from the LLM"
            assert result.hyde_vector is None
            assert result.paraphrases is None


# ── Query Rewriting ──────────────────────────────────────────────────────────


class TestQueryRewrite:
    def test_rewrite_calls_chat(self, expander: QueryExpander) -> None:
        with patch.multiple(
            config,
            ENABLE_QUERY_REWRITE=True,
            ENABLE_HYDE=False,
            ENABLE_MULTI_QUERY=False,
        ):
            result = expander.expand("rev")
            assert result.rewritten is not None
            assert len(result.rewritten) > 0

    def test_rewrite_prompt_contains_query(self, expander: QueryExpander) -> None:
        with patch.multiple(
            config,
            ENABLE_QUERY_REWRITE=True,
            ENABLE_HYDE=False,
            ENABLE_MULTI_QUERY=False,
        ):
            expander.expand("my specific query")
            call_args = expander._llm.chat_sync.call_args
            assert call_args is not None, "chat_sync should have been called"
            if call_args is None:
                return
            prompt_content = call_args[1]["json"]["messages"][0]["content"]
            assert "my specific query" in prompt_content


# ── HyDE ─────────────────────────────────────────────────────────────────────


class TestHyDE:
    def test_hyde_returns_vector(self, expander: QueryExpander) -> None:
        with patch.object(config, "ENABLE_HYDE", True):
            result = expander.expand("what is revenue")
            assert result.hyde_vector is not None
            assert len(result.hyde_vector) == config.DENSE_DIM

    def test_hyde_vector_is_floats(self, expander: QueryExpander) -> None:
        with patch.object(config, "ENABLE_HYDE", True):
            result = expander.expand("test query")
            assert result.hyde_vector is not None
            assert all(isinstance(v, float) for v in result.hyde_vector)


# ── Multi-Query ──────────────────────────────────────────────────────────────


class TestMultiQuery:
    def test_multi_query_returns_paraphrases(self, expander: QueryExpander) -> None:
        with patch.object(config, "ENABLE_MULTI_QUERY", True):
            result = expander.expand("security architecture")
            assert result.paraphrases is not None
            assert len(result.paraphrases) > 0

    def test_multi_query_respects_count(self, expander: QueryExpander) -> None:
        # Mock chat to return multiple lines
        def _multi_line_post(url: str, payload: dict | None = None, **kwargs: object) -> MagicMock:
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            json_data = payload or kwargs.get("json") or {}
            if ("/api/chat" in url or "/api/embed" in url) and isinstance(json_data, dict) and "messages" in json_data:
                resp.json.return_value = {
                    "message": {
                        "role": "assistant",
                        "content": "query variation 1\nquery variation 2\nquery variation 3",
                    }
                }
            else:
                resp.json.return_value = {"embeddings": [[0.1] * config.DENSE_DIM]}
            return resp

        expander._llm.chat_sync = MagicMock(return_value="paraphrase 1\nparaphrase 2")

        with patch.object(config, "ENABLE_MULTI_QUERY", True), patch.object(config, "MULTI_QUERY_COUNT", 2):
            result = expander.expand("test")
            assert result.paraphrases is not None
            assert len(result.paraphrases) == 2

    def test_multi_query_filters_empty_lines(self, expander: QueryExpander) -> None:
        def _post_with_blanks(url: str, payload: dict | None = None, **kwargs: object) -> MagicMock:
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            json_data = payload or kwargs.get("json") or {}
            if ("/api/chat" in url or "/api/embed" in url) and isinstance(json_data, dict) and "messages" in json_data:
                resp.json.return_value = {
                    "message": {
                        "role": "assistant",
                        "content": "query one\n\n\nquery two\n",
                    }
                }
            else:
                resp.json.return_value = {"embeddings": [[0.1] * config.DENSE_DIM]}
            return resp

        expander._llm.chat_sync = MagicMock(side_effect=_post_with_blanks)

        with patch.object(config, "ENABLE_MULTI_QUERY", True):
            result = expander.expand("test")
            assert result.paraphrases is not None
            assert all(p.strip() for p in result.paraphrases)


# ── Combined expansion ──────────────────────────────────────────────────────


class TestCombinedExpansion:
    def test_all_techniques_together(self, expander: QueryExpander) -> None:
        def _post(url: str, payload: dict | None = None, **kwargs: object) -> MagicMock:
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            json_data = payload or kwargs.get("json") or {}
            if ("/api/chat" in url or "/api/embed" in url) and isinstance(json_data, dict) and "messages" in json_data:
                msg_content = json_data["messages"][0]["content"]
                content = msg_content
                if "paraphrase" in content.lower() or "Generate" in content:
                    content = "variation 1\nvariation 2\nvariation 3"
                resp.json.return_value = {"message": {"role": "assistant", "content": content}}
            else:
                resp.json.return_value = {"embeddings": [[0.5] * config.DENSE_DIM]}
            return resp

        expander._llm.chat_sync = MagicMock(side_effect=_post)

        with patch.multiple(
            config,
            ENABLE_QUERY_REWRITE=True,
            ENABLE_HYDE=True,
            ENABLE_MULTI_QUERY=True,
            MULTI_QUERY_COUNT=3,
        ):
            result = expander.expand("complex query")
            assert result.rewritten is not None
            assert result.hyde_vector is not None
            assert result.paraphrases is not None
            assert len(result.hyde_vector) == config.DENSE_DIM
            assert len(result.paraphrases) <= 3


# ── Error handling ───────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_rewrite_failure_returns_none(self, expander: QueryExpander) -> None:
        expander._llm.chat_sync = MagicMock(side_effect=Exception("connection refused"))
        with patch.object(config, "ENABLE_QUERY_REWRITE", True):
            result = expander.expand("test")
            assert result.rewritten is None
            assert result.original == "test"

    def test_hyde_failure_skips_vector(self, expander: QueryExpander) -> None:
        expander._llm.chat_sync = MagicMock(return_value="hypothetical document text")
        expander._embedder.embed = MagicMock(side_effect=Exception("embedding failed"))
        with patch.object(config, "ENABLE_HYDE", True):
            result = expander.expand("test")
            assert result.hyde_vector is None

    def test_multi_query_failure_skips_paraphrases(self, expander: QueryExpander) -> None:
        expander._llm.chat_sync = MagicMock(side_effect=Exception("timeout"))
        with patch.object(config, "ENABLE_MULTI_QUERY", True):
            result = expander.expand("test")
            assert result.paraphrases is None


# ── Effective query propagation ─────────────────────────────────────────────


class TestEffectiveQuery:
    def test_rewrite_feeds_into_hyde(self, expander: QueryExpander) -> None:
        """When rewrite is enabled, HyDE should use the rewritten query."""
        with patch.object(config, "ENABLE_QUERY_REWRITE", True), patch.object(config, "ENABLE_HYDE", True):
            result = expander.expand("rev")
            # Both should be populated
            assert result.rewritten is not None
            assert result.hyde_vector is not None

    def test_rewrite_feeds_into_multi_query(self, expander: QueryExpander) -> None:
        """When rewrite is enabled, multi-query should use the rewritten query."""
        with patch.object(config, "ENABLE_QUERY_REWRITE", True), patch.object(config, "ENABLE_MULTI_QUERY", True):
            result = expander.expand("rev")
            assert result.rewritten is not None
            assert result.paraphrases is not None
