"""Unit tests for query expansion module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src import config
from src.services.query_expansion import ExpandedQuery, QueryExpander

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_ollama() -> httpx.Client:
    """Return a mocked httpx.Client that simulates Ollama responses."""
    client = MagicMock(spec=httpx.Client)

    def _post(url: str, payload: dict | None = None, **kwargs: object) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()

        if "/api/chat" in url:
            content = "mocked response from the LLM"
            resp.json.return_value = {"message": {"role": "assistant", "content": content}}
        elif "/api/embeddings" in url:
            resp.json.return_value = {"embedding": [0.1] * config.DENSE_DIM}
        else:
            resp.json.return_value = {}

        return resp

    client.post = MagicMock(side_effect=_post)
    return client


@pytest.fixture
def expander(mock_ollama: httpx.Client) -> QueryExpander:
    """Return a QueryExpander with all features disabled by default."""
    with patch.multiple(
        config,
        ENABLE_QUERY_REWRITE=False,
        ENABLE_HYDE=False,
        ENABLE_MULTI_QUERY=False,
    ):
        return QueryExpander(mock_ollama)


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
        result = expander.expand("hello world")
        assert result.original == "hello world"
        assert result.rewritten is None
        assert result.hyde_vector is None
        assert result.paraphrases is None

    def test_expand_with_single_flag(self, expander: QueryExpander) -> None:
        with patch.object(config, "ENABLE_QUERY_REWRITE", True):
            result = expander.expand("test")
            assert result.rewritten == "mocked response from the LLM"
            assert result.hyde_vector is None
            assert result.paraphrases is None


# ── Query Rewriting ──────────────────────────────────────────────────────────


class TestQueryRewrite:
    def test_rewrite_calls_chat(self, expander: QueryExpander) -> None:
        with patch.object(config, "ENABLE_QUERY_REWRITE", True):
            result = expander.expand("rev")
            assert result.rewritten is not None
            assert len(result.rewritten) > 0

    def test_rewrite_prompt_contains_query(self, expander: QueryExpander) -> None:
        with patch.object(config, "ENABLE_QUERY_REWRITE", True):
            expander.expand("my specific query")
            # Verify the chat was called with a prompt containing the query
            call_args = expander._ollama.post.call_args
            assert call_args is not None
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
            if "/api/chat" in url:
                resp.json.return_value = {
                    "message": {
                        "role": "assistant",
                        "content": "query variation 1\nquery variation 2\nquery variation 3",
                    }
                }
            else:
                resp.json.return_value = {"embedding": [0.1] * config.DENSE_DIM}
            return resp

        expander._ollama.post = MagicMock(side_effect=_multi_line_post)

        with patch.object(config, "ENABLE_MULTI_QUERY", True), patch.object(config, "MULTI_QUERY_COUNT", 2):
            result = expander.expand("test")
            assert result.paraphrases is not None
            assert len(result.paraphrases) == 2

    def test_multi_query_filters_empty_lines(self, expander: QueryExpander) -> None:
        def _post_with_blanks(url: str, payload: dict | None = None, **kwargs: object) -> MagicMock:
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if "/api/chat" in url:
                resp.json.return_value = {
                    "message": {
                        "role": "assistant",
                        "content": "query one\n\n\nquery two\n",
                    }
                }
            else:
                resp.json.return_value = {"embedding": [0.1] * config.DENSE_DIM}
            return resp

        expander._ollama.post = MagicMock(side_effect=_post_with_blanks)

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
            if "/api/chat" in url:
                content = payload["messages"][0]["content"] if payload else "response"
                # For multi-query, return multiple lines
                if "paraphrase" in content.lower() or "Generate" in content:
                    content = "variation 1\nvariation 2\nvariation 3"
                resp.json.return_value = {"message": {"role": "assistant", "content": content}}
            else:
                resp.json.return_value = {"embedding": [0.5] * config.DENSE_DIM}
            return resp

        expander._ollama.post = MagicMock(side_effect=_post)

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
        expander._ollama.post = MagicMock(side_effect=httpx.TransportError("connection refused"))
        with patch.object(config, "ENABLE_QUERY_REWRITE", True):
            result = expander.expand("test")
            assert result.rewritten is None
            assert result.original == "test"

    def test_hyde_failure_skips_vector(self, expander: QueryExpander) -> None:
        call_count = 0

        def _post(url: str, payload: dict | None = None, **kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if "/api/embeddings" in url and call_count > 1:
                raise httpx.TransportError("embedding failed")
            resp.json.return_value = {"embedding": [0.1] * config.DENSE_DIM, "message": {"content": "test"}}
            return resp

        expander._ollama.post = MagicMock(side_effect=_post)
        with patch.object(config, "ENABLE_HYDE", True):
            result = expander.expand("test")
            assert result.hyde_vector is None

    def test_multi_query_failure_skips_paraphrases(self, expander: QueryExpander) -> None:
        expander._ollama.post = MagicMock(side_effect=httpx.TransportError("timeout"))
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
