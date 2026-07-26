"""Unit tests for contextual retrieval module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from rag import config
from rag.services.contextual_retrieval import ContextGenerator, strip_context_prefix

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_ollama() -> httpx.Client:
    """Return a mocked httpx.Client that simulates Ollama responses."""
    client = MagicMock(spec=httpx.Client)

    def _post(url: str, payload: dict | None = None, **kwargs: object) -> MagicMock:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()

        if "/api/chat" in url:
            content = "This section discusses financial metrics."
            resp.json.return_value = {"message": {"role": "assistant", "content": content}}
        elif "/api/embeddings" in url:
            resp.json.return_value = {"embedding": [0.1] * config.DENSE_DIM}
        else:
            resp.json.return_value = {}

        return resp

    client.post = MagicMock(side_effect=_post)
    return client


@pytest.fixture
def generator(mock_ollama: httpx.Client) -> ContextGenerator:
    """Return a ContextGenerator — caller must patch CONTEXT_STRATEGY."""
    return ContextGenerator(mock_ollama)


# ── Header strategy ─────────────────────────────────────────────────────────


class TestHeaderStrategy:
    def test_header_returns_context_string(self, generator: ContextGenerator) -> None:
        with patch.object(config, "CONTEXT_STRATEGY", "header"):
            ctx = generator.generate_context(
                chunk="Revenue increased 15%",
                section_header="## Revenue",
            )
            assert ctx == "[Context: ## Revenue]"

    def test_empty_header_returns_empty(self, generator: ContextGenerator) -> None:
        with patch.object(config, "CONTEXT_STRATEGY", "header"):
            ctx = generator.generate_context(
                chunk="Revenue increased 15%",
                section_header="",
            )
            assert ctx == ""

    def test_header_with_other_params_ignored(self, generator: ContextGenerator) -> None:
        with patch.object(config, "CONTEXT_STRATEGY", "header"):
            ctx = generator.generate_context(
                chunk="Revenue increased 15%",
                section_header="## Revenue",
                document_summary="A financial report.",
                prev_chunk="Previous section.",
                next_chunk="Next section.",
            )
            assert ctx == "[Context: ## Revenue]"


# ── Summary strategy ────────────────────────────────────────────────────────


class TestSummaryStrategy:
    def test_summary_generates_context(self, generator: ContextGenerator) -> None:
        with patch.multiple(config, CONTEXT_STRATEGY="summary"):
            gen = ContextGenerator(generator._ollama)
            ctx = gen.generate_context(
                chunk="Revenue increased 15%",
                document_summary="Q3 2025 Financial Report for Acme Corp.",
            )
            assert ctx.startswith("[Context:")
            assert ctx.endswith("]")
            assert len(ctx) > len("[Context: ]")

    def test_summary_without_summary_returns_empty(self, generator: ContextGenerator) -> None:
        with patch.multiple(config, CONTEXT_STRATEGY="summary"):
            gen = ContextGenerator(generator._ollama)
            ctx = gen.generate_context(
                chunk="Revenue increased 15%",
                document_summary="",
            )
            assert ctx == ""


# ── Surrounding strategy ────────────────────────────────────────────────────


class TestSurroundingStrategy:
    def test_surrounding_with_neighbors(self, generator: ContextGenerator) -> None:
        with patch.multiple(config, CONTEXT_STRATEGY="surrounding"):
            gen = ContextGenerator(generator._ollama)
            ctx = gen.generate_context(
                chunk="Revenue increased 15%",
                prev_chunk="Operating expenses were $500M.",
                next_chunk="Net income reached $200M.",
            )
            assert ctx.startswith("[Context:")
            assert ctx.endswith("]")

    def test_surrounding_without_neighbors_returns_empty(self, generator: ContextGenerator) -> None:
        with patch.multiple(config, CONTEXT_STRATEGY="surrounding"):
            gen = ContextGenerator(generator._ollama)
            ctx = gen.generate_context(
                chunk="Revenue increased 15%",
                prev_chunk="",
                next_chunk="",
            )
            assert ctx == ""


# ── enrich_chunks ───────────────────────────────────────────────────────────


class TestEnrichChunks:
    def test_enrich_chunks_preserves_count(self, generator: ContextGenerator) -> None:
        chunks = [
            {"content": "First chunk.", "section_header": "## Intro"},
            {"content": "Second chunk.", "section_header": "## Body"},
            {"content": "Third chunk.", "section_header": "## Conclusion"},
        ]
        enriched = generator.enrich_chunks(chunks)
        assert len(enriched) == 3

    def test_enrich_chunks_adds_context_prefix(self, generator: ContextGenerator) -> None:
        chunks = [
            {"content": "First chunk.", "section_header": "## Intro"},
            {"content": "Second chunk.", "section_header": "## Body"},
        ]
        enriched = generator.enrich_chunks(chunks)
        for chunk in enriched:
            assert "context_prefix" in chunk

    def test_enrich_chunks_modifies_content_with_header(self, generator: ContextGenerator) -> None:
        chunks = [
            {"content": "Revenue increased.", "section_header": "## Revenue"},
        ]
        with patch.object(config, "CONTEXT_STRATEGY", "header"):
            enriched = generator.enrich_chunks(chunks)
            assert enriched[0]["content"].startswith("[Context: ## Revenue]")
            assert "Revenue increased." in enriched[0]["content"]

    def test_enrich_chunks_empty_list(self, generator: ContextGenerator) -> None:
        enriched = generator.enrich_chunks([])
        assert enriched == []

    def test_enrich_chunks_no_header_no_prefix(self, generator: ContextGenerator) -> None:
        chunks = [{"content": "Some text.", "section_header": ""}]
        enriched = generator.enrich_chunks(chunks)
        assert enriched[0]["context_prefix"] == ""
        assert enriched[0]["content"] == "Some text."


# ── generate_document_summary ──────────────────────────────────────────────


class TestDocumentSummary:
    def test_summary_calls_chat(self, generator: ContextGenerator) -> None:
        summary = generator.generate_document_summary("This is a test document about AI.")
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_summary_truncates_long_input(self, generator: ContextGenerator) -> None:
        long_text = "word " * 10000
        summary = generator.generate_document_summary(long_text)
        assert isinstance(summary, str)


# ── strip_context_prefix ──────────────────────────────────────────────────


class TestStripContextPrefix:
    def test_strips_header_context(self) -> None:
        content = "[Context: ## Revenue] Revenue increased 15%."
        result = strip_context_prefix(content)
        assert result == "Revenue increased 15%."

    def test_strips_summary_context(self) -> None:
        content = "[Context: From Q3 2025 Financial Report.] Revenue increased."
        result = strip_context_prefix(content)
        assert result == "Revenue increased."

    def test_no_prefix_unchanged(self) -> None:
        content = "Revenue increased 15%."
        result = strip_context_prefix(content)
        assert result == "Revenue increased 15%."

    def test_empty_string(self) -> None:
        assert strip_context_prefix("") == ""

    def test_context_with_special_chars(self) -> None:
        content = "[Context: Section (v2.0): Overview] Some content."
        result = strip_context_prefix(content)
        assert result == "Some content."


# ── Error handling ───────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_chat_failure_propagates(self, mock_ollama: httpx.Client) -> None:
        mock_ollama.post = MagicMock(side_effect=httpx.TransportError("connection refused"))
        gen = ContextGenerator(mock_ollama)
        with pytest.raises(httpx.TransportError):
            gen.generate_document_summary("test")

    def test_enrich_chunks_with_chat_failure(self, mock_ollama: httpx.Client) -> None:
        """When using summary/surrounding strategy and chat fails, context is empty."""
        mock_ollama.post = MagicMock(side_effect=httpx.TransportError("timeout"))
        with patch.multiple(config, CONTEXT_STRATEGY="summary"):
            gen = ContextGenerator(mock_ollama)
            chunks = [{"content": "Test.", "section_header": ""}]
            enriched = gen.enrich_chunks(chunks)
            # Should not raise, context prefix should be empty or contain error marker
            assert len(enriched) == 1
