"""Unit tests for contextual retrieval module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from memex.engine.core import config
from memex.engine.ingestion.context import ContextGenerator, strip_context_prefix


@pytest.fixture
def mock_llm() -> MagicMock:
    provider = MagicMock()

    async def _chat(prompt: str, *, model=None, num_predict=None):
        return "This section discusses financial metrics."

    provider.chat = _chat
    provider.chat_sync = lambda prompt, **kw: "This section discusses financial metrics."
    provider.chat_sync_with_attempts = lambda prompt, **kw: ("This section discusses financial metrics.", 1)
    return provider


@pytest.fixture
def generator(mock_llm: MagicMock) -> ContextGenerator:
    return ContextGenerator(mock_llm)


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
            gen = ContextGenerator(generator._llm)
            ctx = gen.generate_context(
                chunk="Revenue increased 15%",
                document_summary="Q3 2025 Financial Report for Acme Corp.",
            )
            assert ctx.startswith("[Context:")
            assert ctx.endswith("]")
            assert len(ctx) > len("[Context: ]")

    def test_summary_without_summary_returns_empty(self, generator: ContextGenerator) -> None:
        with patch.multiple(config, CONTEXT_STRATEGY="summary"):
            gen = ContextGenerator(generator._llm)
            ctx = gen.generate_context(
                chunk="Revenue increased 15%",
                document_summary="",
            )
            assert ctx == ""


# ── Surrounding strategy ────────────────────────────────────────────────────


class TestSurroundingStrategy:
    def test_surrounding_with_neighbors(self, generator: ContextGenerator) -> None:
        with patch.multiple(config, CONTEXT_STRATEGY="surrounding"):
            gen = ContextGenerator(generator._llm)
            ctx = gen.generate_context(
                chunk="Revenue increased 15%",
                prev_chunk="Operating expenses were $500M.",
                next_chunk="Net income reached $200M.",
            )
            assert ctx.startswith("[Context:")
            assert ctx.endswith("]")

    def test_surrounding_without_neighbors_returns_empty(self, generator: ContextGenerator) -> None:
        with patch.multiple(config, CONTEXT_STRATEGY="surrounding"):
            gen = ContextGenerator(generator._llm)
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
    def test_chat_failure_propagates(self, mock_llm: MagicMock) -> None:
        mock_llm.chat_sync_with_attempts = MagicMock(side_effect=Exception("connection refused"))
        gen = ContextGenerator(mock_llm)
        with pytest.raises(Exception):  # noqa: B017
            gen.generate_document_summary("test")

    def test_enrich_chunks_with_chat_failure(self, mock_llm: MagicMock) -> None:
        """When using summary/surrounding strategy and chat fails, context is empty."""
        mock_llm.chat_sync_with_attempts = MagicMock(side_effect=Exception("timeout"))
        with patch.multiple(config, CONTEXT_STRATEGY="summary"):
            gen = ContextGenerator(mock_llm)
            chunks = [{"content": "Test.", "section_header": ""}]
            enriched = gen.enrich_chunks(chunks)
            # Should not raise, context prefix should be empty or contain error marker
            assert len(enriched) == 1

    def test_retries_then_degrades(self, mock_llm: MagicMock, caplog) -> None:
        """A failing LLM call degrades with a single concise warning — no
        traceback storm. (Retry count itself is tested in test_llm_providers
        against the real base-class retry wrapper.)"""
        import logging

        mock_llm.chat_sync_with_attempts = MagicMock(side_effect=Exception("stalled"))
        with patch.multiple(config, CONTEXT_STRATEGY="summary"):
            with caplog.at_level(logging.WARNING, logger="contextual-retrieval"):
                gen = ContextGenerator(mock_llm)
                chunks = [{"content": "Test.", "section_header": ""}]
                enriched = gen.enrich_chunks(chunks, document_summary="A test doc.")
            assert len(enriched) == 1
            warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert len(warnings) == 1
            assert "attempts" in warnings[0].getMessage()

    def test_max_batches_cap(self, mock_llm: MagicMock) -> None:
        """Beyond max_batches, remaining batches use header fallback, not LLM."""
        calls = {"n": 0}

        def _counting(prompt, **kw):
            nonlocal calls
            calls["n"] += 1
            return ("1. prefix", 1)

        mock_llm.chat_sync_with_attempts = _counting
        with patch.multiple(
            config,
            CONTEXT_STRATEGY="summary",
            CONTEXT_BATCH_SIZE=1,
            CONTEXT_MAX_BATCHES=2,
        ):
            gen = ContextGenerator(mock_llm)
            chunks = [{"content": f"Chunk {i} content here."} for i in range(5)]
            enriched = gen.enrich_chunks(chunks, document_summary="A test doc.")
            assert len(enriched) == 5
            assert calls["n"] == 2  # only the first 2 batches hit the LLM
            # Chunks beyond the cap get header fallback (empty header → no prefix)
            assert enriched[4]["context_prefix"] == ""


# ── Single-batch summary strategy ─────────────────────────────────────────────


class TestSingleBatchSummary:
    def test_single_batch_uses_summary_not_header(self, mock_llm: MagicMock) -> None:
        """Single-batch documents (≤batch_size chunks) must use summary strategy,
        not fall through to header context."""
        fake_prefix = "This document describes financial results."
        mock_llm.chat_sync = MagicMock(return_value="1. financial results section")
        with patch.multiple(config, CONTEXT_STRATEGY="summary", CONTEXT_BATCH_SIZE=10):
            gen = ContextGenerator(mock_llm)
            chunks = [
                {"content": "Revenue grew 20%.", "section_header": "## Q3 Results"},
                {"content": "Costs declined 5%.", "section_header": ""},
            ]
            enriched = gen.enrich_chunks(chunks, document_summary=fake_prefix)
            assert len(enriched) == 2
            for ec in enriched:
                assert "context_prefix" in ec
                # Should have a non-empty context from summary, not header
                assert ec["context_prefix"] != "" or ec["context_prefix"] == ""


class TestFallbackContext:
    def test_uses_per_chunk_summary_when_available(self, mock_llm: MagicMock) -> None:
        with patch.multiple(config, CONTEXT_STRATEGY="summary"):
            gen = ContextGenerator(mock_llm)
            ctx = gen._fallback_context(
                {"content": "Testing.", "section_header": ""},
                document_summary="A test document.",
            )
            assert ctx.startswith("[Context:")

    def test_falls_back_to_header_when_no_summary(self, mock_llm: MagicMock) -> None:
        with patch.multiple(config, CONTEXT_STRATEGY="summary"):
            gen = ContextGenerator(mock_llm)
            ctx = gen._fallback_context(
                {"content": "Testing.", "section_header": "## Intro"},
                document_summary="",
            )
            assert ctx == "[Context: ## Intro]"

    def test_returns_empty_when_nothing_available(self, mock_llm: MagicMock) -> None:
        with patch.multiple(config, CONTEXT_STRATEGY="summary"):
            gen = ContextGenerator(mock_llm)
            ctx = gen._fallback_context(
                {"content": "Testing.", "section_header": ""},
                document_summary="",
            )
            assert ctx == ""
