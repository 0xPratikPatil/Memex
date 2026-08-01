"""Integration tests for contextual retrieval with live Ollama."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from memex.engine.core import config
from memex.engine.ingestion.context import ContextGenerator, strip_context_prefix

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ollama_client() -> httpx.Client:
    """Live httpx.Client pointed at local Ollama. Skips if unreachable."""
    client = httpx.Client(timeout=config.HTTP_TIMEOUT)
    try:
        resp = client.get(f"http://localhost:{config.OLLAMA_PORT}/")
        resp.raise_for_status()
    except Exception:
        client.close()
        pytest.skip("Ollama not reachable at localhost:11434")
    yield client
    client.close()


@pytest.fixture
def context_generator(ollama_client: httpx.Client) -> ContextGenerator:
    """ContextGenerator wired to live Ollama."""
    from memex.engine.llm.ollama import OllamaLLM

    llm = OllamaLLM(base_url="http://localhost:11434", model=config.CHAT_MODEL, timeout=60.0)
    return ContextGenerator(llm)


# ── generate_document_summary ────────────────────────────────────────────────


@pytest.mark.integration
class TestGenerateDocumentSummary:
    def test_returns_string(self, context_generator: ContextGenerator) -> None:
        result = context_generator.generate_document_summary(
            "This document describes a retrieval-augmented generation pipeline."
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_truncates_long_input(self, context_generator: ContextGenerator) -> None:
        long_text = "word " * 10000
        result = context_generator.generate_document_summary(long_text)
        assert isinstance(result, str)
        assert len(result) > 0


# ── generate_context (header strategy) ───────────────────────────────────────


@pytest.mark.integration
class TestGenerateContextHeader:
    def test_header_returns_context_string(self, context_generator: ContextGenerator) -> None:
        with patch.object(config, "CONTEXT_STRATEGY", "header"):
            ctx = context_generator.generate_context(
                chunk="Revenue grew 20% YoY.",
                section_header="## Q3 Financial Results",
            )
            assert ctx == "[Context: ## Q3 Financial Results]"

    def test_empty_header_returns_empty(self, context_generator: ContextGenerator) -> None:
        ctx = context_generator.generate_context(
            chunk="Revenue grew 20% YoY.",
            section_header="",
        )
        assert ctx == ""


# ── enrich_chunks ────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestEnrichChunks:
    def test_adds_context_prefix_and_modifies_content(self, context_generator: ContextGenerator) -> None:
        chunks = [
            {"content": "Introduction to the system.", "section_header": "## Overview"},
            {"content": "Technical architecture details.", "section_header": "## Architecture"},
        ]
        with patch.object(config, "CONTEXT_STRATEGY", "header"):
            enriched = context_generator.enrich_chunks(chunks)
            assert len(enriched) == 2
            assert enriched[0]["context_prefix"] == "[Context: ## Overview]"
            assert enriched[0]["content"].startswith("[Context: ## Overview]")
            assert enriched[1]["context_prefix"] == "[Context: ## Architecture]"

    def test_empty_list(self, context_generator: ContextGenerator) -> None:
        assert context_generator.enrich_chunks([]) == []

    def test_no_header_no_prefix(self, context_generator: ContextGenerator) -> None:
        chunks = [{"content": "Some standalone content.", "section_header": ""}]
        enriched = context_generator.enrich_chunks(chunks)
        assert enriched[0]["context_prefix"] == ""
        assert enriched[0]["content"] == "Some standalone content."


# ── strip_context_prefix ────────────────────────────────────────────────────


@pytest.mark.integration
class TestStripContextPrefix:
    def test_strips_header_prefix(self) -> None:
        assert strip_context_prefix("[Context: ## Intro] Hello world.") == "Hello world."

    def test_no_prefix_unchanged(self) -> None:
        assert strip_context_prefix("Just plain text.") == "Just plain text."

    def test_empty_string(self) -> None:
        assert strip_context_prefix("") == ""

    def test_context_with_special_chars(self) -> None:
        result = strip_context_prefix("[Context: Section (v2.0): Overview] Content here.")
        assert result == "Content here."
