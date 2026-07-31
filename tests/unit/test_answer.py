"""Unit tests for citation-based answer generation."""

from __future__ import annotations

import pytest

from rag.answer import (
    Answer,
    Citation,
    _citation_confidence,
    _is_refusal,
    _pack_context,
    _parse_citations,
    generate_answer,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_chunks() -> list[dict]:
    return [
        {
            "content": "The company reported revenue of $10M in Q3. Profit margins improved to 15%.",
            "source": "/docs/financial_report.pdf",
            "metadata": {"doc_type": "report"},
            "rerank_score": 0.92,
        },
        {
            "content": "Employee headcount grew by 20% this year. The engineering team expanded significantly.",
            "source": "/docs/financial_report.pdf",
            "metadata": {"doc_type": "report"},
            "rerank_score": 0.85,
        },
        {
            "content": "The market analysis shows strong demand in the APAC region. Competitor pricing remains stable.",
            "source": "/docs/market_analysis.pdf",
            "metadata": {"doc_type": "analysis"},
            "rerank_score": 0.78,
        },
    ]


# ── Context packing ───────────────────────────────────────────────────────────


class TestPackContext:
    def test_all_chunks_fit(self, sample_chunks: list[dict]) -> None:
        context, used = _pack_context(sample_chunks, max_context_chars=50000)
        assert len(used) == 3
        assert "[1] Source: /docs/financial_report.pdf" in context
        assert "[2] Source: /docs/financial_report.pdf" in context
        assert "[3] Source: /docs/market_analysis.pdf" in context

    def test_budget_limits_chunks(self, sample_chunks: list[dict]) -> None:
        _, used = _pack_context(sample_chunks, max_context_chars=200)
        assert len(used) < 3
        assert len(used) >= 1

    def test_truncation_of_long_chunk(self) -> None:
        long_chunk = {
            "content": "x" * 5000,
            "source": "/docs/long.txt",
            "metadata": {},
        }
        context, used = _pack_context([long_chunk], max_context_chars=300)
        assert len(used) == 1
        assert "[truncated]" in context
        assert len(context) <= 300 + 100  # header + truncation marker overhead

    def test_drops_tiny_chunks_after_truncation(self) -> None:
        tiny_chunk = {
            "content": "x" * 150,
            "source": "/docs/tiny.txt",
            "metadata": {},
        }
        context, used = _pack_context([tiny_chunk], max_context_chars=100)
        assert len(used) == 0
        assert context == ""

    def test_empty_chunks(self) -> None:
        context, used = _pack_context([], max_context_chars=1000)
        assert context == ""
        assert used == []

    def test_chunk_ordering_preserved(self) -> None:
        chunks = [
            {"content": "First chunk content " * 10, "source": "a.txt", "metadata": {}},
            {"content": "Second chunk content " * 10, "source": "b.txt", "metadata": {}},
        ]
        _, used = _pack_context(chunks, max_context_chars=500)
        assert used[0]["source"] == "a.txt"
        assert used[1]["source"] == "b.txt"


# ── Citation parsing ──────────────────────────────────────────────────────────


class TestParseCitations:
    def test_valid_citations(self, sample_chunks: list[dict]) -> None:
        text = "Revenue was $10M [1]. Headcount grew [2]."
        _, citations = _parse_citations(text, sample_chunks)
        assert len(citations) == 2
        assert citations[0].index == 1
        assert citations[0].source == "/docs/financial_report.pdf"
        assert citations[1].index == 2
        assert citations[1].source == "/docs/financial_report.pdf"

    def test_invalid_citation_stripped(self, sample_chunks: list[dict]) -> None:
        text = "Revenue was $10M [99]."
        result, citations = _parse_citations(text, sample_chunks)
        assert len(citations) == 0
        assert "[99]" not in result

    def test_negative_citation_not_matched(self, sample_chunks: list[dict]) -> None:
        text = "Something [-1] happened."
        result, citations = _parse_citations(text, sample_chunks)
        assert len(citations) == 0
        # [-1] doesn't match the \[(\d+)\] pattern (no negative sign), so it stays
        assert "[-1]" in result

    def test_multiple_citations_same_chunk(self, sample_chunks: list[dict]) -> None:
        text = "Revenue [1] and profit [1] both improved."
        _, citations = _parse_citations(text, sample_chunks)
        assert len(citations) == 1  # deduplicated
        assert citations[0].index == 1

    def test_citations_across_sources(self, sample_chunks: list[dict]) -> None:
        text = "Revenue [1] and market demand [3]."
        _, citations = _parse_citations(text, sample_chunks)
        assert len(citations) == 2
        assert citations[0].source == "/docs/financial_report.pdf"
        assert citations[1].source == "/docs/market_analysis.pdf"

    def test_no_citations(self, sample_chunks: list[dict]) -> None:
        text = "No citations here."
        result, citations = _parse_citations(text, sample_chunks)
        assert len(citations) == 0
        assert result == "No citations here."

    def test_empty_text(self, sample_chunks: list[dict]) -> None:
        result, citations = _parse_citations("", sample_chunks)
        assert citations == []
        assert result == ""

    def test_extra_spaces_cleaned(self, sample_chunks: list[dict]) -> None:
        text = "Hello  [1]  world  [99]"
        result, _ = _parse_citations(text, sample_chunks)
        assert "  " not in result


# ── Refusal detection ─────────────────────────────────────────────────────────


class TestRefusalDetection:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("INSUFFICIENT_CONTEXT", True),
            ("insufficient_context", True),
            ("I'm sorry, but INSUFFICIENT_CONTEXT", True),
            ("The documents contain INSUFFICIENT_CONTEXT to answer.", True),
            ("INSUFFICIENT_CONTEXT " + "word " * 50, False),
            ("", True),
            ("   \n  ", True),
            ("Revenue grew 12% year over year [1].", False),
        ],
        ids=[
            "exact-sentinel",
            "case-insensitive",
            "wrapped-in-prose",
            "full-sentence",
            "long-answer-not-refusal",
            "empty-string",
            "whitespace-only",
            "normal-answer",
        ],
    )
    def test_is_refusal(self, text: str, expected: bool) -> None:
        assert _is_refusal(text) is expected


# ── Confidence scoring ────────────────────────────────────────────────────────


class TestConfidenceScoring:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Revenue grew [1]. Margins fell [2]. Headcount rose [1].", 1.0),
            ("Revenue grew. Margins fell.", 0.0),
            ("Revenue grew [1]. Margins fell.", 0.5),
            ("", 0.0),
            ("Only this one [1].", 1.0),
            ("First [1]. Second. Third [3]. Fourth.", 0.5),
        ],
        ids=[
            "all-cited",
            "none-cited",
            "half-cited",
            "empty-text",
            "single-cited-sentence",
            "mixed-citations",
        ],
    )
    def test_confidence_values(self, text: str, expected: float) -> None:
        assert _citation_confidence(text) == expected


# ── Answer.formatted() ────────────────────────────────────────────────────────


class TestAnswerFormatted:
    def test_with_citations(self) -> None:
        answer = Answer(
            text="Revenue grew [1].",
            refused=False,
            confidence=1.0,
            citations=[
                Citation(index=1, source="/docs/report.pdf", chunk_text="...", metadata={}),
            ],
            sources=["/docs/report.pdf"],
        )
        formatted = answer.formatted()
        assert "Revenue grew [1]." in formatted
        assert "Sources:" in formatted
        assert "[1] /docs/report.pdf" in formatted

    def test_without_citations(self) -> None:
        answer = Answer(
            text="I don't know.",
            refused=True,
            confidence=0.0,
            citations=[],
            sources=[],
        )
        assert answer.formatted() == "I don't know."

    def test_deduplicated_sources(self) -> None:
        answer = Answer(
            text="A [1] and B [2].",
            refused=False,
            confidence=1.0,
            citations=[
                Citation(index=1, source="a.txt", chunk_text="", metadata={}),
                Citation(index=2, source="a.txt", chunk_text="", metadata={}),
                Citation(index=3, source="b.txt", chunk_text="", metadata={}),
            ],
            sources=["a.txt", "b.txt"],
        )
        formatted = answer.formatted()
        # "a.txt" appears once in Sources list (deduplicated), "b.txt" once
        assert formatted.count("a.txt") == 1
        assert formatted.count("b.txt") == 1
        # Both source lines present
        assert "[1] a.txt" in formatted
        assert "[3] b.txt" in formatted

    def test_repr(self) -> None:
        answer = Answer(text="ok", refused=False, confidence=0.8, citations=[1], sources=[])
        assert "1 citations" in repr(answer)
        assert "0.80" in repr(answer)

    def test_repr_refused(self) -> None:
        answer = Answer(text="no", refused=True, confidence=0.0, citations=[], sources=[])
        assert "refused" in repr(answer)


# ── generate_answer integration ───────────────────────────────────────────────


class TestGenerateAnswer:
    @pytest.mark.asyncio
    async def test_basic_answer(self, sample_chunks: list[dict]) -> None:
        async def mock_chat(prompt: str) -> str:
            return "Revenue was $10M [1]. Headcount grew by 20% [2]."

        result = await generate_answer("What were the financials?", sample_chunks, mock_chat)
        assert result.refused is False
        assert len(result.citations) == 2
        assert result.confidence == 1.0
        assert "/docs/financial_report.pdf" in result.sources

    @pytest.mark.asyncio
    async def test_refusal(self, sample_chunks: list[dict]) -> None:
        async def mock_chat(prompt: str) -> str:
            return "INSUFFICIENT_CONTEXT"

        result = await generate_answer("What is the meaning of life?", sample_chunks, mock_chat)
        assert result.refused is True
        assert result.confidence == 0.0
        assert len(result.citations) == 0

    @pytest.mark.asyncio
    async def test_refusal_with_prose(self, sample_chunks: list[dict]) -> None:
        async def mock_chat(prompt: str) -> str:
            return "I'm sorry, but INSUFFICIENT_CONTEXT to answer that question."

        result = await generate_answer("Quantum physics?", sample_chunks, mock_chat)
        assert result.refused is True

    @pytest.mark.asyncio
    async def test_empty_chunks(self) -> None:
        async def mock_chat(prompt: str) -> str:
            return "should not be called"

        result = await generate_answer("What?", [], mock_chat)
        assert result.refused is True
        assert "No documents" in result.text

    @pytest.mark.asyncio
    async def test_llm_failure(self, sample_chunks: list[dict]) -> None:
        async def mock_chat(prompt: str) -> str:
            raise RuntimeError("LLM down")

        result = await generate_answer("What?", sample_chunks, mock_chat)
        assert result.refused is True
        assert "LLM error" in result.text

    @pytest.mark.asyncio
    async def test_citations_from_different_sources(self) -> None:
        chunks = [
            {"content": "Alpha info " * 20, "source": "alpha.pdf", "metadata": {}},
            {"content": "Beta info " * 20, "source": "beta.pdf", "metadata": {}},
        ]

        async def mock_chat(prompt: str) -> str:
            return "Alpha [1] and beta [2]."

        result = await generate_answer("Compare alpha and beta", chunks, mock_chat)
        assert result.sources == ["alpha.pdf", "beta.pdf"]
        assert len(result.citations) == 2

    @pytest.mark.asyncio
    async def test_truncated_chunks_still_usable(self) -> None:
        chunks = [
            {"content": "A" * 15000, "source": "big.txt", "metadata": {}},
        ]

        async def mock_chat(prompt: str) -> str:
            return "Something about the document [1]."

        result = await generate_answer("Tell me about the doc", chunks, mock_chat, max_context_chars=500)
        assert result.refused is False
        assert len(result.citations) == 1

    @pytest.mark.asyncio
    async def test_custom_sentinel(self, sample_chunks: list[dict]) -> None:
        async def mock_chat(prompt: str) -> str:
            return "CANNOT_ANSWER"

        result = await generate_answer(
            "What?",
            sample_chunks,
            mock_chat,
            refusal_sentinel="CANNOT_ANSWER",
        )
        assert result.refused is True

    @pytest.mark.asyncio
    async def test_partial_citations(self, sample_chunks: list[dict]) -> None:
        async def mock_chat(prompt: str) -> str:
            return "Revenue was good [1]. Some opinion without citation."

        result = await generate_answer("How is business?", sample_chunks, mock_chat)
        assert result.refused is False
        assert result.confidence < 1.0
        assert result.confidence == pytest.approx(0.5)
