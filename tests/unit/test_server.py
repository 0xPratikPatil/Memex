"""Unit tests for MCP server tools."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memex.mcp.schemas import IngestBatchInput, IngestFileInput, IngestUrlInput, QueryInput
from memex.mcp.server import rag_ingest_batch, rag_ingest_file, rag_ingest_url, rag_query


@pytest.fixture
def mock_ctx() -> MagicMock:
    """Mock MCP Context with async report_progress."""
    ctx = MagicMock()
    ctx.report_progress = AsyncMock()
    return ctx


# ── rag_ingest_file tests ──────────────────────────────────────────────────


class TestRagIngestFile:
    """Test MCP tool for ingesting local files."""

    @pytest.mark.asyncio
    async def test_ingests_local_file_successfully(self, tmp_path: Path, mock_ctx: MagicMock) -> None:
        """Should ingest a local file and return success message."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"Test content for ingestion")

        with patch("memex.engine.ingestion.loader.parse_file") as mock_parse:
            mock_result = MagicMock()
            mock_result.ok = True
            mock_result.markdown = "# Converted content"
            mock_result.processing_time = 1.5
            mock_parse.return_value = mock_result

            with patch("memex.mcp.server._get_engine") as mock_engine:
                mock_engine.return_value.check_unmodified_local.return_value = (False, 0)
                mock_engine.return_value.compute_file_hash.return_value = "abc123def456"
                mock_engine.return_value.is_already_ingested.return_value = (False, 0)
                mock_engine.return_value.ingest_text.return_value = 5

                result = await rag_ingest_file(IngestFileInput(file_path_or_url=str(test_file)), mock_ctx)

                assert "Successfully ingested" in result
                assert str(test_file) in result
                assert "5 chunks" in result
                assert "1.5s" in result

    @pytest.mark.asyncio
    async def test_returns_error_when_docling_fails(self, tmp_path: Path, mock_ctx: MagicMock) -> None:
        """Should return error message when Docling conversion fails."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"Test content")

        with (
            patch("memex.mcp.server._get_engine") as _mock_engine,
            patch("memex.engine.ingestion.loader.parse_file") as mock_parse,
        ):
            mock_result = MagicMock()
            mock_result.ok = False
            mock_result.status = "failure"
            mock_result.errors = ["Conversion failed"]
            mock_parse.return_value = mock_result

            result = await rag_ingest_file(IngestFileInput(file_path_or_url=str(test_file)), mock_ctx)

            assert "failure" in result or "Error" in result

    @pytest.mark.asyncio
    async def test_skips_already_ingested_file(self, tmp_path: Path, mock_ctx: MagicMock) -> None:
        """Should skip ingestion if file unchanged (pre-check via check_unmodified_local)."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"Test content")

        with patch("memex.mcp.server._get_engine") as mock_engine:
            mock_engine.return_value.check_unmodified_local.return_value = (True, 10)

            result = await rag_ingest_file(IngestFileInput(file_path_or_url=str(test_file)), mock_ctx)

            assert "Already ingested" in result
            assert "10 chunks" in result
            assert "skipping" in result

    @pytest.mark.asyncio
    async def test_handles_file_not_found_error(self, mock_ctx: MagicMock) -> None:
        """Should handle FileNotFoundError gracefully."""
        mock_engine = MagicMock()
        mock_engine.check_unmodified_local.return_value = (False, 0)

        with (
            patch("memex.mcp.server._get_engine", return_value=mock_engine),
            patch("memex.engine.ingestion.loader.parse_file") as mock_parse,
        ):
            mock_parse.side_effect = FileNotFoundError("File not found: /nonexistent/file.txt")

            result = await rag_ingest_file(IngestFileInput(file_path_or_url="/nonexistent/file.txt"), mock_ctx)

            assert "File not found" in result

    @pytest.mark.asyncio
    async def test_handles_generic_exception(self, tmp_path: Path, mock_ctx: MagicMock) -> None:
        """Should handle generic exceptions gracefully."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"Test content")

        mock_engine = MagicMock()
        mock_engine.check_unmodified_local.return_value = (False, 0)

        with (
            patch("memex.mcp.server._get_engine", return_value=mock_engine),
            patch("memex.engine.ingestion.loader.parse_file") as mock_parse,
        ):
            mock_parse.side_effect = RuntimeError("Unexpected error")

            result = await rag_ingest_file(IngestFileInput(file_path_or_url=str(test_file)), mock_ctx)

            assert "Unexpected error" in result

    @pytest.mark.asyncio
    async def test_extracts_content_type_from_extension(self, tmp_path: Path, mock_ctx: MagicMock) -> None:
        """Should extract content type from file extension."""
        test_file = tmp_path / "report.pdf"
        test_file.write_bytes(b"PDF content")

        with patch("memex.engine.ingestion.loader.parse_file") as mock_parse:
            mock_result = MagicMock()
            mock_result.ok = True
            mock_result.markdown = "# PDF content"
            mock_result.processing_time = 2.0
            mock_parse.return_value = mock_result

            with patch("memex.mcp.server._get_engine") as mock_engine:
                mock_engine.return_value.check_unmodified_local.return_value = (False, 0)
                mock_engine.return_value.compute_file_hash.return_value = "abc123def456"
                mock_engine.return_value.is_already_ingested.return_value = (False, 0)
                mock_engine.return_value.ingest_text.return_value = 3

                await rag_ingest_file(IngestFileInput(file_path_or_url=str(test_file)), mock_ctx)

                # Verify metadata was passed with content_type
                call_kwargs = mock_engine.return_value.ingest_text.call_args[1]
                assert call_kwargs["metadata"]["content_type"] == "pdf"

    @pytest.mark.asyncio
    async def test_ingest_text_not_called_with_docling_json(self, tmp_path: Path, mock_ctx: MagicMock) -> None:
        """ingest_text() must NOT receive docling_json — it doesn't accept that parameter."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"Test content")

        with patch("memex.engine.ingestion.loader.parse_file") as mock_parse:
            mock_result = MagicMock()
            mock_result.ok = True
            mock_result.markdown = "# Content"
            mock_result.json_content = {"some": "data"}
            mock_result.processing_time = 1.0
            mock_parse.return_value = mock_result

            with patch("memex.mcp.server._get_engine") as mock_engine:
                mock_engine.return_value.check_unmodified_local.return_value = (False, 0)
                mock_engine.return_value.compute_file_hash.return_value = "abc123"
                mock_engine.return_value.is_already_ingested.return_value = (False, 0)
                mock_engine.return_value.ingest_text.return_value = 3

                await rag_ingest_file(IngestFileInput(file_path_or_url=str(test_file)), mock_ctx)

                call_kwargs = mock_engine.return_value.ingest_text.call_args[1]
                assert "docling_json" not in call_kwargs, (
                    "ingest_text() received docling_json but pipeline doesn't accept it"
                )


# ── rag_ingest_url tests ───────────────────────────────────────────────────


class TestRagIngestUrl:
    """Test MCP tool for ingesting URLs."""

    @pytest.mark.asyncio
    async def test_ingest_url_not_called_with_docling_json(self, mock_ctx: MagicMock) -> None:
        """ingest_text() must NOT receive docling_json from rag_ingest_url."""
        with patch("memex.engine.ingestion.loader.parse_url") as mock_parse:
            mock_result = MagicMock()
            mock_result.ok = True
            mock_result.markdown = "# Web content"
            mock_result.json_content = {"some": "data"}
            mock_result.processing_time = 2.0
            mock_parse.return_value = mock_result

            with patch("memex.mcp.server._get_engine") as mock_engine:
                mock_engine.return_value.compute_file_hash.return_value = "abc123"
                mock_engine.return_value.is_already_ingested.return_value = (False, 0)
                mock_engine.return_value.ingest_text.return_value = 4

                await rag_ingest_url(IngestUrlInput(url="https://example.com/doc"), mock_ctx)

                call_kwargs = mock_engine.return_value.ingest_text.call_args[1]
                assert "docling_json" not in call_kwargs, (
                    "ingest_text() received docling_json but pipeline doesn't accept it"
                )


# ── rag_ingest_batch tests ─────────────────────────────────────────────────


class TestRagIngestBatch:
    """Test MCP tool for batch ingesting multiple items."""

    @pytest.mark.asyncio
    async def test_ingest_batch_delegates_to_orchestrator(self, mock_ctx: MagicMock) -> None:
        """rag_ingest_batch should delegate to IngestionOrchestrator."""
        items = ["https://example.com/doc"]
        with patch("memex.engine.ingestion.ingestion.IngestionOrchestrator") as mock_orch_class:
            mock_orch = mock_orch_class.return_value
            mock_orch.ingest_batch = AsyncMock(return_value={items[0]: "Success (2 chunks, 1.0s conversion)"})

            result = await rag_ingest_batch(
                IngestBatchInput(items=items),
                mock_ctx,
            )

            mock_orch.ingest_batch.assert_called_once_with(items)
            assert "Success" in result[items[0]]


# ── rag_query tests ────────────────────────────────────────────────────────


class TestRagQuery:
    """Test MCP tool for searching the RAG knowledge base."""

    @pytest.fixture
    def mock_engine(self) -> MagicMock:
        engine = MagicMock()
        engine.hybrid_search.return_value = [
            {
                "id": "chunk-1",
                "source": "/docs/report.pdf",
                "content": "Revenue was $10M in Q3.",
                "section_header": "## Financials",
                "context_prefix": "",
                "rrf_score": 0.0167,
                "rerank_score": 0.92,
                "doc_type": "report",
                "topics": ["finance"],
                "language": "en",
                "keywords": ["revenue"],
                "entities": {},
                "dates": [],
            },
            {
                "id": "chunk-2",
                "source": "/docs/report.pdf",
                "content": "Headcount grew by 20%.",
                "section_header": "## Growth",
                "context_prefix": "",
                "rrf_score": 0.0158,
                "rerank_score": 0.85,
                "doc_type": "report",
                "topics": ["finance"],
                "language": "en",
                "keywords": ["headcount"],
                "entities": {},
                "dates": [],
            },
        ]
        engine.mmr_search.return_value = [
            {
                "id": "chunk-1",
                "source": "/docs/report.pdf",
                "content": "Revenue was $10M in Q3.",
                "section_header": "## Financials",
                "context_prefix": "",
                "dense_score": 0.95,
                "doc_type": "report",
                "topics": ["finance"],
                "language": "en",
                "keywords": ["revenue"],
                "entities": {},
                "dates": [],
            },
            {
                "id": "chunk-3",
                "source": "/docs/market.pdf",
                "content": "Market trends show growth.",
                "section_header": "## Market",
                "context_prefix": "",
                "dense_score": 0.82,
                "doc_type": "analysis",
                "topics": ["market"],
                "language": "en",
                "keywords": ["trends"],
                "entities": {},
                "dates": [],
            },
        ]
        return engine

    @pytest.mark.asyncio
    async def test_hybrid_search_returns_results(self, mock_engine: MagicMock) -> None:
        """Should return search results in markdown format by default."""
        with (
            patch("memex.mcp.server._get_engine", return_value=mock_engine),
            patch("memex.mcp.server.config") as mock_config,
        ):
            mock_config.ENABLE_QUERY_EXPANSION = False
            mock_config.SEARCH_MODE = "hybrid"
            mock_config.ENABLE_ANSWER = False
            mock_config.CHARACTER_LIMIT = 25000

            result = await rag_query(QueryInput(query="revenue", top_k=5, use_reranking=True))

            assert isinstance(result, str)
            assert "Search Results" in result
            assert "revenue" in result.lower()
            mock_engine.hybrid_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_mmr_search_mode(self, mock_engine: MagicMock) -> None:
        """Should use MMR search when search_mode='mmr'."""
        with (
            patch("memex.mcp.server._get_engine", return_value=mock_engine),
            patch("memex.mcp.server.config") as mock_config,
        ):
            mock_config.ENABLE_QUERY_EXPANSION = False
            mock_config.SEARCH_MODE = "hybrid"
            mock_config.ENABLE_ANSWER = False
            mock_config.CHARACTER_LIMIT = 25000
            mock_config.MMR_FETCH_K = 20
            mock_config.MMR_LAMBDA_MULT = 0.5

            result = await rag_query(QueryInput(query="revenue", top_k=5, search_mode="mmr"))

            assert isinstance(result, str)
            assert "Search Results" in result
            mock_engine.mmr_search.assert_called_once()
            mock_engine.hybrid_search.assert_not_called()

    @pytest.mark.asyncio
    async def test_json_output_format(self, mock_engine: MagicMock) -> None:
        """Should return QueryOutput in JSON format."""
        from memex.mcp.schemas import ResponseFormat

        with (
            patch("memex.mcp.server._get_engine", return_value=mock_engine),
            patch("memex.mcp.server.config") as mock_config,
        ):
            mock_config.ENABLE_QUERY_EXPANSION = False
            mock_config.SEARCH_MODE = "hybrid"
            mock_config.ENABLE_ANSWER = False

            result = await rag_query(
                QueryInput(
                    query="revenue",
                    top_k=5,
                    response_format=ResponseFormat.JSON,
                )
            )

            from memex.mcp.schemas import QueryOutput

            assert isinstance(result, QueryOutput)
            assert result.total == 2
            assert result.count == 2
            assert len(result.results) == 2
            assert result.results[0].source == "/docs/report.pdf"

    @pytest.mark.asyncio
    async def test_empty_results(self, mock_engine: MagicMock) -> None:
        """Should handle empty search results gracefully."""
        mock_engine.hybrid_search.return_value = []

        with (
            patch("memex.mcp.server._get_engine", return_value=mock_engine),
            patch("memex.mcp.server.config") as mock_config,
        ):
            mock_config.ENABLE_QUERY_EXPANSION = False
            mock_config.SEARCH_MODE = "hybrid"
            mock_config.ENABLE_ANSWER = False

            result = await rag_query(QueryInput(query="nonexistent", top_k=5))

            assert "No results found" in result

    @pytest.mark.asyncio
    async def test_answer_generation_json(self, mock_engine: MagicMock) -> None:
        """Should generate cited answer when answer generation is enabled."""
        with (
            patch("memex.mcp.server._get_engine", return_value=mock_engine),
            patch("memex.mcp.server.config") as mock_config,
        ):
            mock_config.ENABLE_QUERY_EXPANSION = False
            mock_config.SEARCH_MODE = "hybrid"
            mock_config.ENABLE_ANSWER = True
            mock_config.CHAT_MODEL = "qwen2.5:1.5b"
            mock_config.OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
            mock_config.ANSWER_MAX_CONTEXT_CHARS = 12000

            with patch("memex.engine.generation.answers.generate_answer") as mock_gen:
                from memex.engine.generation.answers import Answer, Citation

                mock_gen.return_value = Answer(
                    text="Revenue was $10M [1]. Headcount grew [2].",
                    refused=False,
                    confidence=1.0,
                    citations=[
                        Citation(index=1, source="/docs/report.pdf", chunk_text="Revenue was $10M.", metadata={}),
                        Citation(index=2, source="/docs/report.pdf", chunk_text="Headcount grew.", metadata={}),
                    ],
                    sources=["/docs/report.pdf"],
                )

                from memex.mcp.schemas import AnswerOutput, ResponseFormat

                result = await rag_query(
                    QueryInput(
                        query="revenue",
                        top_k=5,
                        response_format=ResponseFormat.JSON,
                        generate_answer=True,
                    )
                )

                assert isinstance(result, AnswerOutput)
                assert result.refused is False
                assert result.confidence == 1.0
                assert len(result.citations) == 2
                assert result.search_mode == "hybrid"

    @pytest.mark.asyncio
    async def test_answer_generation_markdown(self, mock_engine: MagicMock) -> None:
        """Should generate cited answer in markdown format."""
        with (
            patch("memex.mcp.server._get_engine", return_value=mock_engine),
            patch("memex.mcp.server.config") as mock_config,
        ):
            mock_config.ENABLE_QUERY_EXPANSION = False
            mock_config.SEARCH_MODE = "hybrid"
            mock_config.ENABLE_ANSWER = True
            mock_config.CHAT_MODEL = "qwen2.5:1.5b"
            mock_config.OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
            mock_config.ANSWER_MAX_CONTEXT_CHARS = 12000

            with patch("memex.engine.generation.answers.generate_answer") as mock_gen:
                from memex.engine.generation.answers import Answer, Citation

                mock_gen.return_value = Answer(
                    text="Revenue was $10M [1].",
                    refused=False,
                    confidence=1.0,
                    citations=[
                        Citation(index=1, source="/docs/report.pdf", chunk_text="Revenue was $10M.", metadata={}),
                    ],
                    sources=["/docs/report.pdf"],
                )

                result = await rag_query(QueryInput(query="revenue", top_k=5, generate_answer=True))

                assert isinstance(result, str)
                assert "Revenue was $10M [1]." in result
                assert "Sources:" in result
                assert "[1] /docs/report.pdf" in result

    @pytest.mark.asyncio
    async def test_answer_generation_refusal(self, mock_engine: MagicMock) -> None:
        """Should handle model refusal gracefully."""
        with (
            patch("memex.mcp.server._get_engine", return_value=mock_engine),
            patch("memex.mcp.server.config") as mock_config,
        ):
            mock_config.ENABLE_QUERY_EXPANSION = False
            mock_config.SEARCH_MODE = "hybrid"
            mock_config.ENABLE_ANSWER = True
            mock_config.CHAT_MODEL = "qwen2.5:1.5b"
            mock_config.OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
            mock_config.ANSWER_MAX_CONTEXT_CHARS = 12000

            with patch("memex.engine.generation.answers.generate_answer") as mock_gen:
                from memex.engine.generation.answers import Answer

                mock_gen.return_value = Answer(
                    text="The retrieved documents do not contain enough information to answer this question.",
                    refused=True,
                    confidence=0.0,
                    citations=[],
                    sources=[],
                )

                result = await rag_query(QueryInput(query="quantum physics", top_k=5, generate_answer=True))

                assert isinstance(result, str)
                assert "not contain enough information" in result

    @pytest.mark.asyncio
    async def test_answer_generation_llm_failure_fallback(self, mock_engine: MagicMock) -> None:
        """Should fall back to raw results if answer generation LLM fails."""
        with (
            patch("memex.mcp.server._get_engine", return_value=mock_engine),
            patch("memex.mcp.server.config") as mock_config,
        ):
            mock_config.ENABLE_QUERY_EXPANSION = False
            mock_config.SEARCH_MODE = "hybrid"
            mock_config.ENABLE_ANSWER = True
            mock_config.CHAT_MODEL = "qwen2.5:1.5b"
            mock_config.OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
            mock_config.ANSWER_MAX_CONTEXT_CHARS = 12000

            with patch("memex.engine.generation.answers.generate_answer") as mock_gen:
                from memex.engine.generation.answers import Answer

                mock_gen.return_value = Answer(
                    text="Answer generation failed due to an LLM error.",
                    refused=True,
                    confidence=0.0,
                    citations=[],
                    sources=[],
                )

                result = await rag_query(QueryInput(query="revenue", top_k=5, generate_answer=True))

                # Should return the refusal message as markdown
                assert isinstance(result, str)
                assert "LLM error" in result

    @pytest.mark.asyncio
    async def test_mmr_search_json_format(self, mock_engine: MagicMock) -> None:
        """Should return QueryOutput for MMR search in JSON format."""
        from memex.mcp.schemas import ResponseFormat

        with (
            patch("memex.mcp.server._get_engine", return_value=mock_engine),
            patch("memex.mcp.server.config") as mock_config,
        ):
            mock_config.ENABLE_QUERY_EXPANSION = False
            mock_config.SEARCH_MODE = "hybrid"
            mock_config.ENABLE_ANSWER = False
            mock_config.MMR_FETCH_K = 20
            mock_config.MMR_LAMBDA_MULT = 0.5

            result = await rag_query(
                QueryInput(
                    query="revenue",
                    top_k=5,
                    search_mode="mmr",
                    response_format=ResponseFormat.JSON,
                )
            )

            from memex.mcp.schemas import QueryOutput

            assert isinstance(result, QueryOutput)
            assert result.total == 2
            assert len(result.results) == 2

    @pytest.mark.asyncio
    async def test_search_mode_override_config(self, mock_engine: MagicMock) -> None:
        """Should use search_mode parameter over config default."""
        with (
            patch("memex.mcp.server._get_engine", return_value=mock_engine),
            patch("memex.mcp.server.config") as mock_config,
        ):
            mock_config.ENABLE_QUERY_EXPANSION = False
            mock_config.SEARCH_MODE = "hybrid"  # config default
            mock_config.ENABLE_ANSWER = False
            mock_config.MMR_FETCH_K = 20
            mock_config.MMR_LAMBDA_MULT = 0.5

            await rag_query(QueryInput(query="revenue", top_k=5, search_mode="mmr"))

            # MMR should be used even though config says hybrid
            mock_engine.mmr_search.assert_called_once()
