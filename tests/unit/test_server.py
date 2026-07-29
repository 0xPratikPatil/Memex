"""Unit tests for MCP server tools."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memex.schemas import IngestBatchInput, IngestFileInput, IngestUrlInput
from memex.server import rag_ingest_batch, rag_ingest_file, rag_ingest_url


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

        with patch("rag.docling_client.parse_file") as mock_parse:
            mock_result = MagicMock()
            mock_result.ok = True
            mock_result.markdown = "# Converted content"
            mock_result.processing_time = 1.5
            mock_parse.return_value = mock_result

            with patch("memex.server._get_engine") as mock_engine:
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

        with patch("rag.docling_client.parse_file") as mock_parse:
            mock_result = MagicMock()
            mock_result.ok = False
            mock_result.status = "failure"
            mock_result.errors = ["Conversion failed"]
            mock_parse.return_value = mock_result

            result = await rag_ingest_file(IngestFileInput(file_path_or_url=str(test_file)), mock_ctx)

            assert "Error" in result
            assert "failure" in result
            assert "Conversion failed" in result

    @pytest.mark.asyncio
    async def test_skips_already_ingested_file(self, tmp_path: Path, mock_ctx: MagicMock) -> None:
        """Should skip ingestion if file unchanged (pre-check via check_unmodified_local)."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"Test content")

        with patch("memex.server._get_engine") as mock_engine:
            mock_engine.return_value.check_unmodified_local.return_value = (True, 10)

            result = await rag_ingest_file(IngestFileInput(file_path_or_url=str(test_file)), mock_ctx)

            assert "Already ingested" in result
            assert "10 chunks" in result
            assert "skipping" in result

    @pytest.mark.asyncio
    async def test_handles_file_not_found_error(self, mock_ctx: MagicMock) -> None:
        """Should handle FileNotFoundError gracefully."""
        with patch("rag.docling_client.parse_file") as mock_parse:
            mock_parse.side_effect = FileNotFoundError("File not found: /nonexistent/file.txt")

            result = await rag_ingest_file(IngestFileInput(file_path_or_url="/nonexistent/file.txt"), mock_ctx)

            assert "Error" in result
            assert "File not found" in result

    @pytest.mark.asyncio
    async def test_handles_generic_exception(self, tmp_path: Path, mock_ctx: MagicMock) -> None:
        """Should handle generic exceptions gracefully."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"Test content")

        with patch("rag.docling_client.parse_file") as mock_parse:
            mock_parse.side_effect = RuntimeError("Unexpected error")

            result = await rag_ingest_file(IngestFileInput(file_path_or_url=str(test_file)), mock_ctx)

            assert "Error" in result
            assert "Unexpected error" in result

    @pytest.mark.asyncio
    async def test_extracts_content_type_from_extension(self, tmp_path: Path, mock_ctx: MagicMock) -> None:
        """Should extract content type from file extension."""
        test_file = tmp_path / "report.pdf"
        test_file.write_bytes(b"PDF content")

        with patch("rag.docling_client.parse_file") as mock_parse:
            mock_result = MagicMock()
            mock_result.ok = True
            mock_result.markdown = "# PDF content"
            mock_result.processing_time = 2.0
            mock_parse.return_value = mock_result

            with patch("memex.server._get_engine") as mock_engine:
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

        with patch("rag.docling_client.parse_file") as mock_parse:
            mock_result = MagicMock()
            mock_result.ok = True
            mock_result.markdown = "# Content"
            mock_result.json_content = {"some": "data"}
            mock_result.processing_time = 1.0
            mock_parse.return_value = mock_result

            with patch("memex.server._get_engine") as mock_engine:
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
        with patch("rag.docling_client.parse_url") as mock_parse:
            mock_result = MagicMock()
            mock_result.ok = True
            mock_result.markdown = "# Web content"
            mock_result.json_content = {"some": "data"}
            mock_result.processing_time = 2.0
            mock_parse.return_value = mock_result

            with patch("memex.server._get_engine") as mock_engine:
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
        with patch("rag.ingestion.IngestionOrchestrator") as mock_orch_class:
            mock_orch = mock_orch_class.return_value
            mock_orch.ingest_batch = AsyncMock(return_value={
                items[0]: "Success (2 chunks, 1.0s conversion)"
            })

            result = await rag_ingest_batch(
                IngestBatchInput(items=items),
                mock_ctx,
            )

            mock_orch.ingest_batch.assert_called_once_with(items)
            assert "Success" in result[items[0]]
