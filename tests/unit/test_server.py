"""Unit tests for MCP server tools."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from memex.server import rag_ingest_file

# ── rag_ingest_file tests ──────────────────────────────────────────────────


class TestRagIngestFile:
    """Test MCP tool for ingesting local files."""

    @pytest.mark.asyncio
    async def test_ingests_local_file_successfully(self, tmp_path: Path) -> None:
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
                mock_engine.return_value.compute_file_hash.return_value = "abc123def456"
                mock_engine.return_value.is_already_ingested.return_value = (False, 0)
                mock_engine.return_value.ingest_text.return_value = 5

                result = await rag_ingest_file(str(test_file))

                assert "Successfully ingested" in result
                assert str(test_file) in result
                assert "5 chunks" in result
                assert "1.5s" in result

    @pytest.mark.asyncio
    async def test_returns_error_when_docling_fails(self, tmp_path: Path) -> None:
        """Should return error message when Docling conversion fails."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"Test content")

        with patch("rag.docling_client.parse_file") as mock_parse:
            mock_result = MagicMock()
            mock_result.ok = False
            mock_result.status = "failure"
            mock_result.errors = ["Conversion failed"]
            mock_parse.return_value = mock_result

            result = await rag_ingest_file(str(test_file))

            assert "Error" in result
            assert "failure" in result
            assert "Conversion failed" in result

    @pytest.mark.asyncio
    async def test_skips_already_ingested_file(self, tmp_path: Path) -> None:
        """Should skip ingestion if file already ingested with same hash."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"Test content")

        with patch("rag.docling_client.parse_file") as mock_parse:
            mock_result = MagicMock()
            mock_result.ok = True
            mock_result.markdown = "# Content"
            mock_result.processing_time = 1.0
            mock_parse.return_value = mock_result

            with patch("memex.server._get_engine") as mock_engine:
                mock_engine.return_value.compute_file_hash.return_value = "abc123def456"
                mock_engine.return_value.is_already_ingested.return_value = (True, 10)

                result = await rag_ingest_file(str(test_file))

                assert "Already ingested" in result
                assert "10 chunks" in result
                assert "skipping" in result

    @pytest.mark.asyncio
    async def test_handles_file_not_found_error(self) -> None:
        """Should handle FileNotFoundError gracefully."""
        with patch("rag.docling_client.parse_file") as mock_parse:
            mock_parse.side_effect = FileNotFoundError("File not found: /nonexistent/file.txt")

            result = await rag_ingest_file("/nonexistent/file.txt")

            assert "Error" in result
            assert "File not found" in result

    @pytest.mark.asyncio
    async def test_handles_generic_exception(self, tmp_path: Path) -> None:
        """Should handle generic exceptions gracefully."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"Test content")

        with patch("rag.docling_client.parse_file") as mock_parse:
            mock_parse.side_effect = RuntimeError("Unexpected error")

            result = await rag_ingest_file(str(test_file))

            assert "Error" in result
            assert "Unexpected error" in result

    @pytest.mark.asyncio
    async def test_extracts_content_type_from_extension(self, tmp_path: Path) -> None:
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
                mock_engine.return_value.compute_file_hash.return_value = "abc123def456"
                mock_engine.return_value.is_already_ingested.return_value = (False, 0)
                mock_engine.return_value.ingest_text.return_value = 3

                await rag_ingest_file(str(test_file))

                # Verify metadata was passed with content_type
                call_kwargs = mock_engine.return_value.ingest_text.call_args[1]
                assert call_kwargs["metadata"]["content_type"] == "pdf"
