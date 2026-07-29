"""Unit tests for chunking module (Docling Serve API-based)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag.chunking import (
    _build_chunking_options,
    _build_convert_options,
    _get_chunking_url,
    _parse_chunk_response,
    chunk_local_file,
    chunk_url,
    is_hybrid_chunker_available,
)


class TestGetChunkingUrl:
    def test_constructs_url_from_docling_url(self) -> None:
        with patch("rag.chunking.config.DOCLING_URL", "http://localhost:5001/v1/convert/source"):
            url = _get_chunking_url()
        assert url == "http://localhost:5001/v1/chunk/hybrid/source"

    def test_strips_existing_chunk_path(self) -> None:
        with patch("rag.chunking.config.DOCLING_URL", "http://host:5001/v1/convert/source"):
            url = _get_chunking_url()
        assert "/v1/chunk/hybrid/source" in url


class TestBuildChunkingOptions:
    def test_includes_chunk_size(self) -> None:
        with patch("rag.chunking.config.CHUNK_SIZE", 512):
            opts = _build_chunking_options()
        assert opts["max_tokens"] == 512

    def test_includes_tokenizer(self) -> None:
        with patch("rag.chunking.config.CHUNK_TOKENIZER", "BAAI/bge-m3"):
            opts = _build_chunking_options()
        assert opts["tokenizer"] == "BAAI/bge-m3"

    def test_includes_merge_peers(self) -> None:
        with patch("rag.chunking.config.CHUNK_MERGE_PEERS", True):
            opts = _build_chunking_options()
        assert opts["merge_peers"] is True


class TestBuildConvertOptions:
    def test_requests_markdown_output(self) -> None:
        opts = _build_convert_options()
        assert "md" in opts["to_formats"]

    def test_includes_ocr_setting(self) -> None:
        with patch("rag.chunking.config.ENABLE_OCR", True):
            opts = _build_convert_options()
        assert opts["do_ocr"] is True

    def test_includes_enrichments_when_enabled(self) -> None:
        with (
            patch("rag.chunking.config.DOCLING_ENRICH_CODE", True),
            patch("rag.chunking.config.DOCLING_ENRICH_FORMULA", True),
            patch("rag.chunking.config.DOCLING_PICTURE_CLASSIFY", True),
            patch("rag.chunking.config.DOCLING_CHART_EXTRACT", True),
        ):
            opts = _build_convert_options()
        assert opts["do_code_enrichment"] is True
        assert opts["do_formula_enrichment"] is True
        assert opts["do_picture_classification"] is True
        assert opts["do_chart_extraction"] is True

    def test_omits_enrichments_when_disabled(self) -> None:
        with (
            patch("rag.chunking.config.DOCLING_ENRICH_CODE", False),
            patch("rag.chunking.config.DOCLING_ENRICH_FORMULA", False),
            patch("rag.chunking.config.DOCLING_PICTURE_CLASSIFY", False),
            patch("rag.chunking.config.DOCLING_CHART_EXTRACT", False),
        ):
            opts = _build_convert_options()
        assert "do_code_enrichment" not in opts
        assert "do_formula_enrichment" not in opts
        assert "do_picture_classification" not in opts
        assert "do_chart_extraction" not in opts


class TestParseChunkResponse:
    def test_parses_chunks_correctly(self) -> None:
        data = {
            "chunks": [
                {
                    "text": "First chunk content here",
                    "headings": ["Chapter 1"],
                    "chunk_index": 0,
                },
                {
                    "text": "Second chunk content here",
                    "headings": ["Chapter 1", "Section 1.1"],
                    "chunk_index": 1,
                },
            ],
            "documents": [],
            "processing_time": 1.5,
        }
        result = _parse_chunk_response(data)
        chunks = result["chunks"]
        assert len(chunks) == 2
        assert chunks[0]["content"] == "First chunk content here"
        assert chunks[0]["section_header"] == "Chapter 1"
        assert chunks[1]["section_header"] == "Chapter 1"

    def test_skips_empty_chunks(self) -> None:
        data = {
            "chunks": [
                {"text": "Valid content", "headings": [], "chunk_index": 0},
                {"text": "", "headings": [], "chunk_index": 1},
                {"text": "  ", "headings": [], "chunk_index": 2},
            ],
            "documents": [],
            "processing_time": 0.5,
        }
        result = _parse_chunk_response(data)
        assert len(result["chunks"]) == 1

    def test_empty_headings_uses_empty_section_header(self) -> None:
        data = {
            "chunks": [
                {"text": "Some text", "headings": None, "chunk_index": 0},
            ],
            "documents": [],
            "processing_time": 0.1,
        }
        result = _parse_chunk_response(data)
        assert result["chunks"][0]["section_header"] == ""

    def test_empty_response_returns_empty_list(self) -> None:
        result = _parse_chunk_response({"chunks": [], "documents": [], "processing_time": 0.0})
        assert result["chunks"] == []

    def test_include_doc_returns_markdown(self) -> None:
        data = {
            "chunks": [{"text": "content", "headings": [], "chunk_index": 0}],
            "documents": [{"md_content": "# Title\n\nFull document text."}],
            "processing_time": 1.0,
        }
        result = _parse_chunk_response(data, include_doc=True)
        assert result["markdown"] == "# Title\n\nFull document text."
        assert len(result["chunks"]) == 1


class TestChunkUrl:
    def test_sends_correct_payload(self) -> None:
        with (
            patch("rag.chunking._post_chunking") as mock_post,
            patch("rag.chunking.config.DOCLING_URL", "http://localhost:5001/v1/convert/source"),
        ):
            mock_post.return_value = {
                "chunks": [{"text": "chunk content", "headings": ["Header"], "chunk_index": 0}],
                "documents": [],
                "processing_time": 1.0,
            }
            result = chunk_url("https://example.com/doc.pdf")
            chunks = result["chunks"]
            assert len(chunks) == 1
            call_payload = mock_post.call_args[0][0]
            assert call_payload["sources"] == [{"kind": "http", "url": "https://example.com/doc.pdf"}]
            assert "convert_options" in call_payload
            assert "chunking_options" in call_payload


class TestChunkLocalFile:
    def test_reads_file_and_sends_base64(self, tmp_path) -> None:
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"PDF content")

        with (
            patch("rag.chunking._post_chunking") as mock_post,
            patch("rag.chunking.config.DOCLING_URL", "http://localhost:5001/v1/convert/source"),
        ):
            mock_post.return_value = {
                "chunks": [{"text": "chunk", "headings": [], "chunk_index": 0}],
                "documents": [],
                "processing_time": 0.5,
            }
            result = chunk_local_file(str(test_file))
            chunks = result["chunks"]
            assert len(chunks) == 1
            call_payload = mock_post.call_args[0][0]
            assert call_payload["sources"][0]["kind"] == "file"
            assert call_payload["sources"][0]["filename"] == "test.pdf"

    def test_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="File not found"):
            chunk_local_file("/nonexistent/file.txt")


class TestIsHybridChunkerAvailable:
    def test_returns_true_when_healthy(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200

        with (
            patch("rag.chunking.httpx.Client") as mock_client_cls,
            patch("rag.chunking.config.DOCLING_URL", "http://localhost:5001/v1/convert/source"),
        ):
            mock_client = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = is_hybrid_chunker_available()
            assert result is True
            mock_client.get.assert_called_once_with("http://localhost:5001/health", timeout=5.0)

    def test_returns_false_on_error(self) -> None:
        with (
            patch("rag.chunking._chunking_client", None),
            patch("rag.chunking.config.DOCLING_URL", "http://localhost:5001/v1/convert/source"),
            patch("rag.chunking.config.DOCLING_TIMEOUT", 5.0),
            patch("rag.chunking.httpx.Client", side_effect=Exception("connection failed")),
        ):
            result = is_hybrid_chunker_available()
            assert result is False
