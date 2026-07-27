"""Unit tests for pipeline chunking integration."""

from __future__ import annotations

from unittest.mock import patch

from rag.pipeline import create_chunks

_LONG_TEXT = "## Header\n\nSome text content that is long enough to pass min chunk length test\n\nMore text here"


class TestCreateChunksTextOnly:
    def test_returns_empty_list_for_empty_text(self) -> None:
        chunks = create_chunks(text="")
        assert chunks == []

    def test_uses_recursive_fallback_when_no_source(self) -> None:
        chunks = create_chunks(text=_LONG_TEXT)
        assert len(chunks) > 0
        assert all("content" in c for c in chunks)

    def test_uses_recursive_fallback_for_recursive_strategy(self) -> None:
        chunks = create_chunks(_LONG_TEXT, source_identifier="/some/file.pdf")
        assert len(chunks) > 0


class TestCreateChunksHybridStrategy:
    def test_uses_hybrid_chunker_when_available(self) -> None:
        mock_chunks = [
            {
                "content": "Chunk 1 content long enough",
                "section_header": "Header 1",
                "headings": ["Header 1"],
                "chunk_index": 0,
            },
            {
                "content": "Chunk 2 content also long enough",
                "section_header": "Header 2",
                "headings": ["Header 2"],
                "chunk_index": 1,
            },
        ]
        with (
            patch("rag.pipeline.config.CHUNK_STRATEGY", "hybrid"),
            patch("rag.pipeline.config.MIN_CHUNK_LEN", 5),
            patch("rag.chunking.chunk_file", return_value={"chunks": mock_chunks, "markdown": ""}),
        ):
            chunks = create_chunks(text="", source_identifier="/test/file.pdf")
            assert len(chunks) == 2
            assert chunks[0]["content"] == mock_chunks[0]["content"]
            assert chunks[1]["section_header"] == "Header 2"

    def test_falls_back_to_recursive_on_hybrid_failure(self) -> None:
        with (
            patch("rag.pipeline.config.CHUNK_STRATEGY", "hybrid"),
            patch("rag.pipeline.config.MIN_CHUNK_LEN", 5),
            patch(
                "rag.chunking.chunk_file",
                side_effect=RuntimeError("fail"),
            ),
        ):
            chunks = create_chunks(text=_LONG_TEXT, source_identifier="/test/file.pdf")
            assert len(chunks) > 0

    def test_falls_back_to_recursive_on_import_error(self) -> None:
        short_text = "## Header\n\nSome text content that is long enough\n\nMore text"
        with (
            patch("rag.pipeline.config.CHUNK_STRATEGY", "hybrid"),
            patch("rag.pipeline.config.MIN_CHUNK_LEN", 5),
            patch(
                "rag.chunking.chunk_file",
                side_effect=ImportError("no docling"),
            ),
        ):
            chunks = create_chunks(text=short_text, source_identifier="/test/file.pdf")
            assert len(chunks) > 0
