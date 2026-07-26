"""Unit tests for pipeline chunking integration."""

from __future__ import annotations

from unittest.mock import patch

from rag.pipeline import create_chunks

_LONG_TEXT = "## Header\n\nSome text content that is long enough to pass min chunk length test\n\nMore text here"


class TestCreateChunksTextOnly:
    def test_returns_empty_list_for_empty_text(self):
        chunks = create_chunks(text="")
        assert chunks == []

    def test_uses_recursive_fallback_when_no_docling_json(self):
        chunks = create_chunks(text=_LONG_TEXT)
        assert len(chunks) > 0
        assert all("content" in c for c in chunks)

    def test_uses_recursive_fallback_for_recursive_strategy(self):
        chunks = create_chunks(_LONG_TEXT, docling_json={})
        assert len(chunks) > 0


class TestCreateChunksHybridStrategy:
    def test_uses_hybrid_chunker_when_available(self):
        docling_json = {"name": "TestDoc", "body": {"children": []}}
        mock_chunks = [
            {
                "content": "Chunk 1 content long enough",
                "section_header": "Header 1",
                "heading_level": 1,
                "chunk_type": "text",
            },
            {
                "content": "Chunk 2 content also long enough",
                "section_header": "Header 2",
                "heading_level": 2,
                "chunk_type": "text",
            },
        ]
        with (
            patch("rag.pipeline.config.CHUNK_STRATEGY", "hybrid"),
            patch("rag.pipeline.config.MIN_CHUNK_LEN", 5),
            patch("rag.chunking.chunk_docling_document", return_value=mock_chunks),
        ):
            chunks = create_chunks(text="", docling_json=docling_json)
            assert len(chunks) == 2
            assert chunks[0]["content"] == mock_chunks[0]["content"]
            assert chunks[1]["section_header"] == "Header 2"

    def test_falls_back_to_recursive_on_hybrid_failure(self):
        docling_json = {"name": "TestDoc"}
        with (
            patch("rag.pipeline.config.CHUNK_STRATEGY", "hybrid"),
            patch("rag.pipeline.config.MIN_CHUNK_LEN", 5),
            patch(
                "rag.chunking.chunk_docling_document",
                side_effect=RuntimeError("fail"),
            ),
        ):
            chunks = create_chunks(text=_LONG_TEXT, docling_json=docling_json)
            assert len(chunks) > 0

    def test_falls_back_to_recursive_on_import_error(self):
        short_text = "## Header\n\nSome text content that is long enough\n\nMore text"
        docling_json = {"name": "TestDoc"}
        with (
            patch("rag.pipeline.config.CHUNK_STRATEGY", "hybrid"),
            patch("rag.pipeline.config.MIN_CHUNK_LEN", 5),
            patch("rag.pipeline.create_chunks", wraps=create_chunks),
            patch(
                "rag.chunking.chunk_docling_document",
                side_effect=ImportError("no docling"),
            ),
        ):
            chunks = create_chunks(text=short_text, docling_json=docling_json)
            assert len(chunks) > 0
