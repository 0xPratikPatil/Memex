"""Unit tests for chunking module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag.chunking import (
    _serialize_chunk,
    _serialize_code_chunk,
    _serialize_image_chunk,
    _serialize_table_chunk,
    chunk_docling_document,
    is_hybrid_chunker_available,
)


def _make_chunk(text: str, chunk_type: str = "text", **meta_kwargs: object) -> MagicMock:
    """Build a mock base_chunk with .text and .meta attributes."""
    chunk = MagicMock()
    chunk.text = text
    chunk.meta.chunk_type = chunk_type
    chunk.meta.code_language = None
    chunk.meta.image_caption = None
    for key, value in meta_kwargs.items():
        setattr(chunk.meta, key, value)
    return chunk


class TestSerializeTableChunk:
    """Tests for _serialize_table_chunk."""

    def test_wraps_non_html_in_table_tags(self) -> None:
        """Should wrap plain table text in <table> tags."""
        chunk = _make_chunk("col1 | col2\nval1 | val2")

        result = _serialize_table_chunk(chunk)

        assert result == "<table>\ncol1 | col2\nval1 | val2\n</table>"

    def test_passes_through_existing_html_table_tag(self) -> None:
        """Should return unmodified text when it already contains <table>."""
        existing = "<table>\n<tr><td>col1</td></tr>\n</table>"
        chunk = _make_chunk(existing)

        result = _serialize_table_chunk(chunk)

        assert result == existing

    def test_passes_through_existing_html_tr_tag(self) -> None:
        """Should return unmodified text when it already contains <tr>."""
        existing = "<tr><td>val</td></tr>"
        chunk = _make_chunk(existing)

        result = _serialize_table_chunk(chunk)

        assert result == existing


class TestSerializeCodeChunk:
    """Tests for _serialize_code_chunk."""

    def test_wraps_non_fenced_code(self) -> None:
        """Should wrap code in ``` fence when not already fenced."""
        chunk = _make_chunk("print('hello')")

        result = _serialize_code_chunk(chunk)

        assert result == "```\nprint('hello')\n```"

    def test_wraps_with_language_tag(self) -> None:
        """Should include language in fence when code_language is set."""
        chunk = _make_chunk("print('hello')", code_language="python")

        result = _serialize_code_chunk(chunk)

        assert result == "```python\nprint('hello')\n```"

    def test_passes_through_existing_fence(self) -> None:
        """Should return unmodified when code already starts with ```."""
        existing = "```python\nprint('hello')\n```"
        chunk = _make_chunk(existing)

        result = _serialize_code_chunk(chunk)

        assert result == existing

    def test_empty_language_attribute_yields_no_lang_tag(self) -> None:
        """Should not include language when code_language is empty string."""
        chunk = _make_chunk("const x = 1;", code_language="")

        result = _serialize_code_chunk(chunk)

        assert result == "```\nconst x = 1;\n```"

    def test_empty_text_is_fenced(self) -> None:
        """Should still wrap empty text in fences."""
        chunk = _make_chunk("")

        result = _serialize_code_chunk(chunk)

        assert result == "```\n\n```"


class TestSerializeImageChunk:
    """Tests for _serialize_image_chunk."""

    def test_formats_with_caption(self) -> None:
        """Should format caption as [Image: ...] when image_caption is set."""
        chunk = _make_chunk("raw description", image_caption="A diagram of the architecture")

        result = _serialize_image_chunk(chunk)

        assert result == "[Image: A diagram of the architecture]"

    def test_returns_text_when_no_caption(self) -> None:
        """Should return original text when no image_caption attribute."""
        chunk = _make_chunk("raw description")
        # Ensure image_caption is not present
        del chunk.meta.image_caption

        result = _serialize_image_chunk(chunk)

        assert result == "raw description"

    def test_returns_text_when_caption_is_none(self) -> None:
        """Should return original text when image_caption is None."""
        chunk = _make_chunk("raw description", image_caption=None)

        result = _serialize_image_chunk(chunk)

        assert result == "raw description"

    def test_returns_text_when_caption_is_empty(self) -> None:
        """Should return original text when image_caption is empty string."""
        chunk = _make_chunk("raw description", image_caption="")

        result = _serialize_image_chunk(chunk)

        assert result == "raw description"


class TestSerializeChunk:
    """Tests for _serialize_chunk dispatcher."""

    def test_dispatches_table_by_type(self) -> None:
        """Should use table serializer when chunk_type is 'table'."""
        chunk = _make_chunk("col1 | col2\nval1 | val2", chunk_type="table")

        result = _serialize_chunk(chunk)

        assert result == "<table>\ncol1 | col2\nval1 | val2\n</table>"

    def test_dispatches_code_by_type(self) -> None:
        """Should use code serializer when chunk_type is 'code'."""
        chunk = _make_chunk("print('hello')", chunk_type="code", code_language="python")

        result = _serialize_chunk(chunk)

        assert result == "```python\nprint('hello')\n```"

    def test_dispatches_image_by_type(self) -> None:
        """Should use image serializer when chunk_type is 'image_description'."""
        chunk = _make_chunk("raw desc", chunk_type="image_description", image_caption="A diagram")

        result = _serialize_chunk(chunk)

        assert result == "[Image: A diagram]"

    def test_defaults_to_text_for_unknown_type(self) -> None:
        """Should return plain text when chunk_type is not recognized."""
        chunk = _make_chunk("some text", chunk_type="text")

        result = _serialize_chunk(chunk)

        assert result == "some text"

    def test_defaults_to_text_when_chunk_type_missing(self) -> None:
        """Should return plain text when chunk_type attribute is absent."""
        chunk = MagicMock()
        chunk.text = "fallback text"
        del chunk.meta.chunk_type

        result = _serialize_chunk(chunk)

        assert result == "fallback text"

    def test_returns_plain_text_when_format_disabled(self) -> None:
        """Should return plain text regardless of chunk_type when format is off."""
        chunk = _make_chunk("col1 | col2\nval1 | val2", chunk_type="table")

        with patch("rag.chunking.config.CHUNK_TYPE_FORMAT", False):
            result = _serialize_chunk(chunk)

        assert result == "col1 | col2\nval1 | val2"


class TestIsHybridChunkerAvailable:
    """Tests for is_hybrid_chunker_available."""

    def test_returns_false_when_get_chunker_returns_none(self) -> None:
        """Should return False when _get_hybrid_chunker returns None (import fails)."""
        with patch("rag.chunking._get_hybrid_chunker", return_value=None):
            result = is_hybrid_chunker_available()

        assert result is False

    def test_returns_true_when_get_chunker_returns_object(self) -> None:
        """Should return True when _get_hybrid_chunker returns a chunker instance."""
        with patch("rag.chunking._get_hybrid_chunker", return_value=MagicMock()):
            result = is_hybrid_chunker_available()

        assert result is True


class TestChunkDoclingDocument:
    """Tests for chunk_docling_document error paths."""

    def test_raises_when_hybrid_chunker_not_available(self) -> None:
        """Should raise RuntimeError when _get_hybrid_chunker returns None."""
        with (
            patch("rag.chunking._get_hybrid_chunker", return_value=None),
            pytest.raises(RuntimeError, match="HybridChunker not available"),
        ):
            chunk_docling_document({})

    def test_error_message_mentions_uv_sync(self) -> None:
        """Should guide user to install docling with uv sync."""
        with (
            patch("rag.chunking._get_hybrid_chunker", return_value=None),
            pytest.raises(RuntimeError) as exc_info,
        ):
            chunk_docling_document({})

        assert "uv sync" in str(exc_info.value)

    def test_raises_when_docling_core_not_importable(self) -> None:
        """Should raise RuntimeError when docling_core import fails."""
        mock_chunker = MagicMock()

        with (
            patch("rag.chunking._get_hybrid_chunker", return_value=mock_chunker),
            patch("rag.chunking.DoclingDocument", create=True),
        ):
            import builtins

            original_import = builtins.__import__

            def _fake_import(name, *args, **kwargs):
                if name == "docling_core.docling_document":
                    raise ImportError("No module named 'docling_core'")
                return original_import(name, *args, **kwargs)

            with (
                patch("builtins.__import__", side_effect=_fake_import),
                pytest.raises(RuntimeError, match="docling_core not available"),
            ):
                chunk_docling_document({})
