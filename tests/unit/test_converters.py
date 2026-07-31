"""Unit tests for converter abstraction and implementations."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag.converters import Converter, get_converter

# ── get_converter factory ────────────────────────────────────────────────────


class TestGetConverter:
    def test_returns_docling_by_default(self):
        cfg = MagicMock()
        cfg.get_str.return_value = "docling"
        converter = get_converter(cfg)
        from rag.converters.docling import DoclingConverter

        assert isinstance(converter, DoclingConverter)

    def test_returns_markitdown_when_configured(self):
        cfg = MagicMock()
        cfg.get_str.side_effect = lambda key, default="": {
            "converter.engine": "markitdown",
            "converter.markitdown_url": "http://localhost:5003",
        }.get(key, default)
        converter = get_converter(cfg)
        from rag.converters.markitdown import MarkItDownConverter

        assert isinstance(converter, MarkItDownConverter)

    def test_factory_returns_converter_subclass(self):
        cfg = MagicMock()
        cfg.get_str.return_value = "docling"
        converter = get_converter(cfg)
        assert isinstance(converter, Converter)


# ── DoclingConverter ──────────────────────────────────────────────────────────


class TestDoclingConverter:
    @pytest.mark.asyncio
    async def test_convert_delegates_to_parse_file(self, tmp_path: Path):
        from rag.converters.docling import DoclingConverter

        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"PDF content")

        mock_result = MagicMock()
        mock_result.ok = True
        mock_result.markdown = "# Converted"

        with patch("rag.docling_client.parse_file", return_value=mock_result):
            converter = DoclingConverter("http://localhost:5001")
            result = await converter.convert(str(test_file))
            assert result == "# Converted"

    @pytest.mark.asyncio
    async def test_convert_raises_on_failure(self, tmp_path: Path):
        from rag.converters.docling import DoclingConverter

        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"PDF content")

        mock_result = MagicMock()
        mock_result.ok = False
        mock_result.errors = ["conversion failed"]

        with patch("rag.docling_client.parse_file", return_value=mock_result):
            converter = DoclingConverter("http://localhost:5001")
            with pytest.raises(RuntimeError, match="Docling conversion failed"):
                await converter.convert(str(test_file))

    @pytest.mark.asyncio
    async def test_health_check_returns_true_when_available(self):
        from rag.converters.docling import DoclingConverter

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client_cls.return_value = mock_client

            converter = DoclingConverter("http://localhost:5001")
            assert await converter.health_check()

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_error(self):
        from rag.converters.docling import DoclingConverter

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Connection refused")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client_cls.return_value = mock_client

            converter = DoclingConverter("http://localhost:5001")
            assert not await converter.health_check()


# ── MarkItDownConverter ───────────────────────────────────────────────────────


class TestMarkItDownConverter:
    @pytest.mark.asyncio
    async def test_convert_uploads_file_and_returns_markdown(self, tmp_path: Path):
        from rag.converters.markitdown import MarkItDownConverter

        test_file = tmp_path / "report.pdf"
        test_file.write_bytes(b"PDF content")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"markdown": "# Report content"}

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client_cls.return_value = mock_client

            converter = MarkItDownConverter("http://localhost:5003")
            result = await converter.convert(str(test_file))

            assert result == "# Report content"
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "http://localhost:5003/convert"

    @pytest.mark.asyncio
    async def test_convert_raises_on_missing_file(self):
        from rag.converters.markitdown import MarkItDownConverter

        converter = MarkItDownConverter("http://localhost:5003")
        with pytest.raises(FileNotFoundError, match="File not found"):
            await converter.convert("/nonexistent/file.pdf")

    @pytest.mark.asyncio
    async def test_convert_raises_on_http_error(self, tmp_path: Path):
        from rag.converters.markitdown import MarkItDownConverter

        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"PDF content")

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("500 Internal Server Error")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client_cls.return_value = mock_client

            converter = MarkItDownConverter("http://localhost:5003")
            with pytest.raises(Exception, match="500 Internal Server Error"):
                await converter.convert(str(test_file))

    @pytest.mark.asyncio
    async def test_health_check_returns_true_when_available(self):
        from rag.converters.markitdown import MarkItDownConverter

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client_cls.return_value = mock_client

            converter = MarkItDownConverter("http://localhost:5003")
            assert await converter.health_check()

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_error(self):
        from rag.converters.markitdown import MarkItDownConverter

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Connection refused")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = False
            mock_client_cls.return_value = mock_client

            converter = MarkItDownConverter("http://localhost:5003")
            assert not await converter.health_check()

    def test_strips_trailing_slash_from_url(self):
        from rag.converters.markitdown import MarkItDownConverter

        converter = MarkItDownConverter("http://localhost:5003/")
        assert converter._base_url == "http://localhost:5003"
