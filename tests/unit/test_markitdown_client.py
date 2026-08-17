"""Unit tests for MarkItDown client — HTTP mocking, error handling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from memex.engine.core.errors import (
    ConversionError,
    CorruptedDocumentError,
    ServiceUnavailableError,
)
from memex.engine.ingestion.markitdown_client import (
    MarkItDownResult,
    convert_markdown,
    is_markitdown_available,
)


class TestMarkItDownResult:
    """Test the MarkItDownResult dataclass."""

    def test_ok_with_content(self):
        result = MarkItDownResult(markdown="# Hello\n\nWorld", format="pdf")
        assert result.ok is True

    def test_not_ok_with_empty(self):
        result = MarkItDownResult(markdown="", format="pdf")
        assert result.ok is False

    def test_not_ok_with_whitespace(self):
        result = MarkItDownResult(markdown="   \n  ", format="pdf")
        assert result.ok is False


class TestConvertMarkdown:
    """Test the convert_markdown function with mocked HTTP."""

    @patch("memex.engine.ingestion.markitdown_client._get_client")
    def test_successful_conversion(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "success": True,
            "output": "# Document\n\nThis is content.",
            "format": "pdf",
            "processing_time": 0.5,
            "metadata": {"author": "Test"},
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        result = convert_markdown(b"fake pdf bytes", "test.pdf")

        assert result.ok is True
        assert result.markdown == "# Document\n\nThis is content."
        assert result.format == "pdf"
        assert result.processing_time == 0.5
        assert result.metadata == {"author": "Test"}

    @patch("memex.engine.ingestion.markitdown_client._get_client")
    def test_empty_output_raises_error(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "success": True,
            "output": "",
            "format": "pdf",
            "processing_time": 0.1,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        with pytest.raises(CorruptedDocumentError):
            convert_markdown(b"empty", "empty.pdf")

    @patch("memex.engine.ingestion.markitdown_client._get_client")
    def test_server_failure_raises_error(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "success": False,
            "error": "unsupported format",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp

        with pytest.raises(ConversionError):
            convert_markdown(b"bad", "bad.xyz")

    @patch("memex.engine.ingestion.markitdown_client._get_client")
    def test_http_500_raises_error(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError("500", request=MagicMock(), response=mock_resp)
        mock_client.post.return_value = mock_resp

        with pytest.raises(ConversionError):
            convert_markdown(b"fail", "fail.pdf")

    @patch("memex.engine.ingestion.markitdown_client._get_client")
    def test_connection_error_raises_unavailable(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.post.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(ServiceUnavailableError):
            convert_markdown(b"test", "test.pdf")


class TestIsMarkitdownAvailable:
    """Test the health check function."""

    @patch("memex.engine.ingestion.markitdown_client._get_client")
    def test_available(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.get.return_value = mock_resp

        assert is_markitdown_available() is True

    @patch("memex.engine.ingestion.markitdown_client._get_client")
    def test_unavailable(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")

        assert is_markitdown_available() is False
