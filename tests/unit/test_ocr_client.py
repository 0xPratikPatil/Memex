"""Unit tests for ocr_client — OCR fallback client for scanned PDFs."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from memex.engine.core.errors import ConversionError, ServiceUnavailableError
from memex.engine.ingestion.ocr_client import OcrResult, convert_with_ocr, is_ocr_available


class _FakeResponse:
    """Minimal httpx.Response-like for mocking."""

    def __init__(self, json_body: dict, status_code: int = 200) -> None:
        self._body = json_body
        self.status_code = status_code

    def json(self) -> dict:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{self.status_code}",
                request=httpx.Request("POST", "http://x"),
                response=httpx.Response(self.status_code, request=httpx.Request("POST", "http://x")),
            )


class TestOcrResult:
    def test_ok_when_success(self) -> None:
        r = OcrResult(markdown="text", status="success")
        assert r.ok is True

    def test_not_ok_when_error(self) -> None:
        r = OcrResult(markdown="", status="error")
        assert r.ok is False

    def test_not_ok_when_empty_markdown(self) -> None:
        # Success status but no extracted text = not usable
        r = OcrResult(markdown="  ", status="success")
        assert r.ok is False


class TestIsOcrAvailable:
    def test_returns_true_when_healthy(self) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok", "loaded": True, "model": "pp-ocrv6-small"}
        mock_client.get.return_value = mock_resp

        with (
            patch("memex.engine.ingestion.ocr_client._get_client", return_value=mock_client),
            patch("memex.engine.ingestion.ocr_client.config.OCR_URL", "http://localhost:5004"),
        ):
            assert is_ocr_available() is True

    def test_returns_false_when_not_loaded(self) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok", "loaded": False, "model": "pp-ocrv6-small"}
        mock_client.get.return_value = mock_resp

        with (
            patch("memex.engine.ingestion.ocr_client._get_client", return_value=mock_client),
            patch("memex.engine.ingestion.ocr_client.config.OCR_URL", "http://localhost:5004"),
        ):
            assert is_ocr_available() is False

    def test_returns_false_on_error(self) -> None:
        with (
            patch("memex.engine.ingestion.ocr_client._get_client", side_effect=Exception("down")),
            patch("memex.engine.ingestion.ocr_client.config.OCR_URL", "http://localhost:5004"),
        ):
            assert is_ocr_available() is False


class TestConvertWithOcr:
    def test_success(self) -> None:
        fake_client = MagicMock()
        fake_client.post.return_value = _FakeResponse(
            {
                "markdown": "extracted text",
                "pages": [{"page": 1, "text": "extracted text", "confidence": 0.95}],
                "model": "pp-ocrv6-small",
                "processing_time": 1.2,
            }
        )

        with (
            patch("memex.engine.ingestion.ocr_client._get_client", return_value=fake_client),
            patch("memex.engine.ingestion.ocr_client.config.OCR_URL", "http://localhost:5004"),
            patch("memex.engine.ingestion.ocr_client.config.OCR_TIMEOUT", 120.0),
        ):
            result = convert_with_ocr(b"pdf-bytes", "scanned.pdf")
            assert result.ok
            assert result.markdown == "extracted text"
            assert result.model == "pp-ocrv6-small"
            assert result.processing_time == 1.2

    def test_http_error_raises_conversion_error(self) -> None:
        fake_client = MagicMock()
        resp = httpx.Response(
            500,
            text="internal error",
            request=httpx.Request("POST", "http://x"),
        )
        fake_client.post.return_value = resp

        with (
            patch("memex.engine.ingestion.ocr_client._get_client", return_value=fake_client),
            patch("memex.engine.ingestion.ocr_client.config.OCR_URL", "http://localhost:5004"),
            patch("memex.engine.ingestion.ocr_client.config.OCR_TIMEOUT", 120.0),
            pytest.raises(ConversionError, match="OCR service error"),
        ):
            convert_with_ocr(b"pdf-bytes", "scanned.pdf")

    def test_transport_error_raises_service_unavailable(self) -> None:
        fake_client = MagicMock()
        fake_client.post.side_effect = httpx.ConnectError("connection refused")

        with (
            patch("memex.engine.ingestion.ocr_client._get_client", return_value=fake_client),
            patch("memex.engine.ingestion.ocr_client.config.OCR_URL", "http://localhost:5004"),
            patch("memex.engine.ingestion.ocr_client.config.OCR_TIMEOUT", 120.0),
            pytest.raises(ServiceUnavailableError, match="OCR"),
        ):
            convert_with_ocr(b"pdf-bytes", "scanned.pdf")
