"""Unit tests for ocr_server — FastAPI OCR service endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    """Create a test client with the app."""
    from ocr_server import app

    return TestClient(app)


class TestHealth:
    def test_health_returns_ok(self, client) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "model" in body
        assert "loaded_models" in body


class TestModelSwap:
    def test_swap_to_valid_model(self, client) -> None:
        resp = client.post("/model/swap", json={"model": "pp-ocrv6-small"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["current"] == "pp-ocrv6-small"

    def test_swap_to_invalid_model(self, client) -> None:
        resp = client.post("/model/swap", json={"model": "unknown-model"})
        assert resp.status_code == 400
        assert "Unknown model" in resp.json()["detail"]


class TestConvert:
    @patch("ocr_server._models", {"pp-ocrv6-small": MagicMock()})
    @patch("ocr_server._ocr_pp_ocrv6")
    def test_convert_single_file(self, mock_ocr, client) -> None:
        """Convert with a single file returns valid response."""
        mock_ocr.return_value = {"text": "", "confidence": 0, "lines": 0}
        resp = client.post(
            "/convert",
            files=[("files", ("test.pdf", b"fake-pdf-content", "application/pdf"))],
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["pages"] == [{"page": 1, "text": "", "confidence": 0, "lines": 0}]

    @patch("ocr_server._models", {"pp-ocrv6-small": MagicMock()})
    @patch("ocr_server._ocr_pp_ocrv6")
    def test_convert_with_mock_ocr(self, mock_ocr, client) -> None:
        """Convert with mocked OCR function."""
        mock_ocr.return_value = {
            "text": "extracted text",
            "confidence": 0.95,
            "lines": 1,
        }
        # Create a fake PDF file
        resp = client.post(
            "/convert",
            files=[("files", ("test.pdf", b"fake-pdf-content", "application/pdf"))],
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["markdown"] == "extracted text"
        assert len(body["pages"]) == 1
        assert body["pages"][0]["text"] == "extracted text"
