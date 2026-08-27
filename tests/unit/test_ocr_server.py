"""Unit tests for ocr_server — FastAPI OCR service endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from servers.ocr import ocr_server


@pytest.fixture()
def client():
    """Create a test client with the app."""
    return TestClient(ocr_server.app)


@pytest.fixture()
def loaded_state():
    """Simulate a loaded backend so /convert doesn't return 503."""
    backend = MagicMock()
    backend.provider = "cpu"
    backend.vram_mb = 0
    ocr_server._current = ocr_server.BackendState(
        backend=backend, name="pp-ocrv6-small", loaded=True
    )
    yield
    ocr_server._current = None


class TestHealth:
    def test_health_returns_ok(self, client) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "model" in body
        assert "provider" in body
        assert "loaded" in body
        assert "vram_mb" in body

    def test_health_reports_loaded_state(self, client, loaded_state) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["loaded"] is True
        assert body["provider"] == "cpu"


class TestQueue:
    def test_queue_empty_by_default(self, client) -> None:
        resp = client.get("/queue")
        assert resp.status_code == 200
        body = resp.json()
        assert body["current"] is None
        assert body["pending"] == []
        assert body["queued"] == 0
        assert body["busy"] is False

    def test_queue_reports_current_and_pending(self, client) -> None:
        ocr_server._current_file = "scan1.pdf"
        ocr_server._pending_files.extend(["scan2.pdf", "scan3.pdf"])
        try:
            resp = client.get("/queue")
            assert resp.status_code == 200
            body = resp.json()
            assert body["current"] == "scan1.pdf"
            assert body["pending"] == ["scan2.pdf", "scan3.pdf"]
            assert body["queued"] == 2
        finally:
            ocr_server._current_file = None
            ocr_server._pending_files.clear()


class TestModelSwap:
    @patch("servers.ocr.ocr_server._unload_model")
    def test_swap_to_valid_model(self, mock_unload, client) -> None:
        backend = MagicMock()
        backend.provider = "cpu"
        backend.vram_mb = 0

        def fake_load(name: str) -> None:
            ocr_server._current = ocr_server.BackendState(
                backend=backend, name=name, loaded=True
            )

        with patch("servers.ocr.ocr_server._load_model", side_effect=fake_load):
            resp = client.post("/model/swap", json={"model": "pp-ocrv6-small"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["current"] == "pp-ocrv6-small"
        assert body["provider"] == "cpu"
        mock_unload.assert_called_once()
        ocr_server._current = None

    def test_swap_to_invalid_model(self, client) -> None:
        resp = client.post("/model/swap", json={"model": "unknown-model"})
        assert resp.status_code == 400
        assert "Unknown model" in resp.json()["detail"]


class TestRapidOcrBackend:
    def test_tier_mapping(self) -> None:
        registry = ocr_server._backend_registry
        assert registry["pp-ocrv6-small"]._tier == "small"
        assert registry["pp-ocrv6-medium"]._tier == "medium"


class TestConvert:
    @patch(
        "servers.ocr.ocr_server._process_image_bytes",
        return_value={"text": "", "confidence": 0, "lines": 0},
    )
    def test_convert_single_file(self, mock_ocr, client, loaded_state) -> None:
        """Convert with a single file returns valid response."""
        resp = client.post(
            "/convert",
            files=[("files", ("test.pdf", b"fake-pdf-content", "application/pdf"))],
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["pages"] == [{"page": 1, "text": "", "confidence": 0, "lines": 0}]

    @patch(
        "servers.ocr.ocr_server._process_image_bytes",
        return_value={"text": "extracted text", "confidence": 0.95, "lines": 1},
    )
    def test_convert_with_mock_ocr(self, mock_ocr, client, loaded_state) -> None:
        """Convert with mocked OCR function."""
        resp = client.post(
            "/convert",
            files=[("files", ("test.pdf", b"fake-pdf-content", "application/pdf"))],
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["markdown"] == "extracted text"
        assert len(body["pages"]) == 1
        assert body["pages"][0]["text"] == "extracted text"
        assert body["model"] == "pp-ocrv6-small"
        assert body["provider"] == "cpu"

    def test_convert_no_model_loaded_returns_503(self, client) -> None:
        resp = client.post(
            "/convert",
            files=[("files", ("test.pdf", b"fake-pdf-content", "application/pdf"))],
        )
        assert resp.status_code == 503
