"""Unit tests for MarkItDown server — endpoint testing with mocked conversion."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from servers.markitdown import markitdown_server


@pytest.fixture
def client():
    """Create a test client with mocked MarkItDown."""
    with patch("servers.markitdown.markitdown_server._get_markitdown") as mock_get:
        mock_md = MagicMock()
        mock_result = MagicMock()
        mock_result.text_content = "# Hello\n\nWorld content."
        mock_result.metadata = {"author": "Test Author"}
        mock_md.convert.return_value = mock_result
        mock_get.return_value = mock_md

        with TestClient(markitdown_server.app) as c:
            yield c, mock_md


class TestHealth:
    def test_health_endpoint(self, client):
        c, _ = client
        resp = c.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestQueue:
    def test_queue_empty_by_default(self, client):
        c, _ = client
        resp = c.get("/queue")
        assert resp.status_code == 200
        body = resp.json()
        assert body["current"] is None
        assert body["pending"] == []
        assert body["queued"] == 0
        assert body["busy"] is False
        assert body["max_concurrent"] > 0

    def test_queue_reports_current_and_pending(self, client):
        import servers.markitdown.markitdown_server as markitdown_server

        c, _ = client
        markitdown_server._active_files.extend(["converting.docx"])
        markitdown_server._pending_files.extend(["wait1.pdf", "wait2.pdf"])
        try:
            resp = c.get("/queue")
            assert resp.status_code == 200
            body = resp.json()
            assert body["current"] == "converting.docx"
            assert body["active"] == ["converting.docx"]
            assert body["pending"] == ["wait1.pdf", "wait2.pdf"]
            assert body["queued"] == 2
            assert body["busy"] is True
        finally:
            markitdown_server._active_files.clear()
            markitdown_server._pending_files.clear()

    def test_queue_reports_all_concurrent_files(self, client):
        import servers.markitdown.markitdown_server as markitdown_server

        c, _ = client
        markitdown_server._active_files.extend(["one.pdf", "two.pdf"])
        try:
            resp = c.get("/queue")
            body = resp.json()
            assert body["current"] == "one.pdf"
            assert body["active"] == ["one.pdf", "two.pdf"]
        finally:
            markitdown_server._active_files.clear()


class TestConvert:
    def test_successful_conversion(self, client):
        c, _mock_md = client
        resp = c.post(
            "/convert",
            files={"file": ("test.pdf", b"fake pdf content", "application/octet-stream")},
            data={"filename": "test.pdf"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["output"] == "# Hello\n\nWorld content."
        assert body["format"] == "pdf"
        assert body["metadata"] == {}
        assert "processing_time" in body

    def test_empty_file_returns_error(self, client):
        c, _ = client
        resp = c.post(
            "/convert",
            files={"file": ("empty.pdf", b"", "application/octet-stream")},
            data={"filename": "empty.pdf"},
        )
        assert resp.status_code == 400

    def test_conversion_exception_returns_500(self, client):
        c, mock_md = client
        mock_md.convert.side_effect = RuntimeError("conversion failed")
        resp = c.post(
            "/convert",
            files={"file": ("bad.pdf", b"content", "application/octet-stream")},
            data={"filename": "bad.pdf"},
        )
        assert resp.status_code == 500
        body = resp.json()
        assert body["success"] is False
        assert "conversion failed" in body["error"]

    def test_format_detection(self, client):
        c, _mock_md = client
        resp = c.post(
            "/convert",
            files={"file": ("report.docx", b"content", "application/octet-stream")},
            data={"filename": "report.docx"},
        )
        assert resp.status_code == 200
        assert resp.json()["format"] == "docx"

    def test_unknown_format(self, client):
        c, _mock_md = client
        resp = c.post(
            "/convert",
            files={"file": ("document", b"content", "application/octet-stream")},
            data={"filename": "document"},
        )
        assert resp.status_code == 200
        assert resp.json()["format"] == "unknown"
