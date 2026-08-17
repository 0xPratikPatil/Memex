"""Unit tests for marker_client — job-based Marker conversion client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from memex.engine.core.errors import ConversionError, CorruptedDocumentError
from memex.engine.ingestion.marker_client import convert_markdown, is_marker_available


@pytest.fixture(autouse=True)
def _mock_gpu_lock() -> MagicMock:
    """Neutralize GpuLock in client tests (exercised in test_gpu_lock.py)."""
    lock = MagicMock()
    with patch("memex.engine.ingestion.marker_client.gpu_lock", lock):
        yield lock


def _success_body(markdown: str = "# Converted") -> dict:
    return {
        "format": "markdown",
        "output": markdown,
        "images": {},
        "metadata": {"page_count": 1},
        "success": True,
    }


class _FakeResponse:
    """Minimal httpx.Response-like for status/result mocking."""

    def __init__(self, json_body: dict, status_code: int = 200) -> None:
        self._body = json_body
        self.status_code = status_code

    def json(self) -> dict:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{self.status_code}",
                request=httpx.Request("GET", "http://x"),
                response=httpx.Response(self.status_code, request=httpx.Request("GET", "http://x")),
            )


class TestConvertMarkdown:
    def test_submit_and_poll_success(self) -> None:
        """Full job flow: submit → poll running → done → result."""
        calls = {"n": 0}
        captured: dict = {}

        def fake_post(url, files, data):
            calls["n"] += 1
            captured["data"] = data
            return _FakeResponse({"job_id": "abc", "status": "pending"})

        def fake_get(url):
            if url.endswith("/result"):
                return _FakeResponse(_success_body())
            return _FakeResponse({"job_id": "abc", "status": "done"})

        fake_client = MagicMock()
        fake_client.post.side_effect = fake_post
        fake_client.get.side_effect = fake_get

        with (
            patch("memex.engine.ingestion.marker_client._get_client", return_value=fake_client),
            patch("memex.engine.ingestion.marker_client.config.MARKER_MODE", "fast"),
            patch("memex.engine.ingestion.marker_client.config.MARKER_FORCE_OCR", False),
            patch("memex.engine.ingestion.marker_client.JOB_POLL_INTERVAL", 0),
        ):
            result = convert_markdown(b"pdf-bytes", "doc.pdf")
            assert result.ok
            assert result.markdown == "# Converted"
            assert calls["n"] == 1
            # Submit payload carried the mode
            data = captured["data"]
            assert data["mode"] == "fast"
            assert data["force_ocr"] == "false"

    def test_acquires_gpu_lock(self, _mock_gpu_lock: MagicMock) -> None:
        """convert_markdown acquires+releases the GpuLock around the job."""

        def fake_post(url, files, data):
            return _FakeResponse({"job_id": "abc", "status": "pending"})

        def fake_get(url):
            if url.endswith("/result"):
                return _FakeResponse(_success_body())
            return _FakeResponse({"job_id": "abc", "status": "done"})

        fake_client = MagicMock()
        fake_client.post.side_effect = fake_post
        fake_client.get.side_effect = fake_get

        with (
            patch("memex.engine.ingestion.marker_client._get_client", return_value=fake_client),
            patch("memex.engine.ingestion.marker_client.config.MARKER_MODE", "fast"),
            patch("memex.engine.ingestion.marker_client.config.MARKER_FORCE_OCR", False),
            patch("memex.engine.ingestion.marker_client.JOB_POLL_INTERVAL", 0),
        ):
            convert_markdown(b"pdf-bytes", "doc.pdf")
            _mock_gpu_lock.acquire.assert_called_once_with("marker")
            _mock_gpu_lock.release.assert_called_once_with("marker")

    def test_success_false_raises_conversion_error(self) -> None:
        def fake_post(url, files, data):
            return _FakeResponse({"job_id": "abc", "status": "pending"})

        def fake_get(url):
            if url.endswith("/result"):
                return _FakeResponse({"success": False, "error": "could not convert"})
            return _FakeResponse({"job_id": "abc", "status": "failed"})

        fake_client = MagicMock()
        fake_client.post.side_effect = fake_post
        fake_client.get.side_effect = fake_get

        with (
            patch("memex.engine.ingestion.marker_client._get_client", return_value=fake_client),
            patch("memex.engine.ingestion.marker_client.config.MARKER_MODE", "fast"),
            patch("memex.engine.ingestion.marker_client.config.MARKER_FORCE_OCR", False),
            patch("memex.engine.ingestion.marker_client.JOB_POLL_INTERVAL", 0),
            pytest.raises(ConversionError, match="could not convert"),
        ):
            convert_markdown(b"pdf-bytes", "doc.pdf")

    def test_empty_output_raises_corrupted_document(self) -> None:
        def fake_post(url, files, data):
            return _FakeResponse({"job_id": "abc", "status": "pending"})

        def fake_get(url):
            if url.endswith("/result"):
                return _FakeResponse(_success_body(markdown="   "))
            return _FakeResponse({"job_id": "abc", "status": "done"})

        fake_client = MagicMock()
        fake_client.post.side_effect = fake_post
        fake_client.get.side_effect = fake_get

        with (
            patch("memex.engine.ingestion.marker_client._get_client", return_value=fake_client),
            patch("memex.engine.ingestion.marker_client.config.MARKER_MODE", "fast"),
            patch("memex.engine.ingestion.marker_client.config.MARKER_FORCE_OCR", False),
            patch("memex.engine.ingestion.marker_client.JOB_POLL_INTERVAL", 0),
            pytest.raises(CorruptedDocumentError),
        ):
            convert_markdown(b"pdf-bytes", "doc.pdf")

    def test_transport_error_raises_service_unavailable(self) -> None:
        with patch(
            "memex.engine.ingestion.marker_client._post_transport",
            side_effect=httpx.ConnectError("connection refused"),
        ):
            from memex.engine.core.errors import ServiceUnavailableError

            with pytest.raises(ServiceUnavailableError):
                convert_markdown(b"pdf-bytes", "doc.pdf")


class TestPollResilience:
    """Polling must survive transient poll failures (server restart mid-job)."""

    def test_poll_survives_interrupted_polls(self) -> None:
        attempts = {"n": 0}

        def fake_post(url, files, data):
            return _FakeResponse({"job_id": "abc", "status": "pending"})

        def fake_get(url):
            if url.endswith("/result"):
                return _FakeResponse(_success_body())
            attempts["n"] += 1
            if attempts["n"] <= 2:
                raise httpx.ConnectError("server restarting")
            return _FakeResponse({"job_id": "abc", "status": "done"})

        fake_client = MagicMock()
        fake_client.post.side_effect = fake_post
        fake_client.get.side_effect = fake_get

        with (
            patch("memex.engine.ingestion.marker_client._get_client", return_value=fake_client),
            patch("memex.engine.ingestion.marker_client.config.MARKER_MODE", "fast"),
            patch("memex.engine.ingestion.marker_client.config.MARKER_FORCE_OCR", False),
            patch("memex.engine.ingestion.marker_client.JOB_POLL_INTERVAL", 0),
        ):
            result = convert_markdown(b"pdf-bytes", "doc.pdf")
            assert result.ok
            assert attempts["n"] == 3  # 2 failed polls + 1 success


class TestIsMarkerAvailable:
    def test_returns_true_when_healthy(self) -> None:
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.get.return_value = mock_resp

        with (
            patch("memex.engine.ingestion.marker_client._get_client", return_value=mock_client),
            patch("memex.engine.ingestion.marker_client.config.MARKER_URL", "http://localhost:5001"),
        ):
            assert is_marker_available() is True
            mock_client.get.assert_called_once_with("http://localhost:5001/health", timeout=5.0)

    def test_returns_false_on_error(self) -> None:
        with (
            patch("memex.engine.ingestion.marker_client._get_client", side_effect=Exception("down")),
            patch("memex.engine.ingestion.marker_client.config.MARKER_URL", "http://localhost:5001"),
        ):
            assert is_marker_available() is False
