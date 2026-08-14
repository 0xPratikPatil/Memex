"""Unit tests for DoclingAsyncClient — async Docling Serve client with status polling."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from memex.engine.ingestion.docling_client import DoclingAsyncClient


@pytest.fixture
def client() -> DoclingAsyncClient:
    """Create a DoclingAsyncClient with mock base URL."""
    return DoclingAsyncClient(base_url="http://localhost:5001", api_key="test-key")


@pytest.fixture
def client_no_key() -> DoclingAsyncClient:
    """Create a DoclingAsyncClient without API key."""
    return DoclingAsyncClient(base_url="http://localhost:5001")


# ── health_check tests ───────────────────────────────────────────────────────


class TestHealthCheck:
    """health_check should verify Docling is reachable."""

    @pytest.mark.asyncio
    async def test_returns_true_when_200(self, client: DoclingAsyncClient) -> None:
        """Should return True when /health returns 200."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(client._client, "get", new_callable=AsyncMock, return_value=mock_response):
            result = await client.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_not_200(self, client: DoclingAsyncClient) -> None:
        """Should return False when /health returns non-200."""
        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch.object(client._client, "get", new_callable=AsyncMock, return_value=mock_response):
            result = await client.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_connection_error(self, client: DoclingAsyncClient) -> None:
        """Should return False when connection fails."""
        with patch.object(client._client, "get", new_callable=AsyncMock, side_effect=httpx.ConnectError("refused")):
            result = await client.health_check()

        assert result is False


# ── ready_check tests ────────────────────────────────────────────────────────


class TestReadyCheck:
    """ready_check should verify Docling models are loaded."""

    @pytest.mark.asyncio
    async def test_returns_true_when_200(self, client: DoclingAsyncClient) -> None:
        """Should return True when /ready returns 200."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(client._client, "get", new_callable=AsyncMock, return_value=mock_response):
            result = await client.ready_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_503(self, client: DoclingAsyncClient) -> None:
        """Should return False when /ready returns 503 (models not loaded)."""
        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch.object(client._client, "get", new_callable=AsyncMock, return_value=mock_response):
            result = await client.ready_check()

        assert result is False


# ── submit_conversion tests ──────────────────────────────────────────────────


class TestSubmitConversion:
    """submit_conversion should POST to async endpoint and return task_id."""

    @pytest.mark.asyncio
    async def test_returns_task_id(self, client: DoclingAsyncClient) -> None:
        """Should extract task_id from response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "task_id": "abc-123-def",
            "task_status": "pending",
            "task_position": 1,
        }
        mock_response.raise_for_status = MagicMock()

        payload = {
            "options": {"to_formats": ["md"]},
            "sources": [{"kind": "http", "url": "https://example.com/doc.pdf"}],
        }

        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response):
            task_id = await client.submit_conversion(payload)

        assert task_id == "abc-123-def"

    @pytest.mark.asyncio
    async def test_sends_api_key_header(self, client: DoclingAsyncClient) -> None:
        """Should include X-Api-Key header when api_key is set."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"task_id": "xyz", "task_status": "pending", "task_position": 1}
        mock_response.raise_for_status = MagicMock()

        payload = {"options": {}, "sources": []}

        with patch.object(client._client, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            await client.submit_conversion(payload)

            call_kwargs = mock_post.call_args
            assert call_kwargs[1]["headers"]["X-Api-Key"] == "test-key"

    @pytest.mark.asyncio
    async def test_no_api_key_header_when_not_set(self, client_no_key: DoclingAsyncClient) -> None:
        """Should not include X-Api-Key header when api_key is None."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"task_id": "xyz", "task_status": "pending", "task_position": 1}
        mock_response.raise_for_status = MagicMock()

        payload = {"options": {}, "sources": []}

        with patch.object(
            client_no_key._client, "post", new_callable=AsyncMock, return_value=mock_response
        ) as mock_post:
            await client_no_key.submit_conversion(payload)

            call_kwargs = mock_post.call_args
            assert "X-Api-Key" not in call_kwargs[1]["headers"]


# ── poll_status tests ────────────────────────────────────────────────────────


class TestPollStatus:
    """poll_status should GET task status from Docling."""

    @pytest.mark.asyncio
    async def test_returns_status_dict(self, client: DoclingAsyncClient) -> None:
        """Should return task status dict with task_status and task_position."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "task_status": "started",
            "task_position": 3,
        }

        with patch.object(client._client, "get", new_callable=AsyncMock, return_value=mock_response):
            result = await client.poll_status("task-abc")

        assert result["task_status"] == "started"
        assert result["task_position"] == 3

    @pytest.mark.asyncio
    async def test_calls_correct_url(self, client: DoclingAsyncClient) -> None:
        """Should poll /v1/status/poll/{task_id}."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"task_status": "success", "task_position": 0}

        with patch.object(client._client, "get", new_callable=AsyncMock, return_value=mock_response) as mock_get:
            await client.poll_status("task-xyz")

            called_url = mock_get.call_args[0][0]
            assert called_url == "http://localhost:5001/v1/status/poll/task-xyz"


# ── get_result tests ─────────────────────────────────────────────────────────


class TestGetResult:
    """get_result should fetch completed conversion result."""

    @pytest.mark.asyncio
    async def test_returns_result_dict(self, client: DoclingAsyncClient) -> None:
        """Should return full conversion result from /v1/result/{task_id}."""
        expected = {
            "document": {"md_content": "# Converted"},
            "status": "success",
            "processing_time": 5.0,
            "errors": [],
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected

        with patch.object(client._client, "get", new_callable=AsyncMock, return_value=mock_response):
            result = await client.get_result("task-abc")

        assert result == expected


# ── wait_for_completion tests ────────────────────────────────────────────────


class TestWaitForCompletion:
    """wait_for_completion should poll until success/failure or timeout."""

    @pytest.mark.asyncio
    async def test_returns_when_success(self, client: DoclingAsyncClient) -> None:
        """Should return immediately when status is success."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"task_status": "success", "task_position": 0}

        with patch.object(client._client, "get", new_callable=AsyncMock, return_value=mock_response):
            result = await client.wait_for_completion("task-abc", poll_interval=0.01)

        assert result["task_status"] == "success"

    @pytest.mark.asyncio
    async def test_returns_when_failure(self, client: DoclingAsyncClient) -> None:
        """Should return immediately when status is failure."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"task_status": "failure", "task_position": 0}

        with patch.object(client._client, "get", new_callable=AsyncMock, return_value=mock_response):
            result = await client.wait_for_completion("task-abc", poll_interval=0.01)

        assert result["task_status"] == "failure"

    @pytest.mark.asyncio
    async def test_polls_until_success(self, client: DoclingAsyncClient) -> None:
        """Should poll multiple times before getting success."""
        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            if call_count < 3:
                mock_resp.json.return_value = {"task_status": "started", "task_position": 1}
            else:
                mock_resp.json.return_value = {"task_status": "success", "task_position": 0}
            return mock_resp

        with patch.object(client._client, "get", side_effect=mock_get):
            result = await client.wait_for_completion("task-abc", poll_interval=0.01)

        assert result["task_status"] == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_timeout_when_exceeded(self, client: DoclingAsyncClient) -> None:
        """Should raise TimeoutError when max_wait is exceeded."""
        async def mock_get(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"task_status": "started", "task_position": 1}
            return mock_resp

        with patch.object(client._client, "get", side_effect=mock_get), pytest.raises(
            TimeoutError, match="exceeded"
        ):
            await client.wait_for_completion("task-abc", poll_interval=0.05, max_wait=0.1)


# ── close tests ──────────────────────────────────────────────────────────────


class TestClose:
    """close should properly close the HTTP client."""

    @pytest.mark.asyncio
    async def test_closes_http_client(self, client: DoclingAsyncClient) -> None:
        """Should call aclose on the httpx client."""
        with patch.object(client._client, "aclose", new_callable=AsyncMock) as mock_close:
            await client.close()
            mock_close.assert_called_once()
