"""Unit tests for RetryQueue — exponential backoff retry for failed ingestion."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memex.engine.core.errors import (
    ConfigError,
    ConversionError,
    ConversionTimeoutError,
    ServiceUnavailableError,
)
from memex.engine.ingestion.status import IngestionStatus
from memex.engine.sources.retry_queue import RetryQueue


@pytest.fixture
def mock_store() -> MagicMock:
    """Create a mock FileStatusStore."""
    return MagicMock()


@pytest.fixture
def queue(mock_store: MagicMock) -> RetryQueue:
    """Create a RetryQueue with a mock store and no ingest_fn."""
    return RetryQueue(mock_store)


# ── should_retry tests ───────────────────────────────────────────────────────


class TestShouldRetry:
    """should_retry should determine if an error is retryable."""

    def test_retries_timeout_error(self, queue: RetryQueue) -> None:
        assert queue.should_retry("504 Gateway Timeout", retry_count=0) is True

    def test_retries_503_error(self, queue: RetryQueue) -> None:
        assert queue.should_retry("503 Service Unavailable", retry_count=0) is True

    def test_retries_504_error(self, queue: RetryQueue) -> None:
        assert queue.should_retry("504 Gateway Timeout", retry_count=0) is True

    def test_retries_connection_error(self, queue: RetryQueue) -> None:
        assert queue.should_retry("Connection refused", retry_count=0) is True

    def test_does_not_retry_unknown_error(self, queue: RetryQueue) -> None:
        assert queue.should_retry("Invalid format", retry_count=0) is False

    def test_does_not_retry_after_max_attempts(self, queue: RetryQueue) -> None:
        assert queue.should_retry("504 Gateway Timeout", retry_count=4) is False

    def test_retries_up_to_max(self, queue: RetryQueue) -> None:
        assert queue.should_retry("504 timeout", retry_count=3) is True
        assert queue.should_retry("504 timeout", retry_count=4) is False

    # ── Typed error matching ──────────────────────────────────────────────

    def test_retries_conversion_timeout_error(self, queue: RetryQueue) -> None:
        exc = ConversionTimeoutError("/tmp/a.pdf", timeout_s=120)
        assert queue.should_retry(str(exc), retry_count=0, exc=exc) is True

    def test_retries_service_unavailable(self, queue: RetryQueue) -> None:
        exc = ServiceUnavailableError("Docling", "connection refused")
        assert queue.should_retry(str(exc), retry_count=0, exc=exc) is True

    def test_does_not_retry_config_error(self, queue: RetryQueue) -> None:
        exc = ConfigError("bad config")
        assert queue.should_retry(str(exc), retry_count=0, exc=exc) is False

    def test_does_not_retry_generic_conversion_error(self, queue: RetryQueue) -> None:
        # ConversionTimeoutError is retryable but generic ConversionError is not.
        exc = ConversionError("/tmp/a.pdf", "corrupt pdf")
        assert queue.should_retry(str(exc), retry_count=0, exc=exc) is False


# ── schedule_retry tests ─────────────────────────────────────────────────────


class TestScheduleRetry:
    """schedule_retry should schedule retry with exponential backoff."""

    def test_schedules_retry(self, queue: RetryQueue, mock_store: MagicMock) -> None:
        queue.schedule_retry("/docs/report.pdf", "504 timeout", retry_count=0)

        mock_store.schedule_retry.assert_called_once()
        call_kwargs = mock_store.schedule_retry.call_args[1]
        assert call_kwargs["attempts"] == 1
        assert call_kwargs["backoff_s"] == 60

    def test_marks_failed_after_max_retries(self, queue: RetryQueue, mock_store: MagicMock) -> None:
        queue.schedule_retry("/docs/report.pdf", "504 timeout", retry_count=4)

        mock_store.mark_failed.assert_called_once()

    def test_backoff_increases(self, queue: RetryQueue, mock_store: MagicMock) -> None:
        queue.schedule_retry("/docs/report.pdf", "504 timeout", retry_count=0)
        first = mock_store.schedule_retry.call_args[1]["backoff_s"]
        queue.schedule_retry("/docs/report.pdf", "504 timeout", retry_count=1)
        second = mock_store.schedule_retry.call_args[1]["backoff_s"]
        assert second > first


# ── process_retries tests ────────────────────────────────────────────────────


class TestProcessRetries:
    """process_retries should re-ingest files due for retry."""

    @pytest.mark.asyncio
    async def test_returns_zero_when_empty(self, queue: RetryQueue, mock_store: MagicMock) -> None:
        mock_store.get_due_retries.return_value = []
        assert await queue.process_retries() == 0

    @pytest.mark.asyncio
    async def test_returns_zero_without_ingest_fn(self, queue: RetryQueue, mock_store: MagicMock) -> None:
        mock_store.get_due_retries.return_value = ["/docs/report.pdf"]
        assert await queue.process_retries() == 0

    @pytest.mark.asyncio
    async def test_ingests_due_files(self, mock_store: MagicMock) -> None:
        mock_store.get_due_retries.return_value = ["/docs/report.pdf"]
        queue = RetryQueue(mock_store, ingest_fn=AsyncMock(return_value=None))
        count = await queue.process_retries()
        assert count == 1
        mock_store.reset_for_retry.assert_called_once_with("/docs/report.pdf")

    @pytest.mark.asyncio
    async def test_reschedules_on_failure(self, mock_store: MagicMock) -> None:
        mock_store.get_due_retries.return_value = ["/docs/report.pdf"]
        mock_store.get_status.return_value = {"attempts": 1}
        queue = RetryQueue(mock_store, ingest_fn=AsyncMock(side_effect=ServiceUnavailableError("Docling", "down")))
        count = await queue.process_retries()
        assert count == 0
        mock_store.schedule_retry.assert_called_once()


# ── reset_failed tests ───────────────────────────────────────────────────────


class TestResetFailed:
    """reset_failed should reset failed files to processing."""

    def test_resets_all_failed(self, queue: RetryQueue, mock_store: MagicMock) -> None:
        mock_store.list_records.return_value = [
            {"source": "/docs/a.pdf", "status": IngestionStatus.FAILED},
            {"source": "/docs/b.pdf", "status": IngestionStatus.FAILED},
        ]
        assert queue.reset_failed() == 2
        assert mock_store.reset_for_retry.call_count == 2

    def test_resets_with_filter(self, queue: RetryQueue, mock_store: MagicMock) -> None:
        mock_store.list_records.return_value = [
            {"source": "/docs/a.pdf", "status": IngestionStatus.FAILED},
            {"source": "/docs/b.pdf", "status": IngestionStatus.FAILED},
        ]
        assert queue.reset_failed(status_filter="a.pdf") == 1
