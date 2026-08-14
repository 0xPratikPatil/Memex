"""Unit tests for RetryQueue — exponential backoff retry for failed Docling operations."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memex.engine.sources.retry_queue import RetryQueue
from memex.engine.sources.status_tracker import FileStatus


@pytest.fixture
def mock_tracker() -> MagicMock:
    """Create a mock StatusTracker."""
    return MagicMock()


@pytest.fixture
def mock_docling() -> AsyncMock:
    """Create a mock DoclingAsyncClient."""
    return AsyncMock()


@pytest.fixture
def queue(mock_tracker: MagicMock, mock_docling: AsyncMock) -> RetryQueue:
    """Create a RetryQueue with mocks."""
    return RetryQueue(mock_tracker, mock_docling)


# ── should_retry tests ───────────────────────────────────────────────────────


class TestShouldRetry:
    """should_retry should determine if error is retryable."""

    def test_retries_timeout_error(self, queue: RetryQueue) -> None:
        """Should retry on timeout error."""
        assert queue.should_retry("504 Gateway Timeout", retry_count=0) is True

    def test_retries_503_error(self, queue: RetryQueue) -> None:
        """Should retry on 503 Service Unavailable."""
        assert queue.should_retry("503 Service Unavailable", retry_count=0) is True

    def test_retries_504_error(self, queue: RetryQueue) -> None:
        """Should retry on 504 Gateway Timeout."""
        assert queue.should_retry("504 Gateway Timeout", retry_count=0) is True

    def test_retries_connection_error(self, queue: RetryQueue) -> None:
        """Should retry on connection error."""
        assert queue.should_retry("Connection refused", retry_count=0) is True

    def test_does_not_retry_unknown_error(self, queue: RetryQueue) -> None:
        """Should not retry on unknown errors."""
        assert queue.should_retry("Invalid format", retry_count=0) is False

    def test_does_not_retry_after_max_attempts(self, queue: RetryQueue) -> None:
        """Should not retry after max retries exceeded."""
        assert queue.should_retry("504 Gateway Timeout", retry_count=4) is False

    def test_retries_up_to_max(self, queue: RetryQueue) -> None:
        """Should retry up to MAX_RETRIES."""
        assert queue.should_retry("504 timeout", retry_count=3) is True
        assert queue.should_retry("504 timeout", retry_count=4) is False


# ── schedule_retry tests ─────────────────────────────────────────────────────


class TestScheduleRetry:
    """schedule_retry should schedule retry with exponential backoff."""

    def test_schedules_retry(self, queue: RetryQueue, mock_tracker: MagicMock) -> None:
        """Should update status to RETRY with next_retry_at."""
        queue.schedule_retry("/docs/report.pdf", "504 timeout", retry_count=0)

        mock_tracker.update_status.assert_called_once()
        call_kwargs = mock_tracker.update_status.call_args[1]
        assert call_kwargs["source_id"] == "/docs/report.pdf"
        assert call_kwargs["status"] == FileStatus.RETRY
        assert call_kwargs["retry_count"] == 1
        assert "next_retry_at" in call_kwargs

    def test_marks_failed_after_max_retries(self, queue: RetryQueue, mock_tracker: MagicMock) -> None:
        """Should mark as FAILED when max retries exceeded."""
        queue.schedule_retry("/docs/report.pdf", "504 timeout", retry_count=4)

        mock_tracker.update_status.assert_called_once()
        call_kwargs = mock_tracker.update_status.call_args[1]
        assert call_kwargs["status"] == FileStatus.FAILED

    def test_backoff_increases(self, queue: RetryQueue, mock_tracker: MagicMock) -> None:
        """Should increase backoff with each retry attempt."""
        # First retry: 60s
        queue.schedule_retry("/docs/report.pdf", "504 timeout", retry_count=0)
        first_call = mock_tracker.update_status.call_args[1]

        # Second retry: 300s
        queue.schedule_retry("/docs/report.pdf", "504 timeout", retry_count=1)
        second_call = mock_tracker.update_status.call_args[1]

        # Verify next_retry_at is further in the future for second retry
        from datetime import datetime

        first_retry = datetime.fromisoformat(first_call["next_retry_at"])
        second_retry = datetime.fromisoformat(second_call["next_retry_at"])
        assert second_retry > first_retry


# ── process_retries tests ────────────────────────────────────────────────────


class TestProcessRetries:
    """process_retries should retry files due for retry."""

    @pytest.mark.asyncio
    async def test_returns_count(self, queue: RetryQueue, mock_tracker: MagicMock, mock_docling: AsyncMock) -> None:
        """Should return count of retried files."""
        mock_tracker.get_pending_retries.return_value = ["/docs/report.pdf"]
        mock_docling.submit_conversion.return_value = "task-123"

        count = await queue.process_retries()

        assert count == 1

    @pytest.mark.asyncio
    async def test_returns_zero_when_empty(self, queue: RetryQueue, mock_tracker: MagicMock) -> None:
        """Should return 0 when no files need retry."""
        mock_tracker.get_pending_retries.return_value = []

        count = await queue.process_retries()

        assert count == 0

    @pytest.mark.asyncio
    async def test_updates_status_to_processing(
        self, queue: RetryQueue, mock_tracker: MagicMock, mock_docling: AsyncMock
    ) -> None:
        """Should update status to PROCESSING after successful submission."""
        mock_tracker.get_pending_retries.return_value = ["/docs/report.pdf"]
        mock_docling.submit_conversion.return_value = "task-456"

        await queue.process_retries()

        call_kwargs = mock_tracker.update_status.call_args[1]
        assert call_kwargs["status"] == FileStatus.PROCESSING
        assert call_kwargs["docling_task_id"] == "task-456"

    @pytest.mark.asyncio
    async def test_handles_submission_failure(
        self, queue: RetryQueue, mock_tracker: MagicMock, mock_docling: AsyncMock
    ) -> None:
        """Should schedule retry when submission fails."""
        mock_tracker.get_pending_retries.return_value = ["/docs/report.pdf"]
        mock_docling.submit_conversion.side_effect = Exception("Connection refused")

        count = await queue.process_retries()

        assert count == 0
        # Should have called schedule_retry (which calls update_status with RETRY or FAILED)
        mock_tracker.update_status.assert_called()
