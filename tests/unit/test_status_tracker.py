"""Unit tests for StatusTracker — Qdrant-based file status tracking."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from memex.engine.sources.status_tracker import FileStatus, StatusTracker


@pytest.fixture
def mock_qdrant() -> MagicMock:
    """Create a mock Qdrant client."""
    return MagicMock()


@pytest.fixture
def tracker(mock_qdrant: MagicMock) -> StatusTracker:
    """Create a StatusTracker with mock Qdrant."""
    return StatusTracker(mock_qdrant, "test_collection")


# ── FileStatus constants ─────────────────────────────────────────────────────


class TestFileStatus:
    """FileStatus should define status constants."""

    def test_has_pending(self) -> None:
        assert FileStatus.PENDING == "pending"

    def test_has_processing(self) -> None:
        assert FileStatus.PROCESSING == "processing"

    def test_has_done(self) -> None:
        assert FileStatus.DONE == "done"

    def test_has_retry(self) -> None:
        assert FileStatus.RETRY == "retry"

    def test_has_failed(self) -> None:
        assert FileStatus.FAILED == "failed"


# ── update_status tests ──────────────────────────────────────────────────────


class TestUpdateStatus:
    """update_status should update file status in Qdrant payload."""

    def test_updates_status(self, tracker: StatusTracker, mock_qdrant: MagicMock) -> None:
        """Should call set_payload with correct status."""
        mock_point = MagicMock()
        mock_point.id = "point-123"
        mock_qdrant.scroll.return_value = ([mock_point], None)

        tracker.update_status("/docs/report.pdf", FileStatus.PROCESSING)

        mock_qdrant.set_payload.assert_called_once()
        call_kwargs = mock_qdrant.set_payload.call_args[1]
        assert call_kwargs["payload"]["processing_status"] == "processing"
        assert call_kwargs["collection_name"] == "test_collection"

    def test_updates_with_error(self, tracker: StatusTracker, mock_qdrant: MagicMock) -> None:
        """Should include error in payload when provided."""
        mock_point = MagicMock()
        mock_point.id = "point-456"
        mock_qdrant.scroll.return_value = ([mock_point], None)

        tracker.update_status("/docs/report.pdf", FileStatus.FAILED, error="504 timeout")

        call_kwargs = mock_qdrant.set_payload.call_args[1]
        assert call_kwargs["payload"]["last_error"] == "504 timeout"

    def test_updates_with_retry_count(self, tracker: StatusTracker, mock_qdrant: MagicMock) -> None:
        """Should include retry_count in payload when > 0."""
        mock_point = MagicMock()
        mock_point.id = "point-789"
        mock_qdrant.scroll.return_value = ([mock_point], None)

        tracker.update_status("/docs/report.pdf", FileStatus.RETRY, retry_count=2)

        call_kwargs = mock_qdrant.set_payload.call_args[1]
        assert call_kwargs["payload"]["retry_count"] == 2

    def test_updates_with_next_retry_at(self, tracker: StatusTracker, mock_qdrant: MagicMock) -> None:
        """Should include next_retry_at in payload when provided."""
        mock_point = MagicMock()
        mock_point.id = "point-abc"
        mock_qdrant.scroll.return_value = ([mock_point], None)

        retry_time = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
        tracker.update_status("/docs/report.pdf", FileStatus.RETRY, next_retry_at=retry_time)

        call_kwargs = mock_qdrant.set_payload.call_args[1]
        assert call_kwargs["payload"]["next_retry_at"] == retry_time

    def test_updates_with_docling_task_id(self, tracker: StatusTracker, mock_qdrant: MagicMock) -> None:
        """Should include docling_task_id in payload when provided."""
        mock_point = MagicMock()
        mock_point.id = "point-def"
        mock_qdrant.scroll.return_value = ([mock_point], None)

        tracker.update_status("/docs/report.pdf", FileStatus.PROCESSING, docling_task_id="task-xyz")

        call_kwargs = mock_qdrant.set_payload.call_args[1]
        assert call_kwargs["payload"]["docling_task_id"] == "task-xyz"

    def test_no_update_when_not_found(self, tracker: StatusTracker, mock_qdrant: MagicMock) -> None:
        """Should not call set_payload when source_id not found."""
        mock_qdrant.scroll.return_value = ([], None)

        tracker.update_status("/docs/missing.pdf", FileStatus.DONE)

        mock_qdrant.set_payload.assert_not_called()

    def test_sets_timestamp(self, tracker: StatusTracker, mock_qdrant: MagicMock) -> None:
        """Should include status_updated_at timestamp."""
        mock_point = MagicMock()
        mock_point.id = "point-ts"
        mock_qdrant.scroll.return_value = ([mock_point], None)

        tracker.update_status("/docs/report.pdf", FileStatus.DONE)

        call_kwargs = mock_qdrant.set_payload.call_args[1]
        assert "status_updated_at" in call_kwargs["payload"]


# ── get_pending_retries tests ────────────────────────────────────────────────


class TestGetPendingRetries:
    """get_pending_retries should return files due for retry."""

    def test_returns_source_ids(self, tracker: StatusTracker, mock_qdrant: MagicMock) -> None:
        """Should return list of source_id strings."""
        mock_point = MagicMock()
        mock_point.payload = {
            "source_id": "/docs/report.pdf",
            "next_retry_at": (datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
        }
        mock_qdrant.scroll.return_value = ([mock_point], None)

        result = tracker.get_pending_retries()

        assert result == ["/docs/report.pdf"]

    def test_returns_empty_when_no_retries(self, tracker: StatusTracker, mock_qdrant: MagicMock) -> None:
        """Should return empty list when no files need retry."""
        mock_qdrant.scroll.return_value = ([], None)

        result = tracker.get_pending_retries()

        assert result == []

    def test_filters_by_retry_status(self, tracker: StatusTracker, mock_qdrant: MagicMock) -> None:
        """Should query for processing_status=retry."""
        mock_qdrant.scroll.return_value = ([], None)

        tracker.get_pending_retries()

        call_kwargs = mock_qdrant.scroll.call_args[1]
        scroll_filter = call_kwargs["scroll_filter"]
        assert scroll_filter is not None


# ── get_status_summary tests ─────────────────────────────────────────────────


class TestGetStatusSummary:
    """get_status_summary should return counts by status."""

    def test_returns_counts(self, tracker: StatusTracker, mock_qdrant: MagicMock) -> None:
        """Should return dict with counts for each status."""
        mock_qdrant.count.return_value = 5

        result = tracker.get_status_summary()

        assert "pending" in result
        assert "processing" in result
        assert "done" in result
        assert "retry" in result
        assert "failed" in result

    def test_counts_all_statuses(self, tracker: StatusTracker, mock_qdrant: MagicMock) -> None:
        """Should call count for each status type."""
        tracker.get_status_summary()

        assert mock_qdrant.count.call_count == 5
