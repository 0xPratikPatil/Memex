"""Unit tests for FileStatusStore — Qdrant-backed file status state machine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from memex.engine.core.errors import StorageError
from memex.engine.ingestion.status import (
    VALID_TRANSITIONS,
    FileStatusStore,
    IngestionStatus,
)


@pytest.fixture
def mock_qdrant() -> MagicMock:
    """Create a mock Qdrant client."""
    client = MagicMock()
    client.collection_exists.return_value = True
    return client


@pytest.fixture
def store(mock_qdrant: MagicMock) -> FileStatusStore:
    """Create a FileStatusStore with mock Qdrant."""
    return FileStatusStore(mock_qdrant, collection="test_status")


def _mock_points(payloads: list[dict]) -> list[MagicMock]:
    points = []
    for p in payloads:
        point = MagicMock()
        point.payload = p
        points.append(point)
    return points


# ── State machine constants ──────────────────────────────────────────────────


class TestStateMachine:
    """VALID_TRANSITIONS should encode a legal state machine."""

    def test_terminal_states_not_sources(self) -> None:
        # DONE/SKIPPED are terminal; no transition INTO them other than from PROCESSING.
        assert IngestionStatus.DONE in VALID_TRANSITIONS[IngestionStatus.PROCESSING]
        assert IngestionStatus.SKIPPED in VALID_TRANSITIONS[IngestionStatus.PROCESSING]

    def test_failed_retryable(self) -> None:
        assert IngestionStatus.RETRY in VALID_TRANSITIONS[IngestionStatus.FAILED]
        assert IngestionStatus.PROCESSING in VALID_TRANSITIONS[IngestionStatus.FAILED]

    def test_no_illegal_jump_to_done_from_pending(self) -> None:
        assert IngestionStatus.DONE not in VALID_TRANSITIONS[IngestionStatus.PENDING]


# ── Lifecycle ────────────────────────────────────────────────────────────────


class TestLifecycle:
    def test_mark_pending_upserts(self, store: FileStatusStore, mock_qdrant: MagicMock) -> None:
        store.mark_pending("/docs/a.pdf", source_name="docs")
        mock_qdrant.upsert.assert_called_once()
        point = mock_qdrant.upsert.call_args[1]["points"][0]
        assert point.payload["status"] == IngestionStatus.PENDING
        assert point.payload["source_name"] == "docs"

    def test_valid_transition_done(self, store: FileStatusStore, mock_qdrant: MagicMock) -> None:
        mock_qdrant.scroll.return_value = (
            _mock_points([{"source": "/docs/a.pdf", "status": IngestionStatus.PROCESSING}]),
            None,
        )
        store.mark_done("/docs/a.pdf", chunks=7)
        point = mock_qdrant.upsert.call_args[1]["points"][0]
        assert point.payload["status"] == IngestionStatus.DONE
        assert point.payload["chunks"] == 7

    def test_illegal_transition_raises(self, store: FileStatusStore, mock_qdrant: MagicMock) -> None:
        # PENDING → DONE is illegal.
        mock_qdrant.scroll.return_value = (
            _mock_points([{"source": "/docs/a.pdf", "status": IngestionStatus.PENDING}]),
            None,
        )
        with pytest.raises(StorageError):
            store.mark_done("/docs/a.pdf")

    def test_new_source_defaults_to_pending(self, store: FileStatusStore, mock_qdrant: MagicMock) -> None:
        # No existing record → treats as PENDING; DONE from PENDING is illegal.
        mock_qdrant.scroll.return_value = ([], None)
        with pytest.raises(StorageError):
            store.mark_done("/docs/a.pdf")

    def test_mark_failed_legal_from_non_terminal_states(self, store: FileStatusStore, mock_qdrant: MagicMock) -> None:
        # FAILED is reachable from PENDING, PROCESSING, FAILED, RETRY — not from terminal DONE/SKIPPED.
        source_states = (
            IngestionStatus.PENDING,
            IngestionStatus.PROCESSING,
            IngestionStatus.FAILED,
            IngestionStatus.RETRY,
        )
        for status in source_states:
            mock_qdrant.scroll.return_value = (
                _mock_points([{"source": "/docs/a.pdf", "status": status}]),
                None,
            )
            store.mark_failed("/docs/a.pdf", "boom", stage="Converting")
        # No exception raised for any of the above source states.

    def test_mark_failed_illegal_from_done(self, store: FileStatusStore, mock_qdrant: MagicMock) -> None:
        mock_qdrant.scroll.return_value = (
            _mock_points([{"source": "/docs/a.pdf", "status": IngestionStatus.DONE}]),
            None,
        )
        with pytest.raises(StorageError):
            store.mark_failed("/docs/a.pdf", "boom")

    def test_mark_failed_records_error_and_type(self, store: FileStatusStore, mock_qdrant: MagicMock) -> None:
        mock_qdrant.scroll.return_value = (
            _mock_points([{"source": "/docs/a.pdf", "status": IngestionStatus.PROCESSING}]),
            None,
        )
        try:
            raise ValueError("bad thing")
        except ValueError as exc:
            store.mark_failed("/docs/a.pdf", "bad thing", exc=exc, stage="Converting")
        point = mock_qdrant.upsert.call_args[1]["points"][0]
        assert point.payload["error"] == "bad thing"
        assert point.payload["error_type"] == "ValueError"

    def test_stage_update_self_loop(self, store: FileStatusStore, mock_qdrant: MagicMock) -> None:
        mock_qdrant.scroll.return_value = (
            _mock_points([{"source": "/docs/a.pdf", "status": IngestionStatus.PROCESSING}]),
            None,
        )
        store.update_stage("/docs/a.pdf", "Embedding")
        point = mock_qdrant.upsert.call_args[1]["points"][0]
        assert point.payload["status"] == IngestionStatus.PROCESSING
        assert point.payload["stage"] == "Embedding"

    def test_transition_preserves_existing_fields(self, store: FileStatusStore, mock_qdrant: MagicMock) -> None:
        # Stage update from an existing record must keep source_name/created_at.
        mock_qdrant.scroll.return_value = (
            _mock_points(
                [
                    {
                        "source": "/docs/a.pdf",
                        "status": IngestionStatus.PROCESSING,
                        "source_name": "docs",
                        "created_at": "2026-01-01T00:00:00+00:00",
                    }
                ]
            ),
            None,
        )
        store.update_stage("/docs/a.pdf", "Embedding", chunks=3)
        point = mock_qdrant.upsert.call_args[1]["points"][0]
        assert point.payload["source_name"] == "docs"
        assert point.payload["created_at"] == "2026-01-01T00:00:00+00:00"
        assert point.payload["stage"] == "Embedding"


# ── Retry scheduling ─────────────────────────────────────────────────────────


class TestRetryScheduling:
    def test_schedule_retry_sets_next_retry_at(self, store: FileStatusStore, mock_qdrant: MagicMock) -> None:
        mock_qdrant.scroll.return_value = (
            _mock_points([{"source": "/docs/a.pdf", "status": IngestionStatus.FAILED}]),
            None,
        )
        store.schedule_retry("/docs/a.pdf", "504 timeout", attempts=1, backoff_s=60)
        point = mock_qdrant.upsert.call_args[1]["points"][0]
        assert point.payload["status"] == IngestionStatus.RETRY
        assert point.payload["attempts"] == 1
        assert point.payload["next_retry_at"] > datetime.now(UTC).isoformat()

    def test_reset_for_retry(self, store: FileStatusStore, mock_qdrant: MagicMock) -> None:
        mock_qdrant.scroll.return_value = (
            _mock_points([{"source": "/docs/a.pdf", "status": IngestionStatus.FAILED}]),
            None,
        )
        store.reset_for_retry("/docs/a.pdf")
        point = mock_qdrant.upsert.call_args[1]["points"][0]
        assert point.payload["status"] == IngestionStatus.PROCESSING


# ── Queries ──────────────────────────────────────────────────────────────────


class TestQueries:
    def test_get_summary_counts(self, store: FileStatusStore, mock_qdrant: MagicMock) -> None:
        mock_qdrant.scroll.return_value = (
            _mock_points(
                [
                    {"source": "a", "status": IngestionStatus.DONE},
                    {"source": "b", "status": IngestionStatus.DONE},
                    {"source": "c", "status": IngestionStatus.FAILED},
                ]
            ),
            None,
        )
        summary = store.get_summary()
        assert summary[IngestionStatus.DONE] == 2
        assert summary[IngestionStatus.FAILED] == 1

    def test_list_records_sorted(self, store: FileStatusStore, mock_qdrant: MagicMock) -> None:
        mock_qdrant.scroll.return_value = (
            _mock_points(
                [
                    {"source": "old", "status": IngestionStatus.DONE, "updated_at": "2026-01-01T00:00:00+00:00"},
                    {"source": "new", "status": IngestionStatus.PROCESSING, "updated_at": "2026-08-01T00:00:00+00:00"},
                ]
            ),
            None,
        )
        records = store.list_records()
        assert records[0]["source"] == "new"

    def test_get_due_retries(self, store: FileStatusStore, mock_qdrant: MagicMock) -> None:
        past = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
        future = (datetime.now(UTC) + timedelta(minutes=10)).isoformat()
        mock_qdrant.scroll.return_value = (
            _mock_points(
                [
                    {"source": "due", "status": IngestionStatus.RETRY, "next_retry_at": past},
                    {"source": "later", "status": IngestionStatus.RETRY, "next_retry_at": future},
                ]
            ),
            None,
        )
        assert store.get_due_retries() == ["due"]


# ── Cleanup ──────────────────────────────────────────────────────────────────


class TestCleanup:
    def test_cleanup_stale_marks_failed(self, store: FileStatusStore, mock_qdrant: MagicMock) -> None:
        stale = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        fresh = datetime.now(UTC).isoformat()
        mock_qdrant.scroll.return_value = (
            _mock_points(
                [
                    {"source": "zombie", "status": IngestionStatus.PROCESSING, "updated_at": stale},
                    {"source": "active", "status": IngestionStatus.PROCESSING, "updated_at": fresh},
                ]
            ),
            None,
        )
        cleaned = store.cleanup_stale(stale_after_s=7 * 86400)
        assert cleaned == 1


# ── Collection setup ─────────────────────────────────────────────────────────


class TestCollectionSetup:
    def test_creates_collection_when_missing(self, store: FileStatusStore, mock_qdrant: MagicMock) -> None:
        mock_qdrant.collection_exists.return_value = False
        store.mark_pending("/docs/a.pdf")
        mock_qdrant.create_collection.assert_called_once()
        mock_qdrant.create_payload_index.assert_called_once()
