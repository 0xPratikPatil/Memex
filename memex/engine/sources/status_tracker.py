"""StatusTracker — Qdrant-based file status tracking.

Stores file processing status in Qdrant payload alongside vectors.
Tracks: pending, processing, done, retry, failed states.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue

logger = logging.getLogger("status-tracker")


class FileStatus:
    """File processing status constants."""

    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    RETRY = "retry"
    FAILED = "failed"


class StatusTracker:
    """Track file processing status in Qdrant payload.

    Args:
        qdrant_client: Qdrant client instance.
        collection: Qdrant collection name.
    """

    def __init__(self, qdrant_client: Any, collection: str) -> None:
        self._qdrant = qdrant_client
        self._collection = collection

    def update_status(
        self,
        source_id: str,
        status: str,
        error: str | None = None,
        retry_count: int = 0,
        next_retry_at: str | None = None,
        docling_task_id: str | None = None,
    ) -> None:
        """Update file status in Qdrant payload.

        Args:
            source_id: File path or URL identifier.
            status: New status (FileStatus constant).
            error: Error message if failed.
            retry_count: Number of retry attempts.
            next_retry_at: ISO timestamp for next retry.
            docling_task_id: Docling async task ID.
        """
        # Find points with this source_id
        results = self._qdrant.scroll(
            collection_name=self._collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
            ),
            limit=1,
        )
        points = results[0] if results else []
        if not points:
            return

        point_id = points[0].id
        payload_update: dict = {
            "processing_status": status,
            "status_updated_at": datetime.now(UTC).isoformat(),
        }
        if error:
            payload_update["last_error"] = error
        if retry_count > 0:
            payload_update["retry_count"] = retry_count
        if next_retry_at:
            payload_update["next_retry_at"] = next_retry_at
        if docling_task_id:
            payload_update["docling_task_id"] = docling_task_id

        self._qdrant.set_payload(
            collection_name=self._collection,
            payload=payload_update,
            points=[point_id],
        )

    def get_pending_retries(self) -> list[str]:
        """Get files due for retry (status=retry and next_retry_at <= now).

        Returns:
            List of source_id strings.
        """
        results, _ = self._qdrant.scroll(
            collection_name=self._collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="processing_status", match=MatchValue(value=FileStatus.RETRY)),
                ]
            ),
            limit=100,
        )
        # Filter by next_retry_at in Python (Qdrant Range doesn't support strings)
        now = datetime.now(UTC)
        pending = []
        for p in results:
            next_retry = p.payload.get("next_retry_at", "")
            if next_retry:
                try:
                    retry_time = datetime.fromisoformat(next_retry)
                    if retry_time <= now:
                        pending.append(p.payload.get("source_id", ""))
                except (ValueError, TypeError):
                    pass
        return pending

    def get_status_summary(self) -> dict[str, int]:
        """Get counts by status.

        Returns:
            Dict with counts for each FileStatus.
        """
        summary: dict[str, int] = {}
        for status in [
            FileStatus.PENDING,
            FileStatus.PROCESSING,
            FileStatus.DONE,
            FileStatus.RETRY,
            FileStatus.FAILED,
        ]:
            count = self._qdrant.count(
                collection_name=self._collection,
                count_filter=Filter(
                    must=[FieldCondition(key="processing_status", match=MatchValue(value=status))]
                ),
            )
            summary[status] = count
        return summary
