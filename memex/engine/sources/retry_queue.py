"""RetryQueue — exponential backoff retry for failed Docling operations.

Determines if errors are retryable and schedules retries with exponential backoff.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from memex.engine.sources.status_tracker import FileStatus

logger = logging.getLogger("retry-queue")


class RetryQueue:
    """Exponential backoff retry queue for failed Docling operations.

    Args:
        status_tracker: StatusTracker instance for updating file status.
        docling_client: DoclingAsyncClient for resubmitting jobs.
    """

    BACKOFF_SCHEDULE = [
        60,       # 1 minute
        300,      # 5 minutes
        1800,     # 30 minutes
        7200,     # 2 hours
    ]
    MAX_RETRIES = 4

    def __init__(self, status_tracker: Any, docling_client: Any) -> None:
        self._tracker = status_tracker
        self._docling = docling_client

    def should_retry(self, error: str, retry_count: int) -> bool:
        """Determine if error is retryable.

        Args:
            error: Error message string.
            retry_count: Current retry attempt count.

        Returns:
            True if error is retryable and under max retries.
        """
        if retry_count >= self.MAX_RETRIES:
            return False
        retryable = ["timeout", "503", "504", "connection", "gateway"]
        return any(r in error.lower() for r in retryable)

    def schedule_retry(self, source_id: str, error: str, retry_count: int) -> None:
        """Schedule retry with exponential backoff.

        Args:
            source_id: File path or URL identifier.
            error: Error message.
            retry_count: Current retry attempt count.
        """
        if not self.should_retry(error, retry_count):
            self._tracker.update_status(source_id=source_id, status=FileStatus.FAILED, error=error)
            return

        backoff = self.BACKOFF_SCHEDULE[min(retry_count, len(self.BACKOFF_SCHEDULE) - 1)]
        next_retry = datetime.now(UTC) + timedelta(seconds=backoff)
        self._tracker.update_status(
            source_id=source_id,
            status=FileStatus.RETRY,
            error=error,
            retry_count=retry_count + 1,
            next_retry_at=next_retry.isoformat(),
        )

    async def process_retries(self) -> int:
        """Process files due for retry.

        Returns:
            Count of files successfully resubmitted.
        """
        pending = self._tracker.get_pending_retries()
        retried = 0
        for source_id in pending:
            try:
                # Re-submit to Docling (payload would come from stored state)
                task_id = await self._docling.submit_conversion({})
                self._tracker.update_status(
                    source_id=source_id,
                    status=FileStatus.PROCESSING,
                    docling_task_id=task_id,
                )
                retried += 1
            except Exception as exc:
                logger.warning("Retry failed for %s: %s", source_id, exc)
                self.schedule_retry(source_id, str(exc), retry_count=0)
        return retried
