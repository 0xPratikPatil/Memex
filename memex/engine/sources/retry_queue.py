"""RetryQueue — exponential backoff retry for failed ingestion operations.

Determines whether an error is retryable (typed ``MemexError`` subclasses or
known transient HTTP states) and schedules retries with exponential backoff
through :class:`FileStatusStore`.

This replaces the previous dead implementation that resubmitted a bogus empty
payload to a Docling async client that was never wired into any path.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from memex.engine.core.errors import (
    ConfigError,
    ConversionError,
    ConversionTimeoutError,
    CorruptedDocumentError,
    EmbeddingError,
    RetrievalError,
    ServiceUnavailableError,
    StorageError,
)
from memex.engine.ingestion.status import FileStatusStore, IngestionStatus

logger = logging.getLogger("retry-queue")

# Typed errors that indicate a transient condition worth retrying.
# NOTE: ConversionTimeoutError must be checked before ConversionError.
_RETRYABLE_ERROR_TYPES = (
    ConversionTimeoutError,
    ServiceUnavailableError,
    RetrievalError,
    EmbeddingError,
    StorageError,
)

# Typed errors that are deterministic — retrying will not help.
_NON_RETRYABLE_ERROR_TYPES = (ConfigError, ConversionError, CorruptedDocumentError)


class RetryQueue:
    """Exponential backoff retry queue for failed ingestion operations.

    Args:
        status_store: FileStatusStore instance for reading/scheduling status.
        ingest_fn: Callable that re-ingests a single file (parse → ingest).
            Signature: ``ingest_fn(source) -> None``. If None, retries are
            surfaced via ``schedule_retry`` only and not auto-resubmitted.
    """

    BACKOFF_SCHEDULE = [
        60,       # 1 minute
        300,      # 5 minutes
        1800,     # 30 minutes
        7200,     # 2 hours
    ]
    MAX_RETRIES = 4

    def __init__(self, status_store: FileStatusStore, ingest_fn: Any | None = None) -> None:
        self._store = status_store
        self._ingest_fn = ingest_fn

    def should_retry(self, error: str, retry_count: int, exc: BaseException | None = None) -> bool:
        """Determine if an error is retryable.

        Args:
            error: Error message string.
            retry_count: Current retry attempt count.
            exc: Optional original exception for typed matching.

        Returns:
            True if retryable and under max retries.
        """
        if retry_count >= self.MAX_RETRIES:
            return False

        # Typed matching takes precedence over string matching.
        # NOTE: retryable is checked FIRST because ConversionTimeoutError is a
        # subclass of ConversionError (a non-retryable type).
        if exc is not None:
            if isinstance(exc, _RETRYABLE_ERROR_TYPES):
                return True
            if isinstance(exc, _NON_RETRYABLE_ERROR_TYPES):
                return False

        return any(t in error.lower() for t in ("timeout", "503", "504", "connection", "gateway", "unavailable"))

    def schedule_retry(self, source: str, error: str, retry_count: int, exc: BaseException | None = None) -> None:
        """Schedule a retry with exponential backoff, or mark permanently failed."""
        if not self.should_retry(error, retry_count, exc=exc):
            self._store.mark_failed(source, error, exc=exc)
            return

        backoff = self.BACKOFF_SCHEDULE[min(retry_count, len(self.BACKOFF_SCHEDULE) - 1)]
        self._store.schedule_retry(source, error, attempts=retry_count + 1, backoff_s=backoff)
        logger.info(
            "Scheduled retry for %s in %ds (attempt %d)",
            source,
            backoff,
            retry_count + 1,
        )

    async def process_retries(self) -> int:
        """Re-ingest files whose retry window has passed.

        Returns:
            Count of files successfully resubmitted.
        """
        due = self._store.get_due_retries(datetime.now(UTC))
        if not due:
            return 0
        if self._ingest_fn is None:
            logger.warning("No ingest_fn configured — %d due retries not resubmitted", len(due))
            return 0

        resubmitted = 0
        for source in due:
            try:
                await self._ingest_fn(source)
                self._store.reset_for_retry(source)
                resubmitted += 1
            except Exception as exc:
                logger.warning("Retry failed for %s: %s", source, exc)
                record = self._store.get_status(source) or {}
                self.schedule_retry(source, str(exc), retry_count=record.get("attempts", 0) or 0, exc=exc)
        return resubmitted

    def reset_failed(self, status_filter: str | None = None) -> int:
        """Manually reset failed files to processing (bypasses backoff).

        Args:
            status_filter: Optional source substring to limit the reset.

        Returns:
            Count of files reset.
        """
        count = 0
        for record in self._store.list_records(status_filter=IngestionStatus.FAILED):
            source = record["source"]
            if status_filter and status_filter not in source:
                continue
            self._store.reset_for_retry(source)
            count += 1
        return count


__all__ = ["RetryQueue"]
