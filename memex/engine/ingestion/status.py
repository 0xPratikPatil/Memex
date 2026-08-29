"""FileStatusStore — single source of truth for per-file ingestion status.

Backed by a dedicated Qdrant collection (one point per file) so a file has a
status record from ``mark_pending`` onward — including failures that happen
before any chunk is stored (the F6 flaw in the old StatusTracker, which wrote
to chunk payloads).

Two-axis model:
  - ``status`` — coarse lifecycle state machine (drives retry/resume).
  - ``stage``  — fine live position shown in UI, updated within ``processing``
    via a self-loop.

All writers (pipeline, sync, ingestion orchestrator, CLI, MCP) go through this
store. Invalid transitions raise :class:`StorageError` — never a silent no-op.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue

from memex.engine.core.errors import ServiceUnavailableError, StorageError
from memex.engine.core.progress import PipelineStage

logger = logging.getLogger("file-status")

COLLECTION = "memex_file_status"


class IngestionStatus:
    """Coarse lifecycle status values."""

    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"
    RETRY = "retry"


# Explicit state machine — no illegal transitions.
# Terminal self-transitions (done→done, skipped→skipped, failed→failed) are
# allowed: multiple writers (pipeline, CLI, sync, MCP) may re-report the same
# terminal state — e.g. the pipeline marks a file done and the CLI confirms it.
VALID_TRANSITIONS: dict[str, set[str]] = {
    IngestionStatus.PENDING: {IngestionStatus.PROCESSING, IngestionStatus.SKIPPED, IngestionStatus.FAILED},
    IngestionStatus.PROCESSING: {
        IngestionStatus.PROCESSING,  # self-loop = stage update
        IngestionStatus.DONE,
        IngestionStatus.SKIPPED,
        IngestionStatus.FAILED,
    },
    IngestionStatus.FAILED: {
        IngestionStatus.RETRY,
        IngestionStatus.PROCESSING,
        IngestionStatus.FAILED,  # re-failure during retry
    },
    IngestionStatus.RETRY: {IngestionStatus.PROCESSING, IngestionStatus.FAILED},
    IngestionStatus.DONE: {IngestionStatus.PROCESSING, IngestionStatus.DONE},
    IngestionStatus.SKIPPED: {IngestionStatus.PROCESSING, IngestionStatus.SKIPPED},
}

_TERMINAL = {IngestionStatus.DONE, IngestionStatus.SKIPPED}


def _point_id(source: str) -> str:
    """Deterministic UUID point ID from a source identifier (idempotent upserts)."""
    digest = hashlib.sha1(source.encode("utf-8")).digest()[:16]
    return str(uuid.UUID(bytes=digest))


def _now() -> str:
    return datetime.now(UTC).isoformat()


class FileStatusStore:
    """Persist per-file ingestion status in a dedicated Qdrant collection.

    Args:
        qdrant_client: Qdrant client instance.
        collection: Override collection name (default ``memex_file_status``).
    """

    def __init__(self, qdrant_client: Any, collection: str = COLLECTION) -> None:
        self._qdrant = qdrant_client
        self._collection = collection

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def mark_pending(self, source: str, source_name: str = "") -> None:
        """Record a file as pending before work starts."""
        self._ensure_collection()
        self._upsert(
            source,
            {
                "status": IngestionStatus.PENDING,
                "stage": PipelineStage.SCANNING,
                "source_name": source_name,
                "created_at": _now(),
                "updated_at": _now(),
            },
        )

    def update_stage(self, source: str, stage: str, chunks: int = 0) -> None:
        """Advance the fine stage within ``processing`` (self-loop transition)."""
        self._ensure_collection()
        self._transition(
            source,
            IngestionStatus.PROCESSING,
            payload={"stage": stage, "chunks": chunks, "updated_at": _now()},
        )

    def mark_done(self, source: str, chunks: int = 0) -> None:
        """Mark a file successfully ingested."""
        self._ensure_collection()
        self._transition(
            source,
            IngestionStatus.DONE,
            payload={"stage": PipelineStage.DONE, "chunks": chunks, "completed_at": _now(), "updated_at": _now()},
        )

    def mark_skipped(self, source: str, reason: str = "") -> None:
        """Mark a file skipped (dedup / unchanged / timeout-skip)."""
        self._ensure_collection()
        self._transition(
            source,
            IngestionStatus.SKIPPED,
            payload={"stage": PipelineStage.SKIPPED, "error": reason, "completed_at": _now(), "updated_at": _now()},
        )

    def mark_failed(
        self,
        source: str,
        error: str,
        exc: BaseException | None = None,
        stage: str = PipelineStage.ERROR,
    ) -> None:
        """Mark a file failed from ANY stage (always legal)."""
        from memex.engine.core.errors import error_context

        ctx = error_context(exc) if exc is not None else {}
        self._ensure_collection()
        self._transition(
            source,
            IngestionStatus.FAILED,
            payload={
                "stage": stage,
                "error": error,
                "error_type": ctx.get("error_type", type(exc).__name__ if exc else ""),
                "hint": ctx.get("hint", ""),
                "updated_at": _now(),
            },
        )

    def mark_deleted(self, source: str) -> None:
        """Record that a file's chunks were removed by sync reconciliation."""
        self._ensure_collection()
        self._upsert(
            source,
            {"status": IngestionStatus.SKIPPED, "stage": PipelineStage.DELETING, "updated_at": _now()},
        )

    def schedule_retry(self, source: str, error: str, attempts: int, backoff_s: int) -> None:
        """Move a failed file to ``retry`` with a scheduled ``next_retry_at``."""
        from datetime import timedelta

        self._ensure_collection()
        next_retry = (datetime.now(UTC) + timedelta(seconds=backoff_s)).isoformat()
        self._transition(
            source,
            IngestionStatus.RETRY,
            payload={
                "stage": PipelineStage.ERROR,
                "error": error,
                "attempts": attempts,
                "next_retry_at": next_retry,
                "updated_at": _now(),
            },
        )

    def reset_for_retry(self, source: str) -> None:
        """Manually reset a failed file back to processing (bypasses backoff)."""
        self._ensure_collection()
        self._transition(
            source,
            IngestionStatus.PROCESSING,
            payload={"stage": PipelineStage.CONVERTING, "next_retry_at": "", "updated_at": _now()},
        )

    # ── Queries ─────────────────────────────────────────────────────────────

    def get_status(self, source: str) -> dict[str, Any] | None:
        """Return the status record for a source, or None if absent."""
        self._ensure_collection()
        try:
            points, _ = self._qdrant.scroll(
                collection_name=self._collection,
                limit=1,
                with_vectors=False,
                scroll_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]),
            )
        except Exception as exc:
            raise ServiceUnavailableError("Qdrant", f"status lookup failed: {exc}") from exc
        if not points:
            return None
        payload = dict(points[0].payload or {})
        payload["source"] = source
        return payload

    def get_summary(self) -> dict[str, int]:
        """Count files by status (one aggregate query)."""
        self._ensure_collection()
        try:
            points, _ = self._qdrant.scroll(
                collection_name=self._collection,
                limit=100_000,
                with_vectors=False,
                with_payload=["status"],
            )
        except Exception as exc:
            raise ServiceUnavailableError("Qdrant", f"status summary failed: {exc}") from exc
        summary = {s: 0 for s in VALID_TRANSITIONS}
        for p in points:
            status = (p.payload or {}).get("status", IngestionStatus.PENDING)
            if status in summary:
                summary[status] += 1
        return summary

    def list_records(self, status_filter: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        """Return per-file status records, optionally filtered by status."""
        self._ensure_collection()
        try:
            if status_filter:
                qfilter = Filter(must=[FieldCondition(key="status", match=MatchValue(value=status_filter))])
            else:
                qfilter = None
            points, _ = self._qdrant.scroll(
                collection_name=self._collection,
                limit=limit,
                with_vectors=False,
                scroll_filter=qfilter,
            )
        except Exception as exc:
            raise ServiceUnavailableError("Qdrant", f"status list failed: {exc}") from exc
        records: list[dict[str, Any]] = []
        for p in points:
            payload = dict(p.payload or {})
            payload["source"] = payload.get("source", "")
            records.append(payload)
        records.sort(key=lambda r: r.get("updated_at", ""), reverse=True)
        return records

    def get_due_retries(self, now: datetime | None = None) -> list[str]:
        """Return sources with status=retry whose ``next_retry_at`` has passed."""
        self._ensure_collection()
        now = now or datetime.now(UTC)
        due: list[str] = []
        for record in self.list_records(status_filter=IngestionStatus.RETRY, limit=100_000):
            nxt = record.get("next_retry_at", "")
            if not nxt:
                continue
            try:
                if datetime.fromisoformat(nxt) <= now:
                    due.append(record["source"])
            except (ValueError, TypeError):
                continue
        return due

    def cleanup_stale(self, stale_after_s: int = 7 * 86400) -> int:
        """Mark long-unfinished ``processing`` records as failed (zombie recovery)."""
        self._ensure_collection()
        now = datetime.now(UTC)
        cleaned = 0
        for record in self.list_records(status_filter=IngestionStatus.PROCESSING, limit=100_000):
            updated = record.get("updated_at", "")
            if not updated:
                continue
            try:
                updated_dt = datetime.fromisoformat(updated)
            except (ValueError, TypeError):
                continue
            if (now - updated_dt).total_seconds() > stale_after_s:
                source = record["source"]
                self.mark_failed(
                    source,
                    "process likely died — status stale >7d",
                    stage=PipelineStage.ERROR,
                )
                cleaned += 1
        if cleaned:
            logger.warning("Marked %d stale processing records as failed", cleaned)
        return cleaned

    # ── Internals ───────────────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """Create the status collection if missing, with a payload index on status."""
        try:
            if not self._qdrant.collection_exists(self._collection):
                self._qdrant.create_collection(
                    collection_name=self._collection,
                    vectors_config={},
                )
                self._qdrant.create_payload_index(
                    collection_name=self._collection,
                    field_name="status",
                    field_schema="keyword",
                )
                logger.info("Created status collection: %s", self._collection)
        except Exception as exc:
            raise ServiceUnavailableError("Qdrant", f"status collection setup failed: {exc}") from exc

    def _transition(self, source: str, to_status: str, payload: dict[str, Any]) -> None:
        """Validate the status transition against the state machine, then persist."""
        current = self.get_status(source)
        from_status = (current or {}).get("status", IngestionStatus.PENDING)

        allowed = VALID_TRANSITIONS.get(from_status, set())
        if to_status not in allowed:
            raise StorageError(
                f"illegal status transition {from_status} → {to_status} for {source}",
                component="ingestion",
            )

        # Preserve existing record fields (source_name, created_at, attempts,
        # error history) so a stage update never wipes them.
        base = {k: v for k, v in (current or {}).items() if k != "status"}
        merged = {**base, "status": to_status, "updated_at": _now(), **payload}
        merged.setdefault("source", source)
        self._upsert(source, merged)

    def _upsert(self, source: str, payload: dict[str, Any]) -> None:
        from qdrant_client.models import PointStruct

        payload = {**payload, "source": source, "updated_at": payload.get("updated_at", _now())}
        try:
            self._qdrant.upsert(
                collection_name=self._collection,
                points=[PointStruct(id=_point_id(source), payload=payload, vector={})],
            )
        except Exception as exc:
            raise ServiceUnavailableError("Qdrant", f"status upsert failed for {source}: {exc}") from exc


__all__ = ["COLLECTION", "VALID_TRANSITIONS", "FileStatusStore", "IngestionStatus"]
