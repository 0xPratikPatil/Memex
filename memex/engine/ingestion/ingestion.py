"""IngestionOrchestrator — concurrent parsing, checkpointing, and timeout management.

Coordinates multi-file ingestion with:
- Concurrent document conversion via asyncio + thread pool (marker/markitdown/docling)
- Per-document timeouts with graceful skip
- Batch checkpointing for crash-resume
- Pre-checks to skip unchanged files
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from memex.engine.core import config
from memex.engine.core.pipeline import RAGEngine
from memex.engine.core.progress import PipelineStage
from memex.engine.ingestion.hashing import (
    clear_source_chunks,
    compute_content_hash,
    is_already_ingested,
)
from memex.engine.ingestion.status import FileStatusStore

logger = logging.getLogger("ingestion")


def _data_dir() -> Path:
    """Resolve the memex data directory for batch state files."""
    env_dir = os.getenv("MEMEX_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    home = os.getenv("HOME")
    if home:
        return Path(home) / ".memex"
    return Path("/tmp/memex")


def _batch_state_dir() -> Path:
    p = _data_dir() / "batches"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _batch_id(items: list[str]) -> str:
    """Generate a deterministic batch ID from the sorted item list."""
    sorted_items = sorted(items)
    return hashlib.sha256(json.dumps(sorted_items).encode()).hexdigest()[:12]


class IngestionOrchestrator:
    """Coordinates concurrent document ingestion with checkpointing.

    Usage::

        orchestrator = IngestionOrchestrator(engine)
        results = await orchestrator.ingest_batch(["/docs/a.pdf", "/docs/b.pdf"])
    """

    def __init__(self, engine: RAGEngine) -> None:
        self._engine = engine
        self._parse_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_PARSES)

    # ── Public API ────────────────────────────────────────────────────────

    async def ingest_single(self, item: str) -> str:
        """Ingest a single file or URL with pre-check and timeout.

        Returns a status message string.
        """
        from memex.engine.ingestion.loader import parse_file

        status_store = FileStatusStore(self._engine._get_qdrant())
        status_store.mark_pending(item)

        # Pre-check: skip if local file unchanged
        can_skip, chunk_count = self._engine.check_unmodified_local(item)
        if can_skip:
            status_store.mark_skipped(item, reason="unchanged")
            return f"Skipped ({chunk_count} chunks, unchanged)"

        # Parse with timeout
        try:
            result = await self._parse_with_timeout(item, parse_file)
        except TimeoutError:
            err = f"parse timeout exceeded {config.INGEST_TIMEOUT_PARSE}s"
            status_store.mark_failed(item, err, stage=PipelineStage.CONVERTING)
            return f"Failed: {err}"
        except Exception as exc:
            status_store.mark_failed(item, str(exc), exc=exc)
            return f"Failed: {exc}"

        if not result.ok:  # type: ignore[union-attr]
            err = f"{config.CONVERTER_ENGINE.title()} status '{result.status}', errors: {result.errors}"  # type: ignore[union-attr]
            status_store.mark_failed(item, err, stage=PipelineStage.CONVERTING)
            return f"Failed: {err}"

        content_hash = compute_content_hash(result.markdown.encode())  # type: ignore[union-attr]

        # Content-hash dedup: skip if identical content already ingested
        qdrant = self._engine._get_qdrant()
        already, existing = await is_already_ingested(qdrant, config.COLLECTION_NAME, item, content_hash)
        if already:
            status_store.mark_skipped(item, reason="dedup")
            return f"Skipped ({existing} chunks, unchanged)"

        # Check for partial prior ingest: if some chunks exist but content
        # changed (not caught by dedup above), clear stale partial data.
        deleted = await clear_source_chunks(qdrant, config.COLLECTION_NAME, item)
        if deleted > 0:
            logger.info("Cleared %d stale/partial chunks for %s before re-ingest", deleted, item)

        try:
            count = await asyncio.wait_for(
                asyncio.to_thread(
                    self._engine.ingest_text,
                    result.markdown,
                    source_identifier=item,
                    metadata={
                        "content_type": item.rsplit(".", 1)[-1] if "." in item else "",
                        "content_hash": content_hash,
                    },
                    content_hash=content_hash,
                ),
                timeout=config.INGEST_TIMEOUT_TOTAL,
            )
            status_store.mark_done(item, chunks=count)
            return f"Success ({count} chunks, {result.processing_time:.1f}s conversion)"
        except TimeoutError:
            err = f"total timeout exceeded {config.INGEST_TIMEOUT_TOTAL}s"
            status_store.mark_failed(item, err, stage=PipelineStage.STORING)
            return f"Failed: {err}"
        except Exception as exc:
            status_store.mark_failed(item, str(exc), exc=exc)
            return f"Failed: {exc}"

    async def ingest_batch(self, items: list[str]) -> dict[str, str]:
        """Concurrently ingest multiple files with checkpointing.

        On resume, skips completed items. Failures are per-item — partial
        results are returned.
        """
        if not items:
            return {}

        bid = _batch_id(items)
        state_dir = _batch_state_dir()
        state_file = state_dir / f"{bid}.json"
        state = self._load_state(state_file, items)

        # Determine which items still need processing
        completed_set = set(state["completed"])
        failed_map: dict[str, str] = dict(state["failed"])
        pending = [item for item in items if item not in completed_set and item not in failed_map]

        if completed_set:
            logger.info(
                "Resuming batch %s — %d completed, %d failed, %d remaining",
                bid,
                len(completed_set),
                len(failed_map),
                len(pending),
            )
        else:
            logger.info("Starting batch %s — %d items", bid, len(pending))

        # Concurrent parse phase
        parse_results: dict[str, object] = {}
        parse_tasks = [self._parse_one(item, parse_results) for item in pending]
        await asyncio.gather(*parse_tasks, return_exceptions=True)

        # Concurrent ingest phase with bounded concurrency
        # Each ingest includes chunking, metadata extraction, embedding, and Qdrant upsert.
        # Using a semaphore to limit concurrent ingests prevents Ollama overload.
        ingest_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_PARSES)

        async def _ingest_one(item: str) -> None:
            result = parse_results.get(item)
            if isinstance(result, Exception):
                failed_map[item] = str(result)
                return
            if result is None:
                failed_map[item] = "parse returned None"
                return
            async with ingest_semaphore:
                try:
                    status = await self._ingest_parsed(item, result)
                    if status.startswith("Success") or status.startswith("Skipped"):
                        state["completed"].append(item)
                    else:
                        failed_map[item] = status
                except Exception as exc:
                    failed_map[item] = str(exc)
                    try:
                        FileStatusStore(self._engine._get_qdrant()).mark_failed(item, str(exc), exc=exc)
                    except Exception:
                        logger.debug("Status mark failed for %s", item, exc_info=True)

        ingest_tasks = [_ingest_one(item) for item in pending]
        await asyncio.gather(*ingest_tasks, return_exceptions=True)

        # Save final state once (not per-item)
        self._save_state(state_file, state, failed_map)

        summary: dict[str, str] = {}
        for item in items:
            if item in state["completed"]:
                summary[item] = "Success (checkpointed)"
            elif item in failed_map:
                summary[item] = f"Failed: {failed_map[item]}"
            else:
                summary[item] = "Unknown state"

        state["completed_at"] = datetime.now(UTC).isoformat()
        self._save_state(state_file, state, failed_map)

        # Clean old state files
        self._cleanup_old_batches(state_dir)

        return summary

    # ── Private helpers ───────────────────────────────────────────────────

    async def _parse_one(self, item: str, results: dict[str, object]) -> None:
        """Parse one file concurrently with semaphore and timeout."""
        from memex.engine.ingestion.loader import parse_file

        # Pre-check
        can_skip, _ = self._engine.check_unmodified_local(item)
        if can_skip:
            results[item] = "skip"
            return

        async with self._parse_semaphore:
            try:
                loop = asyncio.get_running_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, parse_file, item),
                    timeout=config.INGEST_TIMEOUT_PARSE,
                )
                results[item] = result
            except TimeoutError:
                results[item] = TimeoutError(f"parse timeout exceeded {config.INGEST_TIMEOUT_PARSE}s")
            except Exception as exc:
                results[item] = exc

    async def _parse_with_timeout(self, item: str, parse_fn) -> object:
        """Parse a single item with a timeout."""
        loop = asyncio.get_running_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, parse_fn, item),
            timeout=config.INGEST_TIMEOUT_PARSE,
        )

    async def _ingest_parsed(self, item: str, result) -> str:
        """Ingest an already-parsed ConversionResult with total timeout."""
        from memex.engine.ingestion.loader import ConversionResult

        status_store = FileStatusStore(self._engine._get_qdrant())
        status_store.mark_pending(item)

        if isinstance(result, str) and result == "skip":
            _exists, chunk_count, _, _ = self._engine.source_exists(item)
            status_store.mark_skipped(item, reason="unchanged")
            return f"Skipped ({chunk_count} chunks, unchanged)"

        if not isinstance(result, ConversionResult) or not result.ok:
            err = getattr(result, "errors", []) if hasattr(result, "errors") else []
            status = getattr(result, "status", "unknown")
            status_store.mark_failed(
                item,
                f"{config.CONVERTER_ENGINE.title()} status '{status}', errors: {err}",
                stage=PipelineStage.CONVERTING,
            )
            return f"Failed: {config.CONVERTER_ENGINE.title()} status '{status}', errors: {err}"

        content_hash = compute_content_hash(result.markdown.encode())

        # Content-hash dedup: skip if identical content already ingested
        qdrant = self._engine._get_qdrant()
        already, existing = await is_already_ingested(qdrant, config.COLLECTION_NAME, item, content_hash)
        if already:
            status_store.mark_skipped(item, reason="dedup")
            return f"Skipped ({existing} chunks, unchanged)"

        # Clear stale/partial chunks before re-ingest
        deleted = await clear_source_chunks(qdrant, config.COLLECTION_NAME, item)
        if deleted > 0:
            logger.info("Cleared %d stale/partial chunks for %s before re-ingest", deleted, item)

        try:
            count = await asyncio.wait_for(
                asyncio.to_thread(
                    self._engine.ingest_text,
                    result.markdown,
                    source_identifier=item,
                    metadata={
                        "content_type": item.rsplit(".", 1)[-1] if "." in item else "",
                        "content_hash": content_hash,
                    },
                    content_hash=content_hash,
                ),
                timeout=config.INGEST_TIMEOUT_TOTAL,
            )
            status_store.mark_done(item, chunks=count)
            return f"Success ({count} chunks, {result.processing_time:.1f}s conversion)"
        except TimeoutError:
            err = f"total timeout exceeded {config.INGEST_TIMEOUT_TOTAL}s"
            status_store.mark_failed(item, err, stage=PipelineStage.STORING)
            return f"Failed: {err}"
        except Exception as exc:
            status_store.mark_failed(item, str(exc), exc=exc)
            return f"Failed: {exc}"

    # ── State file management ─────────────────────────────────────────────

    @staticmethod
    def _load_state(state_file: Path, items: list[str]) -> dict:
        """Load existing state or create a fresh one."""
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text())
                if state.get("items") and sorted(state["items"]) == sorted(items):
                    return state
            except (json.JSONDecodeError, KeyError):
                logger.warning("Corrupt batch state file, starting fresh: %s", state_file)

        return {
            "batch_id": _batch_id(items),
            "created_at": datetime.now(UTC).isoformat(),
            "completed_at": None,
            "items": items,
            "completed": [],
            "failed": {},
        }

    @staticmethod
    def _save_state(state_file: Path, state: dict, failed_map: dict[str, str]) -> None:
        """Flush state to disk atomically via tmp file + rename."""
        state["failed"] = failed_map
        tmp = state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(state_file)

    @staticmethod
    def _cleanup_old_batches(state_dir: Path) -> None:
        """Remove completed batches older than 7 days, incomplete older than 30."""
        now = time.time()
        for f in state_dir.glob("*.json"):
            try:
                state = json.loads(f.read_text())
                created = state.get("created_at", "")
                if not created:
                    continue
                created_ts = datetime.fromisoformat(created).timestamp()
                age = now - created_ts
                if state.get("completed_at") and age > 7 * 86400:
                    f.unlink(missing_ok=True)
                    logger.debug("Cleaned completed batch state: %s", f.name)
                elif not state.get("completed_at") and age > 30 * 86400:
                    logger.warning("Cleaning incomplete batch state older than 30 days: %s", f.name)
                    f.unlink(missing_ok=True)
            except (json.JSONDecodeError, ValueError, OSError):
                # Corrupt or unreadable — clean after 7 days of mtime
                try:
                    if now - f.stat().st_mtime > 7 * 86400:
                        f.unlink(missing_ok=True)
                except OSError:
                    pass
