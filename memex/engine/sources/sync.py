"""Sync engine — reconcile sources against the vector collection.

Parallel processing with bounded concurrency, progress callbacks through
the entire pipeline, and single-Docling-call hybrid chunking.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

from memex.engine.core import config
from memex.engine.core.progress import FileProgress, ProgressCallback
from memex.engine.sources import Source, SourceFile, get_source

log = logging.getLogger(__name__)


# ── Thread-safe counter for concurrent progress tracking ──────────────────────


class _AtomicCounter:
    """Thread-safe counter for tracking completed files across concurrent tasks."""

    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


# ── Path normalization ────────────────────────────────────────────────────────


def _path_forms(path: str) -> list[str]:
    import contextlib

    candidates = {path}
    with contextlib.suppress(Exception):
        candidates.add(str(Path(path).resolve()))
    with contextlib.suppress(Exception):
        candidates.add(str(Path(path).absolute()))
    candidates.add(path.replace("\\", "/"))
    candidates.add(path.replace("/", "\\"))
    return list(candidates)


# ── Stats ─────────────────────────────────────────────────────────────────────


@dataclass
class SyncStats:
    added: int = 0
    changed: int = 0
    deleted: int = 0
    unchanged: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"added={self.added} changed={self.changed} "
            f"deleted={self.deleted} unchanged={self.unchanged} "
            f"errors={len(self.errors)}"
        )


# ── Stored hash query ─────────────────────────────────────────────────────────


def _get_stored_hashes(engine, source_name: str) -> dict[str, str]:
    """Query Qdrant for all (source_path -> content_hash) pairs for a source."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    qdrant = engine._get_qdrant()
    stored: dict[str, str] = {}
    offset = None
    while True:
        result = qdrant.scroll(
            collection_name=config.COLLECTION_NAME,
            limit=500,
            offset=offset,
            scroll_filter=Filter(must=[FieldCondition(key="source_name", match=MatchValue(value=source_name))]),
            with_payload=["source", "source_name", "content_hash"],
            with_vectors=False,
        )
        points, next_offset = result
        for point in points:
            payload = point.payload or {}
            src = payload.get("source", "")
            content_hash = payload.get("content_hash", "")
            if src and content_hash and src not in stored:
                stored[src] = content_hash
        if next_offset is None:
            break
        offset = next_offset
    return stored


# ── Single-file ingest (hybrid-aware, with progress) ─────────────────────────


def _ingest_file(
    engine,
    source_identifier: str,
    file_path: str,
    source_name: str,
    progress_cb: ProgressCallback | None = None,
    file_idx: int = 0,
    total_files: int = 0,
) -> int:
    """Convert and ingest a single file. Returns chunk count.

    Stages surfaced via progress_cb:
      - "Converting" (70%) — Docling conversion / chunking
      - "Context" (73%) — Context enrichment per chunk
      - "Metadata" (74%) — Entity/topic extraction
      - "Embedding" (75-89%) — Dense + sparse embedding generation
      - "Storing" (90%) — Qdrant upsert (immediate per-file)

    Args:
        source_identifier: Logical path stored in Qdrant (real local path or S3 key).
        file_path: Where the file actually lives for reading/parsing.
        source_name: Source name for tracking/reconciliation.
        progress_cb: Callback for fine-grained pipeline stage updates.
        file_idx: Current file index for progress reporting.
        total_files: Total file count for progress reporting.
    """

    def _emit_pipeline(stage: str, msg: str, pct: int) -> None:
        """Emit a pipeline sub-stage progress event."""
        if progress_cb is not None:
            progress_cb(
                FileProgress(
                    path=source_identifier,
                    total=total_files,
                    current=file_idx,
                    stage=stage,
                    chunks=0,
                )
            )

    strategy = config.CHUNK_STRATEGY.lower()

    if strategy == "hybrid":
        from memex.engine.ingestion.splitter import chunk_file

        _emit_pipeline("Converting", "Converting + chunking (Docling)...", 70)
        try:
            result = chunk_file(file_path, include_doc=True)
            markdown = result.get("markdown", "")
            chunks = result.get("chunks", [])
        except Exception as exc:
            log.warning("Hybrid chunking failed for %s, falling back to parse+chunk", file_path, exc_info=True)
            from memex.engine.ingestion.loader import parse_file

            parse_result = parse_file(file_path)
            if not parse_result.ok:
                raise RuntimeError(
                    f"Docling conversion failed for {file_path}: {parse_result.status} -- {parse_result.errors}"
                ) from exc
            markdown = parse_result.markdown
            chunks = None

        if not markdown:
            from memex.engine.ingestion.loader import parse_file

            parse_result = parse_file(file_path)
            if not parse_result.ok:
                raise RuntimeError(
                    f"Docling conversion failed for {file_path}: {parse_result.status} -- {parse_result.errors}"
                )
            markdown = parse_result.markdown
    else:
        from memex.engine.ingestion.loader import parse_file

        _emit_pipeline("Converting", "Converting (Docling)...", 70)
        parse_result = parse_file(file_path)
        if not parse_result.ok:
            raise RuntimeError(
                f"Docling conversion failed for {file_path}: {parse_result.status} -- {parse_result.errors}"
            )
        markdown = parse_result.markdown
        chunks = None

    content_hash = engine.compute_file_hash(markdown.encode())

    def _pipeline_progress(msg: str, pct: int) -> None:
        """Forward pipeline progress to sync-level callback."""
        if progress_cb is not None:
            # Map pipeline pct to stage name for display
            if pct <= 72:
                stage = "Converting"
            elif pct <= 74:
                stage = "Context"
            elif pct <= 76:
                stage = "Metadata"
            elif pct <= 89:
                stage = "Embedding"
            else:
                stage = "Storing"
            progress_cb(
                FileProgress(
                    path=source_identifier,
                    total=total_files,
                    current=file_idx,
                    stage=stage,
                    chunks=0,
                )
            )

    if chunks is not None:
        count = engine.ingest_prechunked(
            chunks=chunks,
            markdown=markdown,
            source_identifier=source_identifier,
            metadata={
                "content_type": file_path.rsplit(".", 1)[-1] if "." in file_path else "",
                "content_hash": content_hash,
                "source_name": source_name,
            },
            content_hash=content_hash,
            progress_cb=_pipeline_progress,
        )
    else:
        count = engine.ingest_text(
            markdown,
            source_identifier=source_identifier,
            metadata={
                "content_type": file_path.rsplit(".", 1)[-1] if "." in file_path else "",
                "content_hash": content_hash,
                "source_name": source_name,
            },
            content_hash=content_hash,
            progress_cb=_pipeline_progress,
        )
    return count


# ── Main sync function (parallel) ─────────────────────────────────────────────


async def sync(
    config_module,
    source_name: str | None = None,
    dry_run: bool = False,
    progress_cb: ProgressCallback | None = None,
) -> SyncStats:
    """Sync collection against configured sources.

    1. Load sources from config
    2. For each source: list_files()
    3. For each file: compute content hash
    4. Query Qdrant for stored hashes
    5. Reconcile: new -> ingest, changed -> delete+ingest, deleted -> delete
    6. Safety: if any source fails to list, suppress deletions
    7. Return stats

    Files are processed concurrently with bounded parallelism (MAX_CONCURRENT_SYNC).
    Each file's embeddings are stored to Qdrant immediately upon completion.
    """
    from memex.engine.core.pipeline import RAGEngine
    from memex.engine.sources.status_tracker import StatusTracker

    stats = SyncStats()
    suppress_deletions = False
    completed = _AtomicCounter()

    # 1. Load sources from config — try YAML first, then module attribute
    if hasattr(config_module, "get_list"):
        source_configs = config_module.get_list("sources", [])
    elif hasattr(config_module, "_yaml") and config_module._yaml is not None:
        source_configs = config_module._yaml.get_list("sources", [])
    else:
        source_configs = []
    if source_name:
        source_configs = [s for s in source_configs if s.get("name") == source_name]

    if not source_configs:
        log.warning("No sources configured for sync")
        return stats

    engine = RAGEngine()
    engine._get_qdrant()  # ensure collection exists
    
    status_tracker = StatusTracker(qdrant_client=engine._qdrant, collection=config.COLLECTION_NAME)

    max_concurrent = config.MAX_CONCURRENT_SYNC
    semaphore = asyncio.Semaphore(max_concurrent)

    # 2-3. List files from each source
    all_source_files: dict[str, list[SourceFile]] = {}
    for src_cfg in source_configs:
        src_type = src_cfg.get("type", "local")
        src_name = src_cfg.get("name", src_type)
        try:
            source = get_source(src_type, src_cfg)
            files = source.list_files()
            all_source_files[src_name] = files
            log.info("Source '%s' listed %d files", src_name, len(files))
        except Exception as exc:
            log.error("Failed to list source '%s': %s", src_name, exc)
            stats.errors.append(f"source '{src_name}' listing failed: {exc}")
            suppress_deletions = True

    def _emit(path: str, stage: str, idx: int, total: int, chunks: int = 0, error: str = "") -> None:
        if progress_cb is not None:
            progress_cb(
                FileProgress(
                    path=path,
                    total=total,
                    current=completed.value,
                    stage=stage,
                    chunks=chunks,
                    error=error,
                )
            )

    # 4. Get stored hashes for each source and reconcile
    for src_cfg in source_configs:
        src_type = src_cfg.get("type", "local")
        src_name = src_cfg.get("name", src_type)
        current_files = all_source_files.get(src_name, [])
        source = get_source(src_type, src_cfg)

        download_dir: Path = getattr(source, "cache_dir", None) or Path.cwd()

        current_map: dict[str, SourceFile] = {f.path: f for f in current_files}

        _current_form_map: dict[str, str] = {}
        for fp in current_map:
            for form in _path_forms(fp):
                _current_form_map[form] = fp

        stored_hashes = await asyncio.to_thread(_get_stored_hashes, engine, src_name)

        resolved_stored: dict[str, str] = {}
        for sp, h in stored_hashes.items():
            if sp in current_map:
                resolved_stored[sp] = h
            else:
                canonical = _current_form_map.get(sp)
                if canonical:
                    resolved_stored[canonical] = h
                else:
                    resolved_stored[sp] = h
        stored_hashes = resolved_stored

        stored_paths = set(stored_hashes.keys())
        current_paths = set(current_map.keys())

        new_paths = current_paths - stored_paths
        common_paths = current_paths & stored_paths
        deleted_paths = stored_paths - current_paths

        total_files = len(new_paths) + len(common_paths) + len(deleted_paths)

        # ── Phase 3: Concurrent file processing ──────────────────────────
        tasks: list[asyncio.Task] = []
        file_idx_counter = 0

        async def _process_new(
            path: str,
            sf: SourceFile,
            idx: int,
            total: int,
            _source: Source = source,
            _download_dir: Path = download_dir,
            _src_name: str = src_name,
        ) -> tuple[str, str] | tuple[str, str, str]:
            async with semaphore:
                if dry_run:
                    log.info("[dry-run] Would add: %s", path)
                    _emit(path, "Done", idx, total)
                    completed.increment()
                    return ("added", path)
                try:
                    _emit(path, "Converting", idx, total)
                    status_tracker.update_status(path, "converting")
                    local = await asyncio.to_thread(_source.download, sf, _download_dir)
                    chunk_count = await asyncio.to_thread(
                        _ingest_file, engine, sf.path, str(local), _src_name, progress_cb, idx, total
                    )
                    _emit(path, "Done", idx, total, chunks=chunk_count)
                    status_tracker.update_status(path, "done")
                    completed.increment()
                    log.info("Added '%s' (%d chunks)", path, chunk_count)
                    return ("added", path)
                except Exception as exc:
                    log.error("Failed to ingest '%s': %s", path, exc)
                    _emit(path, "Error", idx, total, error=str(exc))
                    status_tracker.update_status(path, "error", error=str(exc))
                    completed.increment()
                    return ("error", path, str(exc))

        async def _process_changed(
            path: str,
            sf: SourceFile,
            idx: int,
            total: int,
            _source: Source = source,
            _download_dir: Path = download_dir,
            _src_name: str = src_name,
        ) -> tuple[str, str] | tuple[str, str, str]:
            async with semaphore:
                try:
                    _emit(path, "Deleting", idx, total)
                    status_tracker.update_status(path, "deleting")
                    await asyncio.to_thread(engine.delete_by_source, path)
                    _emit(path, "Converting", idx, total)
                    status_tracker.update_status(path, "converting")
                    local = await asyncio.to_thread(_source.download, sf, _download_dir)
                    chunk_count = await asyncio.to_thread(
                        _ingest_file, engine, sf.path, str(local), _src_name, progress_cb, idx, total
                    )
                    _emit(path, "Done", idx, total, chunks=chunk_count)
                    status_tracker.update_status(path, "done")
                    completed.increment()
                    log.info("Updated '%s' (%d chunks)", path, chunk_count)
                    return ("changed", path)
                except Exception as exc:
                    log.error("Failed to update '%s': %s", path, exc)
                    _emit(path, "Error", idx, total, error=str(exc))
                    status_tracker.update_status(path, "error", error=str(exc))
                    completed.increment()
                    return ("error", path, str(exc))

        async def _process_deleted(path: str, idx: int, total: int) -> tuple[str, str] | tuple[str, str, str]:
            async with semaphore:
                if suppress_deletions:
                    log.warning(
                        "Suppressing deletion for '%s' (source listing failed or empty)",
                        path,
                    )
                    _emit(path, "Done", idx, total)
                    completed.increment()
                    return ("unchanged", path)
                if dry_run:
                    log.info("[dry-run] Would delete: %s", path)
                    _emit(path, "Done", idx, total)
                    completed.increment()
                    return ("deleted", path)
                try:
                    _emit(path, "Deleting", idx, total)
                    status_tracker.update_status(path, "deleting")
                    await asyncio.to_thread(engine.delete_by_source, path)
                    status_tracker.update_status(path, "deleted")
                    _emit(path, "Done", idx, total)
                    completed.increment()
                    log.info("Deleted '%s'", path)
                    return ("deleted", path)
                except Exception as exc:
                    log.error("Failed to delete '%s': %s", path, exc)
                    _emit(path, "Error", idx, total, error=str(exc))
                    status_tracker.update_status(path, "error", error=str(exc))
                    completed.increment()
                    return ("error", path, str(exc))

        async def _process_unchanged(
            path: str,
            sf: SourceFile,
            idx: int,
            total: int,
            _source: Source = source,
            _download_dir: Path = download_dir,
            _src_name: str = src_name,
            _stored_hashes: dict[str, str] = stored_hashes,
        ) -> tuple[str, str] | tuple[str, str, str]:
            async with semaphore:
                try:
                    current_hash = await asyncio.to_thread(_source.get_content_hash, sf)
                except Exception as exc:
                    log.error("Failed to hash '%s': %s", path, exc)
                    _emit(path, "Error", idx, total, error=str(exc))
                    status_tracker.update_status(path, "error", error=str(exc))
                    completed.increment()
                    return ("error", path, str(exc))

                stored_hash = _stored_hashes.get(path, "")
                if current_hash == stored_hash:
                    _emit(path, "Done", idx, total)
                    status_tracker.update_status(path, "done")
                    completed.increment()
                    return ("unchanged", path)

                if dry_run:
                    log.info("[dry-run] Would update: %s", path)
                    _emit(path, "Done", idx, total)
                    completed.increment()
                    return ("changed", path)

                # Changed file
                try:
                    _emit(path, "Deleting", idx, total)
                    status_tracker.update_status(path, "deleting")
                    await asyncio.to_thread(engine.delete_by_source, path)
                    _emit(path, "Converting", idx, total)
                    status_tracker.update_status(path, "converting")
                    local = await asyncio.to_thread(_source.download, sf, _download_dir)
                    chunk_count = await asyncio.to_thread(
                        _ingest_file, engine, sf.path, str(local), _src_name, progress_cb, idx, total
                    )
                    _emit(path, "Done", idx, total, chunks=chunk_count)
                    status_tracker.update_status(path, "done")
                    completed.increment()
                    log.info("Updated '%s' (%d chunks)", path, chunk_count)
                    return ("changed", path)
                except Exception as exc:
                    log.error("Failed to update '%s': %s", path, exc)
                    _emit(path, "Error", idx, total, error=str(exc))
                    status_tracker.update_status(path, "error", error=str(exc))
                    completed.increment()
                    return ("error", path, str(exc))

        # Build task list
        for path in new_paths:
            file_idx_counter += 1
            tasks.append(asyncio.create_task(_process_new(path, current_map[path], file_idx_counter, total_files)))

        for path in common_paths:
            file_idx_counter += 1
            tasks.append(
                asyncio.create_task(_process_unchanged(path, current_map[path], file_idx_counter, total_files))
            )

        for path in deleted_paths:
            file_idx_counter += 1
            tasks.append(asyncio.create_task(_process_deleted(path, file_idx_counter, total_files)))

        # Run all tasks concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in results:
                if isinstance(r, Exception):
                    stats.errors.append(str(r))
                elif isinstance(r, tuple):
                    tag = r[0]
                    if tag == "added":
                        stats.added += 1
                    elif tag == "changed":
                        stats.changed += 1
                    elif tag == "deleted":
                        stats.deleted += 1
                    elif tag == "unchanged":
                        stats.unchanged += 1
                    elif tag == "error":
                        stats.errors.append(r[2])  # type: ignore[index]

        if deleted_paths and suppress_deletions:
            log.warning(
                "Suppressing %d deletions for source '%s' (listing failed or empty)",
                len(deleted_paths),
                src_name,
            )

    log.info("Sync complete: %s", stats.summary())
    return stats
