"""Sync engine — reconcile sources against the vector collection.

Parallel processing with bounded concurrency, progress callbacks through
the entire pipeline, and multi-engine conversion (marker/markitdown/docling).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from memex.engine.core import config
from memex.engine.core.errors import (
    ConversionTimeoutError,
    IngestionError,
    MemexError,
)
from memex.engine.core.progress import FileProgress, PipelineStage, ProgressCallback
from memex.engine.ingestion.status import FileStatusStore
from memex.engine.sources import Source, SourceFile, get_source

log = logging.getLogger(__name__)


def _log_file_error(message: str, source: str, exc: BaseException, *, stage: str) -> None:
    """Log a per-file error with structured extras for observability."""
    extra = {"source": source, "stage": stage}
    if isinstance(exc, MemexError) and exc.hint:
        extra["hint"] = exc.hint
    log.error(message, source, exc, extra=extra)


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
    skipped: int = 0

    def summary(self) -> str:
        return (
            f"added={self.added} changed={self.changed} "
            f"deleted={self.deleted} unchanged={self.unchanged} "
            f"skipped={self.skipped} errors={len(self.errors)}"
        )


@dataclass
class _SkipResult:
    """Sentinel result for skipped files (e.g., timeout with skip_on_timeout=True)."""

    file_path: str

    def __bool__(self) -> bool:
        return False


# ── Stored hash query ─────────────────────────────────────────────────────────


def _get_stored_hashes(engine, source_name: str) -> dict[str, str]:
    """Query Qdrant for all (source_path -> raw_file_hash) pairs for a source.

    Prefers ``file_content_hash`` (raw file bytes) which is what reconciliation
    compares against ``Source.get_content_hash()``. Falls back to the legacy
    ``content_hash`` (markdown) for records ingested before the field existed.
    """
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
            with_payload=["source", "source_name", "content_hash", "file_content_hash"],
            with_vectors=False,
        )
        points, next_offset = result
        for point in points:
            payload = point.payload or {}
            src = payload.get("source", "")
            file_hash = payload.get("file_content_hash") or payload.get("content_hash", "")
            if src and file_hash and src not in stored:
                stored[src] = file_hash
        if next_offset is None:
            break
        offset = next_offset
    return stored


# ── Single-file pipeline: convert → (OCR) → ingest ──────────────────────────


def _convert_file(
    engine,
    file_path: str,
    source_identifier: str,
    progress_cb: ProgressCallback | None = None,
    file_idx: int = 0,
    total_files: int = 0,
) -> dict:
    """Convert a single file WITHOUT ingesting. OCR runs inline if needed.

    Returns:
        {"markdown": str, "chunks": list | None}
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

        _emit_pipeline("Converting", f"Converting + chunking ({config.CONVERTER_ENGINE.title()})...", 70)
        try:
            result = chunk_file(file_path, include_doc=True)
            markdown = result.get("markdown", "")
            chunks = result.get("chunks", [])
        except Exception as exc:
            if isinstance(exc, MemexError):
                log.warning(
                    "Hybrid chunking failed for %s (%s: %s), falling back to parse+chunk",
                    file_path,
                    type(exc).__name__,
                    exc,
                )
            else:
                log.warning("Hybrid chunking failed for %s, falling back to parse+chunk", file_path)
            log.debug("Hybrid chunking fallback detail for %s", file_path, exc_info=True)
            from memex.engine.ingestion.loader import parse_file

            parse_result = parse_file(file_path)
            if not parse_result.ok:
                raise IngestionError(
                    file_path,
                    f"conversion failed: {parse_result.status} -- {parse_result.errors}",
                    stage=PipelineStage.CONVERTING,
                    cause=exc,
                ) from exc
            markdown = parse_result.markdown
            chunks = None

        if not markdown:
            from memex.engine.ingestion.loader import parse_file

            parse_result = parse_file(file_path)
            if not parse_result.ok:
                raise IngestionError(
                    file_path,
                    f"conversion returned no content: {parse_result.status} -- {parse_result.errors}",
                    stage=PipelineStage.CONVERTING,
                )
            markdown = parse_result.markdown
    else:
        from memex.engine.ingestion.loader import parse_file

        _emit_pipeline("Converting", f"Converting ({config.CONVERTER_ENGINE.title()})...", 70)
        parse_result = parse_file(file_path)
        if not parse_result.ok:
            raise IngestionError(
                file_path,
                f"conversion failed: {parse_result.status} -- {parse_result.errors}",
                stage=PipelineStage.CONVERTING,
            )
        markdown = parse_result.markdown
        chunks = None

    return {"markdown": markdown, "chunks": chunks}


def _ingest_markdown(
    engine,
    markdown: str,
    chunks,
    source_identifier: str,
    file_path: str,
    source_name: str,
    progress_cb: ProgressCallback | None = None,
) -> int:
    """Hash + embed + store converted markdown. Returns chunk count."""
    content_hash = engine.compute_file_hash(markdown.encode())

    # Raw file bytes hash — the value reconciliation compares against
    # get_content_hash(). content_hash (markdown) differs from the raw file
    # hash, so storing only content_hash made every file look 'changed' on
    # every sync run (full re-ingest each time).
    import hashlib as _hashlib

    file_content_hash = ""
    try:
        h = _hashlib.sha256()
        with open(file_path, "rb") as f:
            for _chunk in iter(lambda: f.read(8192), b""):
                h.update(_chunk)
        file_content_hash = h.hexdigest()
    except OSError:
        log.warning("Could not hash raw file %s for reconciliation", file_path)

    metadata = {
        "content_type": file_path.rsplit(".", 1)[-1] if "." in file_path else "",
        "content_hash": content_hash,
        "file_content_hash": file_content_hash,
        "source_name": source_name,
    }

    if chunks is not None:
        return engine.ingest_prechunked(
            chunks=chunks,
            markdown=markdown,
            source_identifier=source_identifier,
            metadata=metadata,
            content_hash=content_hash,
            progress_cb=progress_cb,
        )
    return engine.ingest_text(
        markdown,
        source_identifier=source_identifier,
        metadata=metadata,
        content_hash=content_hash,
        progress_cb=progress_cb,
    )


def _ingest_file(
    engine,
    source_identifier: str,
    file_path: str,
    source_name: str,
    progress_cb: ProgressCallback | None = None,
    file_idx: int = 0,
    total_files: int = 0,
) -> int | _SkipResult:
    """Convert and ingest a single file (OCR inline). Returns chunk count.

    Kept for compatibility with single-file callers and tests. The sync
    engine itself uses the staged _convert_file / _ingest_markdown flow.
    """
    conv = _convert_file(engine, file_path, source_identifier, progress_cb, file_idx, total_files)
    return _ingest_markdown(
        engine, conv["markdown"], conv["chunks"], source_identifier, file_path, source_name, progress_cb
    )


def _int_setting(value, default: int, minimum: int = 0) -> int:
    """Coerce a config value to an int, falling back to default (test mocks)."""
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return default


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

    status_store = FileStatusStore(qdrant_client=engine._get_qdrant())

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

        # Fold in files scheduled for retry whose backoff window has passed.
        # They re-enter reconciliation as pending → processing.
        try:
            due_retries = status_store.get_due_retries()
            for rp in due_retries:
                if rp in current_map and rp not in new_paths:
                    log.info("Retrying '%s' (backoff elapsed)", rp)
                    new_paths.add(rp)
                    if rp in common_paths:
                        common_paths.discard(rp)
        except Exception as exc:
            log.warning("Retry fold-in failed: %s", exc)

        total_files = len(new_paths) + len(common_paths) + len(deleted_paths)

        # ── Phase 3: Concurrent file processing ──────────────────────────
        # Files are processed by a bounded synchronous thread pool.
        # MarkItDown and OCR are separate containers with their own FIFO
        # queues — a scanned PDF falls back to OCR inline, the converters
        # never block the pool.
        file_idx_counter = 0

        def _schedule_auto_retry(path: str, error: str) -> None:
            """Queue a failed file for automatic retry on the next sync run."""
            max_attempts = _int_setting(config.RETRY_MAX_ATTEMPTS, 5, 1)
            backoff_s = _int_setting(config.RETRY_BACKOFF_SECONDS, 300, 5)
            try:
                rec = status_store.get_status(path)
                attempts = int((rec or {}).get("attempts") or 0) + 1
                if attempts <= max_attempts:
                    status_store.schedule_retry(path, error, attempts=attempts, backoff_s=backoff_s)
                    log.info(
                        "Scheduled auto-retry %d/%d for '%s' (next sync)",
                        attempts,
                        max_attempts,
                        path,
                    )
                else:
                    log.warning("Giving up auto-retry for '%s' after %d attempts", path, attempts)
            except Exception as exc:
                log.warning("Auto-retry scheduling failed for '%s': %s", path, exc)

        def _fail(path: str, exc: BaseException, idx: int, total: int, *, msg: str) -> tuple:
            _log_file_error(msg, path, exc, stage=PipelineStage.ERROR)
            _emit(path, "Error", idx, total, error=str(exc))
            status_store.mark_failed(path, str(exc), exc=exc)
            _schedule_auto_retry(path, str(exc))
            completed.increment()
            return ("error", path, str(exc))

        def _skip_timeout(path: str, exc: BaseException, idx: int, total: int) -> tuple | None:
            """Return a Skipped tuple when the exception is a timeout-skip."""
            if isinstance(exc, ConversionTimeoutError):
                stats.skipped += 1
                _emit(path, "Skipped", idx, total)
                status_store.mark_skipped(path, reason="timeout")
                completed.increment()
                log.info("Skipped '%s' (timeout)", path)
                return ("skipped", path)
            return None

        def _convert_stage(
            path: str,
            sf: SourceFile | None,
            idx: int,
            total: int,
            kind: str,
            _source: Source = source,
            _download_dir: Path = download_dir,
            _src_name: str = src_name,
            _suppress: bool = suppress_deletions,
            _stored: dict[str, str] = stored_hashes,
        ) -> tuple:
            """Stage 1 (convert pool): hash-check / delete / convert — NO LLM work.

            Returns ("ingest", kind, path, local, markdown, chunks, idx, total)
            when the file must go through the ingest stage, or a result-tag
            tuple for short-circuit paths (dry-run, unchanged, deleted).
            Conversions run ahead of the ingest stage, so the MarkItDown/OCR
            queues stay busy while other files are in their LLM phases.
            """
            try:
                if kind == "added":
                    status_store.mark_pending(path, source_name=_src_name)
                if kind == "unchanged":
                    try:
                        current_hash = _source.get_content_hash(sf)
                    except Exception as exc:
                        return _fail(path, exc, idx, total, msg="Failed to hash '%s': %s")
                    stored_hash = _stored.get(path, "")
                    if current_hash == stored_hash:
                        # Hash unchanged — but the stored metadata may be stale
                        # (e.g. METADATA_VERSION bumped). Re-ingest in that case
                        # so new metadata fields/prompts reach the collection.
                        try:
                            already, _ = engine.is_already_ingested(path, current_hash)
                        except Exception:
                            already = True
                        if already:
                            _emit(path, "Done", idx, total)
                            completed.increment()
                            return ("unchanged", path)
                        log.info("Metadata version changed for %s — re-ingesting", path)
                        kind = "changed"
                    elif dry_run:
                        log.info("[dry-run] Would update: %s", path)
                        _emit(path, "Done", idx, total)
                        completed.increment()
                        return ("changed", path)
                    # Changed file — delete old chunks, then convert + ingest
                    kind = "changed"
                if kind == "deleted":
                    if _suppress:
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
                    _emit(path, "Deleting", idx, total)
                    status_store.update_stage(path, PipelineStage.DELETING)
                    engine.delete_by_source(path)
                    status_store.mark_deleted(path)
                    _emit(path, "Done", idx, total)
                    completed.increment()
                    log.info("Deleted '%s'", path)
                    return ("deleted", path)
                if kind == "changed":
                    _emit(path, "Deleting", idx, total)
                    status_store.update_stage(path, PipelineStage.DELETING)
                    engine.delete_by_source(path)
                if dry_run:
                    log.info("[dry-run] Would add: %s", path)
                    _emit(path, "Done", idx, total)
                    completed.increment()
                    return ("added", path)
                # ── Phase 1: download + convert (MarkItDown + OCR inline) ──
                _emit(path, "Converting", idx, total)
                status_store.update_stage(path, PipelineStage.CONVERTING)
                local = _source.download(sf, _download_dir)
                conv = _convert_file(engine, str(local), path, progress_cb, idx, total)
                # Converted but not yet consumed by the LLM stage — say so on
                # the row instead of leaving a stale "Converting".
                _emit(path, "Queued", idx, total)
                return ("ingest", kind, path, str(local), conv["markdown"], conv["chunks"], idx, total)
            except Exception as exc:
                skipped = _skip_timeout(path, exc, idx, total)
                if skipped is not None:
                    return skipped
                return _fail(path, exc, idx, total, msg="Failed to convert '%s': %s")

        def _ingest_stage(
            item: tuple, _src_name: str = src_name
        ) -> tuple[str, str] | tuple[str, str, str]:
            """Stage 2 (serialized): embed + store. Runs only for converted files.

            Runs one file at a time — the LLM phases (context, metadata,
            embedding) are serialized by the global LLM lock anyway, and
            keeping a single consumer means conversions (stage 1) never
            stall behind a file that is mid-LLM.
            """
            _, kind, path, local, markdown, chunks, idx, total = item
            try:
                # ── Phase 2: ingest (embed + store) ──────────────────────
                chunk_count = _ingest_markdown(
                    engine,
                    markdown,
                    chunks,
                    path,
                    local,
                    _src_name,
                    progress_cb,
                )
                _emit(path, "Done", idx, total, chunks=chunk_count)
                status_store.mark_done(path, chunks=chunk_count)
                completed.increment()
                log.info(
                    "%s '%s' (%d chunks)",
                    "Added" if kind == "added" else "Updated",
                    path,
                    chunk_count,
                )
                return (kind, path)
            except Exception as exc:
                skipped = _skip_timeout(path, exc, idx, total)
                if skipped is not None:
                    return skipped
                return _fail(path, exc, idx, total, msg="Failed to ingest '%s': %s")

        # Build work list (kind, path, SourceFile, idx, total)
        work_items: list[tuple] = []
        for path in new_paths:
            file_idx_counter += 1
            work_items.append(("added", path, current_map[path], file_idx_counter, total_files))

        for path in common_paths:
            file_idx_counter += 1
            work_items.append(("unchanged", path, current_map[path], file_idx_counter, total_files))

        for path in deleted_paths:
            file_idx_counter += 1
            work_items.append(("deleted", path, None, file_idx_counter, total_files))

        # Two-stage pipeline: a convert pool plus a single serialized
        # consumer for the LLM phases. Conversions are fed in bounded
        # just-in-time waves — the consumer tops the pipeline up before
        # each file's LLM phases, so the MarkItDown/OCR queues keep
        # converting the next files while the LLM works on the current
        # one (no upfront burst, no long converter idle). The per-file
        # pipeline is written with sync calls and proven stable in plain
        # threads.
        results: list[tuple | BaseException] = []
        if work_items:
            import concurrent.futures

            CONVERT_AHEAD = 8
            convert_workers = max(
                4, int(getattr(config, "MAX_CONCURRENT_SYNC", 2) or 1) + 2
            )
            convert_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=convert_workers, thread_name_prefix="convert"
            )
            work_iter = iter(work_items)
            futures: dict[concurrent.futures.Future, str] = {}

            def _in_flight(
                _futures: dict[concurrent.futures.Future, str] = futures,
            ) -> int:
                return sum(1 for f in _futures if not f.done())

            def _top_up(
                _futures: dict[concurrent.futures.Future, str] = futures,
                _work_iter: Iterator[tuple] = work_iter,
                _convert_pool: concurrent.futures.ThreadPoolExecutor = convert_pool,
                _ahead: int = CONVERT_AHEAD,
            ) -> None:
                """Submit conversions until CONVERT_AHEAD are in flight."""
                while sum(1 for f in _futures if not f.done()) < _ahead:
                    try:
                        kind, path, sf, idx, total = next(_work_iter)
                    except StopIteration:
                        return
                    fut = _convert_pool.submit(_convert_stage, path, sf, idx, total, kind)
                    _futures[fut] = path

            _top_up()
            try:
                while futures:
                    done, _ = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for fut in done:
                        futures.pop(fut)
                        try:
                            item = fut.result()
                        except BaseException as exc:
                            results.append(exc)
                            continue
                        if isinstance(item, BaseException):
                            results.append(item)
                            continue
                        if item[0] == "ingest":
                            # Top up BEFORE the LLM phases — the next wave
                            # converts while this file is mid-LLM.
                            _top_up()
                            results.append(_ingest_stage(item))
                        else:
                            results.append(item)
                    _top_up()
            except KeyboardInterrupt:
                # Ctrl+C must stop the sync promptly. Cancel queued work and
                # don't wait for in-flight files (shutdown(wait=True) would
                # block minutes on LLM-heavy files). Per-file statuses are
                # checkpointed; the next sync resumes pending files.
                log.info("Sync interrupted — cancelling queued work")
                for future in futures:
                    future.cancel()
                convert_pool.shutdown(wait=False, cancel_futures=True)
                raise
            convert_pool.shutdown(wait=True)

            for r in results:
                if isinstance(r, BaseException):
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
                    elif tag == "skipped":
                        stats.skipped += 1
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
