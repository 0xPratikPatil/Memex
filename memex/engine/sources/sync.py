"""Sync engine — reconcile sources against the vector collection."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from memex.engine.core import config
from memex.engine.sources import SourceFile, get_source

log = logging.getLogger(__name__)


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
            scroll_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_name))]),
            with_payload=["source", "content_hash"],
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


def _ingest_file(engine, file_path: str, source_name: str) -> int:
    """Download, convert, and ingest a single file. Returns chunk count."""
    from memex.engine.ingestion.loader import parse_file

    result = parse_file(file_path)
    if not result.ok:
        raise RuntimeError(f"Docling conversion failed for {file_path}: {result.status} -- {result.errors}")

    content_hash = engine.compute_file_hash(result.markdown.encode())
    count = engine.ingest_text(
        result.markdown,
        source_identifier=file_path,
        metadata={
            "content_type": file_path.rsplit(".", 1)[-1] if "." in file_path else "",
            "content_hash": content_hash,
            "source_name": source_name,
        },
        content_hash=content_hash,
    )
    return count


async def sync(config_module, source_name: str | None = None, dry_run: bool = False) -> SyncStats:
    """Sync collection against configured sources.

    1. Load sources from config
    2. For each source: list_files()
    3. For each file: compute content hash
    4. Query Qdrant for stored hashes
    5. Reconcile: new -> ingest, changed -> delete+ingest, deleted -> delete
    6. Safety: if any source fails to list, suppress deletions
    7. Return stats
    """
    from memex.engine.core.pipeline import RAGEngine

    stats = SyncStats()
    suppress_deletions = False

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

    # 4. Get stored hashes for each source and reconcile
    with tempfile.TemporaryDirectory(prefix="memex_sync_") as tmp_dir:
        for src_cfg in source_configs:
            src_type = src_cfg.get("type", "local")
            src_name = src_cfg.get("name", src_type)
            current_files = all_source_files.get(src_name, [])
            source = get_source(src_type, src_cfg)

            # Build current file map: path -> SourceFile
            current_map: dict[str, SourceFile] = {f.path: f for f in current_files}

            # Build expanded path form map for cross-form matching
            _current_form_map: dict[str, str] = {}
            for fp in current_map:
                for form in _path_forms(fp):
                    _current_form_map[form] = fp

            # Query stored hashes for this source
            stored_hashes = _get_stored_hashes(engine, src_name)

            # Resolve stored paths to canonical current paths via form expansion
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

            # Determine new, changed, and unchanged files
            new_paths = current_paths - stored_paths
            common_paths = current_paths & stored_paths

            for path in new_paths:
                sf = current_map[path]
                if dry_run:
                    log.info("[dry-run] Would add: %s", path)
                    stats.added += 1
                    continue
                try:
                    local_path = source.download(sf, Path(tmp_dir))
                    chunk_count = _ingest_file(engine, str(local_path), src_name)
                    stats.added += 1
                    log.info("Added '%s' (%d chunks)", path, chunk_count)
                except Exception as exc:
                    log.error("Failed to ingest '%s': %s", path, exc)
                    stats.errors.append(f"ingest failed for '{path}': {exc}")

            for path in common_paths:
                sf = current_map[path]
                try:
                    current_hash = source.get_content_hash(sf)
                except Exception as exc:
                    log.error("Failed to hash '%s': %s", path, exc)
                    stats.errors.append(f"hash failed for '{path}': {exc}")
                    continue

                stored_hash = stored_hashes.get(path, "")
                if current_hash == stored_hash:
                    stats.unchanged += 1
                    continue

                if dry_run:
                    log.info("[dry-run] Would update: %s", path)
                    stats.changed += 1
                    continue

                # Changed file: delete old chunks then ingest new
                try:
                    engine.delete_by_source(path)
                    local_path = source.download(sf, Path(tmp_dir))
                    chunk_count = _ingest_file(engine, str(local_path), src_name)
                    stats.changed += 1
                    log.info("Updated '%s' (%d chunks)", path, chunk_count)
                except Exception as exc:
                    log.error("Failed to update '%s': %s", path, exc)
                    stats.errors.append(f"update failed for '{path}': {exc}")

            # 5. Detect deleted files
            deleted_paths = stored_paths - current_paths
            if deleted_paths and not suppress_deletions:
                for path in deleted_paths:
                    if dry_run:
                        log.info("[dry-run] Would delete: %s", path)
                        stats.deleted += 1
                        continue
                    try:
                        engine.delete_by_source(path)
                        stats.deleted += 1
                        log.info("Deleted '%s'", path)
                    except Exception as exc:
                        log.error("Failed to delete '%s': %s", path, exc)
                        stats.errors.append(f"delete failed for '{path}': {exc}")
            elif deleted_paths and suppress_deletions:
                log.warning(
                    "Suppressing %d deletions for source '%s' (listing failed or empty)",
                    len(deleted_paths),
                    src_name,
                )

    log.info("Sync complete: %s", stats.summary())
    return stats
