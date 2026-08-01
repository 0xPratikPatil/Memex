"""Content-hash deduplication and partial ingest recovery."""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from memex.engine.core import config

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

log = logging.getLogger(__name__)


def compute_content_hash(content: str | bytes) -> str:
    """Compute SHA256 hash of content."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def compute_chunk_hash(chunk_id: str, content: str) -> str:
    """Compute SHA256 hash of a chunk (id + content)."""
    return hashlib.sha256(f"{chunk_id}:{content}".encode()).hexdigest()


async def is_already_ingested(
    qdrant_client: QdrantClient,
    collection: str,
    source: str,
    content_hash: str,
) -> tuple[bool, int]:
    """Check if a document with this content hash is already ingested.

    Returns (is_duplicate, existing_chunk_count).
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    result = qdrant_client.scroll(
        collection_name=collection,
        limit=1,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="source", match=MatchValue(value=source)),
                FieldCondition(key="content_hash", match=MatchValue(value=content_hash)),
            ]
        ),
        with_payload=["total_chunks"],
        with_vectors=False,
    )
    points, _ = result
    if points:
        count = (points[0].payload or {}).get("total_chunks", 0)
        log.debug("Duplicate detected for source=%s, content_hash=%s (%d chunks)", source, content_hash, count)
        return True, count
    return False, 0


async def check_partial_ingest(
    qdrant_client: QdrantClient,
    collection: str,
    source: str,
    expected_chunks: int,
) -> bool:
    if expected_chunks <= 0:
        return False

    from qdrant_client.models import FieldCondition, Filter, MatchValue

    result = qdrant_client.scroll(
        collection_name=collection,
        limit=1,
        scroll_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]),
        with_payload=["total_chunks"],
        with_vectors=False,
    )
    points, _ = result
    if not points:
        return False

    payload = points[0].payload or {}
    stored_total = int(payload.get("total_chunks", 1))
    return stored_total != expected_chunks


async def clear_source_chunks(
    qdrant_client: QdrantClient,
    collection: str,
    source: str,
) -> int:
    """Delete all chunks for a given source. Returns count deleted."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    point_ids: list[str] = []
    offset = None
    while True:
        result = qdrant_client.scroll(
            collection_name=collection,
            limit=500,
            offset=offset,
            scroll_filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))]),
            with_payload=[],
            with_vectors=False,
        )
        points, next_offset = result
        for p in points:
            point_ids.append(str(p.id))
        if next_offset is None:
            break
        offset = next_offset

    if not point_ids:
        return 0

    for i in range(0, len(point_ids), config.EMBED_BATCH_SIZE):
        qdrant_client.delete(
            collection_name=collection,
            points_selector=point_ids[i : i + config.EMBED_BATCH_SIZE],  # type: ignore[arg-type]
        )

    log.info("Cleared %d chunks for source=%s", len(point_ids), source)
    return len(point_ids)


def dedup_chunks(chunks: list[dict]) -> list[dict]:
    """Remove exact duplicate chunks within a document.

    Scoped per-document, not cross-document.
    Uses content-only hash for comparison.
    """
    seen: set[str] = set()
    result: list[dict] = []
    for chunk in chunks:
        content = chunk.get("content", "")
        h = compute_content_hash(content)
        if h not in seen:
            seen.add(h)
            result.append(chunk)
    if len(result) < len(chunks):
        log.debug("Dedup: %d -> %d chunks", len(chunks), len(result))
    return result
