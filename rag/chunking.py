"""Docling HybridChunker integration — structure-aware, tokenizer-aligned chunking.

Replaces the old regex-based ``_recursive_chunk`` with Docling's native
``HybridChunker`` that operates directly on the ``DoclingDocument`` structure.
Preserves headings, captions, table boundaries, and list groupings.
"""

from __future__ import annotations

import logging
from typing import Any

from rag import config

logger = logging.getLogger("chunking")


def _get_hybrid_chunker():
    """Lazy-import and construct a HybridChunker configured from settings.

    Returns None when ``docling`` is not installed, so callers can fall back
    to the legacy chunker.
    """
    try:
        from docling.chunking import HybridChunker
    except ImportError:
        logger.warning(
            "docling package not installed — HybridChunker unavailable. Install with: uv sync --extra chunking"
        )
        return None

    chunker = HybridChunker(
        tokenizer=config.CHUNK_TOKENIZER,
        max_tokens=config.CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP,
        merge_peers=config.CHUNK_MERGE_PEERS,
        repeat_table_header=config.CHUNK_REPEAT_TABLE_HEADER,
    )
    return chunker


def chunk_docling_document(
    docling_json: dict[str, Any],
) -> list[dict[str, Any]]:
    """Chunk a DoclingDocument JSON using HybridChunker.

    Args:
        docling_json: Raw ``json_content`` from Docling Serve response.

    Returns:
        List of chunk dicts, each with keys: ``content``, ``section_header``,
        ``heading_level``, ``chunk_type``, and any metadata from the chunker.
    """
    chunker = _get_hybrid_chunker()
    if chunker is None:
        raise RuntimeError("HybridChunker not available. Install docling: uv sync --extra chunking")

    try:
        from docling_core.docling_document import DoclingDocument
    except ImportError as e:
        raise RuntimeError("docling_core not available. Install docling: uv sync --extra chunking") from e

    dl_doc = DoclingDocument.model_validate(docling_json)

    chunks: list[dict[str, Any]] = []
    for base_chunk in chunker.chunk(dl_doc):
        serialized = _serialize_chunk(base_chunk)
        heading = base_chunk.meta.heading or ""
        heading_text = heading.heading_text if hasattr(heading, "heading_text") else str(heading)

        chunks.append(
            {
                "content": serialized,
                "section_header": heading_text,
                "heading_level": base_chunk.meta.heading_level or 0,
                "chunk_type": base_chunk.meta.chunk_type or "text",
            }
        )

    if not chunks:
        logger.warning("HybridChunker produced no chunks for document")

    return chunks


def _serialize_chunk(base_chunk: Any) -> str:
    """Serialize a chunk for embedding, choosing format by chunk type."""
    if not config.CHUNK_TYPE_FORMAT:
        return base_chunk.text

    chunk_type = getattr(base_chunk.meta, "chunk_type", "text")

    if chunk_type == "table":
        return _serialize_table_chunk(base_chunk)
    elif chunk_type == "code":
        return _serialize_code_chunk(base_chunk)
    elif chunk_type == "image_description":
        return _serialize_image_chunk(base_chunk)
    else:
        return base_chunk.text


def _serialize_table_chunk(base_chunk: Any) -> str:
    """Serialize a table chunk as HTML to preserve cell/column structure."""
    text = base_chunk.text
    if "<table>" in text or "<tr>" in text:
        return text
    return f"<table>\n{text}\n</table>"


def _serialize_code_chunk(base_chunk: Any) -> str:
    """Serialize a code chunk as a markdown fenced code block."""
    text = base_chunk.text
    if text.startswith("```"):
        return text
    language = getattr(base_chunk.meta, "code_language", "") or ""
    return f"```{language}\n{text}\n```"


def _serialize_image_chunk(base_chunk: Any) -> str:
    """Serialize an image description chunk."""
    caption = getattr(base_chunk.meta, "image_caption", "") or ""
    if caption:
        return f"[Image: {caption}]"
    return base_chunk.text


def is_hybrid_chunker_available() -> bool:
    """Check whether HybridChunker can be imported."""
    return _get_hybrid_chunker() is not None
