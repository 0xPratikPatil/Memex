"""Document chunking — markdown-aware recursive chunker.

When ``converter.engine`` is ``markitdown``, the converter produces markdown
and we chunk it with the markdown-aware chunker that respects tables, lists,
and code blocks.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from memex.engine.core import config
from memex.engine.core.errors import (
    ChunkingError,
    IngestionError,
)
from memex.engine.core.progress import PipelineStage

logger = logging.getLogger("chunker")

# Pool chunking requests through a semaphore to avoid overwhelming the API.
_chunking_semaphore = threading.BoundedSemaphore(max(1, config.CONVERTER_MAX_CONCURRENT))


# ── Markdown-aware chunking ──────────────────────────────────────────────────


def _classify_block(block: str) -> str:
    """Classify a markdown block as table, code, list, heading, or paragraph."""
    block = block.strip()

    if not block:
        return "empty"

    # Table: starts with | or has multiple | on first line
    if block.startswith("|") and block.count("|") >= 2:
        return "table"

    # Code block: starts with ```
    if block.startswith("```"):
        return "code"

    # Heading: starts with #
    if block.startswith("#"):
        return "heading"

    # List: starts with -, *, or digit + .
    if any(block.startswith(p) for p in ("- ", "* ", "1. ", "2. ", "3. ", "4. ", "5. ")):
        return "list"

    # Indented list item
    if block.startswith("  ") and any(block.lstrip().startswith(p) for p in ("- ", "* ")):
        return "list"

    return "paragraph"


def chunk_markdown_aware(
    markdown: str,
    chunk_size: int = 1024,
    overlap: int = 128,
    filename: str = "",
) -> list[dict[str, Any]]:
    """Chunk markdown with awareness of tables, lists, and code blocks.

    Splits by double-newline into blocks, classifies each block, and never
    splits a table/list/code block mid-element. Groups blocks into chunks
    respecting chunk_size and overlap.
    """
    if not markdown or not markdown.strip():
        return []

    # Split into blocks by double newline
    blocks = [b.strip() for b in markdown.split("\n\n") if b.strip()]

    # Classify each block
    classified: list[tuple[str, str]] = []  # (type, content)
    for block in blocks:
        block_type = _classify_block(block)
        classified.append((block_type, block))

    # Group blocks into chunks without splitting atomic elements
    chunks: list[dict[str, Any]] = []
    current_blocks: list[str] = []
    current_size = 0
    section_header = ""

    def _flush_chunk() -> None:
        if current_blocks:
            chunk_text = "\n\n".join(current_blocks)
            chunks.append({
                "content": chunk_text,
                "section_header": section_header,
                "chunk_index": len(chunks),
            })

    for block_type, block in classified:
        if block_type == "heading":
            # New heading starts a new chunk
            _flush_chunk()
            current_blocks = [block]
            current_size = len(block)
            section_header = block.split("\n")[0].lstrip("#").strip()
        elif block_type in ("table", "code", "list"):
            # Atomic block — don't split it
            if current_size + len(block) > chunk_size and current_blocks:
                _flush_chunk()
                current_blocks = []
                current_size = 0
            current_blocks.append(block)
            current_size += len(block)
        else:
            # Paragraph — can split if needed
            if current_size + len(block) > chunk_size and current_blocks:
                _flush_chunk()
                current_blocks = []
                current_size = 0
            current_blocks.append(block)
            current_size += len(block)

    _flush_chunk()
    return chunks


def _parse_chunk_response(data: dict, include_doc: bool = False) -> dict[str, Any]:
    """Parse a chunk response dict into our chunk format."""
    chunks_raw = data.get("chunks", [])
    documents = data.get("documents", [])

    if not chunks_raw:
        logger.warning("Chunking API returned no chunks")
        result: dict[str, Any] = {"chunks": [], "markdown": ""}
        if include_doc and documents:
            doc = documents[0] if documents else {}
            result["markdown"] = doc.get("md_content") or doc.get("markdown", "")
        return result

    chunks: list[dict[str, Any]] = []
    for item in chunks_raw:
        text = item.get("text", "")
        if isinstance(text, dict):
            text = text.get("text", "") or text.get("content", "") or str(text)
        if not isinstance(text, str) or not text.strip():
            continue

        headings = item.get("headings") or []
        if headings and isinstance(headings[0], dict):
            section_header = headings[0].get("text", "") or headings[0].get("content", "") or str(headings[0])
        else:
            section_header = headings[0] if headings else ""
        if not isinstance(section_header, str):
            section_header = str(section_header)

        chunks.append(
            {
                "content": text,
                "section_header": section_header,
                "headings": headings,
                "chunk_index": item.get("chunk_index", len(chunks)),
            }
        )

    logger.info("Chunking complete — %d chunks from %d raw items", len(chunks), len(chunks_raw))

    result = {"chunks": chunks, "markdown": ""}
    if include_doc and documents:
        doc = documents[0] if documents else {}
        result["markdown"] = doc.get("md_content") or doc.get("markdown", "")
    return result


# ── Public API ───────────────────────────────────────────────────────────────


def chunk_file(file_path_or_url: str, include_doc: bool = False) -> dict[str, Any]:
    """Unified entry point: detect URL vs local path and route accordingly.

    When ``converter.engine`` is ``markitdown``, the converter
    produces markdown and we chunk it with the markdown-aware chunker.
    """
    if config.CONVERTER_ENGINE in ("marker", "markitdown"):
        from memex.engine.ingestion.loader import parse_file

        parse_result = parse_file(file_path_or_url)
        if not parse_result.ok:
            raise IngestionError(
                file_path_or_url,
                f"conversion failed: {parse_result.status} -- {parse_result.errors}",
                stage=PipelineStage.CONVERTING,
            )
        chunks = chunk_markdown_aware(
            parse_result.markdown or "",
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
            filename=file_path_or_url.split("/")[-1],
        )
        return {"chunks": chunks, "markdown": parse_result.markdown}

    raise ChunkingError(
        f"Unsupported converter engine: {config.CONVERTER_ENGINE}. "
        "Only 'markitdown' is supported."
    )
