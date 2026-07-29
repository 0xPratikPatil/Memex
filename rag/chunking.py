"""Docling HybridChunker — via Docling Serve API.

Uses the Docling Serve ``/v1/chunk/hybrid/source`` endpoint for
structure-aware, tokenizer-aligned chunking. No local ``docling`` or
``docling-core`` packages required — all heavy processing runs in Docker.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rag import config

logger = logging.getLogger("chunking")

# Singleton httpx client for chunking API calls (reused across calls)
_chunking_client: httpx.Client | None = None


# ── Chunking API helpers ─────────────────────────────────────────────────────


def _get_chunking_url() -> str:
    """Build the chunking endpoint URL from the Docling base URL."""
    base = config.DOCLING_URL.split("/v1/convert")[0]
    return f"{base}/v1/chunk/hybrid/source"


def _get_chunking_client() -> httpx.Client:
    """Return a singleton httpx client for chunking API calls."""
    global _chunking_client
    if _chunking_client is None:
        _chunking_client = httpx.Client(
            timeout=httpx.Timeout(config.DOCLING_TIMEOUT, connect=10.0),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )
    return _chunking_client


@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
    stop=stop_after_attempt(config.HTTP_MAX_RETRIES),
    wait=wait_exponential(multiplier=config.HTTP_RETRY_BACKOFF, max=10),
    reraise=True,
)
def _post_chunking(payload: dict) -> dict:
    """POST to the Docling Serve chunking endpoint."""
    url = _get_chunking_url()
    client = _get_chunking_client()
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if config.DOCLING_API_KEY:
        headers["X-Api-Key"] = config.DOCLING_API_KEY
    resp = client.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()


# ── Chunking options ─────────────────────────────────────────────────────────


def _build_chunking_options() -> dict[str, Any]:
    """Build HybridChunkerOptions for the Docling Serve chunking API."""
    return {
        "chunker": "hybrid",
        "max_tokens": config.CHUNK_SIZE,
        "tokenizer": config.CHUNK_TOKENIZER,
        "merge_peers": config.CHUNK_MERGE_PEERS,
    }


def _build_convert_options() -> dict[str, Any]:
    """Build conversion options for the chunking endpoint."""
    opts: dict[str, Any] = {
        "from_formats": ["docx", "pptx", "html", "image", "pdf", "md", "csv", "xlsx"],
        "to_formats": ["md"],
        "do_ocr": config.ENABLE_OCR,
        "table_mode": "accurate",
        "do_table_structure": True,
        "image_export_mode": config.DOCLING_IMAGE_EXPORT,
    }

    if config.DOCLING_ENRICH_CODE:
        opts["do_code_enrichment"] = True
    if config.DOCLING_ENRICH_FORMULA:
        opts["do_formula_enrichment"] = True
    if config.DOCLING_PICTURE_CLASSIFY:
        opts["do_picture_classification"] = True
    if config.DOCLING_CHART_EXTRACT:
        opts["do_chart_extraction"] = True
    if config.DOCLING_PDF_BACKEND:
        opts["pdf_backend"] = config.DOCLING_PDF_BACKEND.lower()

    return opts


# ── Public API ───────────────────────────────────────────────────────────────


def chunk_url(url: str, include_doc: bool = False) -> dict[str, Any]:
    """Convert and chunk a URL via Docling Serve chunking API.

    Returns a dict with keys: ``chunks`` (list of chunk dicts),
    and optionally ``markdown`` (converted document text) when *include_doc* is True.
    """
    payload = {
        "convert_options": _build_convert_options(),
        "chunking_options": _build_chunking_options(),
        "sources": [{"kind": "http", "url": url}],
        "include_converted_doc": include_doc,
    }

    try:
        data = _post_chunking(payload)
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Docling chunking API returned HTTP {exc.response.status_code}: {exc.response.text[:500]}"
        ) from exc
    except httpx.TransportError as exc:
        raise RuntimeError(f"Cannot reach Docling server at {config.DOCLING_URL}: {exc}") from exc

    return _parse_chunk_response(data, include_doc=include_doc)


def chunk_local_file(file_path: str, include_doc: bool = False) -> dict[str, Any]:
    """Convert and chunk a local file via Docling Serve chunking API.

    Args:
        file_path: Absolute path to the file (e.g., /mnt/docs/report.pdf)
        include_doc: If True, include the full markdown in the result.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    import base64
    from pathlib import Path

    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    file_bytes = p.read_bytes()
    b64 = base64.b64encode(file_bytes).decode("ascii")

    payload = {
        "convert_options": _build_convert_options(),
        "chunking_options": _build_chunking_options(),
        "sources": [
            {
                "kind": "file",
                "base64_string": b64,
                "filename": p.name,
            }
        ],
        "include_converted_doc": include_doc,
    }

    try:
        data = _post_chunking(payload)
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Docling chunking API returned HTTP {exc.response.status_code}: {exc.response.text[:500]}"
        ) from exc
    except httpx.TransportError as exc:
        raise RuntimeError(f"Cannot reach Docling server at {config.DOCLING_URL}: {exc}") from exc

    return _parse_chunk_response(data, include_doc=include_doc)


def chunk_file(file_path_or_url: str, include_doc: bool = False) -> dict[str, Any]:
    """Unified entry point: detect URL vs local path and route accordingly."""
    from urllib.parse import urlparse

    parsed = urlparse(file_path_or_url)
    if parsed.scheme in ("http", "https"):
        return chunk_url(file_path_or_url, include_doc=include_doc)
    return chunk_local_file(file_path_or_url, include_doc=include_doc)


def _parse_chunk_response(data: dict, include_doc: bool = False) -> dict[str, Any]:
    """Parse the Docling Serve chunk response into our chunk format.

    Returns a dict with ``chunks`` (list) and optionally ``markdown`` (str).
    """
    chunks_raw = data.get("chunks", [])
    documents = data.get("documents", [])

    if not chunks_raw:
        logger.warning("Docling chunking API returned no chunks")
        result: dict[str, Any] = {"chunks": [], "markdown": ""}
        if include_doc and documents:
            doc = documents[0] if documents else {}
            result["markdown"] = doc.get("md_content") or doc.get("markdown", "")
        return result

    chunks: list[dict[str, Any]] = []
    for item in chunks_raw:
        text = item.get("text", "")
        if not text.strip():
            continue

        headings = item.get("headings") or []
        section_header = headings[0] if headings else ""

        chunks.append(
            {
                "content": text,
                "section_header": section_header,
                "headings": headings,
                "chunk_index": item.get("chunk_index", len(chunks)),
            }
        )

    logger.info(
        "Docling chunking complete — %d chunks from %d raw items",
        len(chunks),
        len(chunks_raw),
    )

    result = {"chunks": chunks, "markdown": ""}
    if include_doc and documents:
        doc = documents[0] if documents else {}
        result["markdown"] = doc.get("md_content") or doc.get("markdown", "")
    return result


def is_hybrid_chunker_available() -> bool:
    """Check whether the Docling Serve chunking endpoint is reachable."""
    try:
        url = _get_chunking_url()
        base = url.split("/v1/chunk")[0]
        health_url = f"{base}/health"
        client = httpx.Client(timeout=5.0)
        try:
            resp = client.get(health_url)
            return resp.status_code == 200
        finally:
            client.close()
    except Exception:
        return False
