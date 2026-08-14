"""Chunking module — Docling HybridChunker (legacy) + local recursive/fixed.

The Docling Serve ``/v1/chunk/hybrid/source`` endpoint is only used when
``converter.engine=docling`` (legacy rollback path). With the default
``marker`` engine, chunking happens locally (recursive/fixed).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from memex.engine.core import config
from memex.engine.core.errors import (
    ChunkingError,
    ConversionTimeoutError,
    IngestionError,
    ServiceUnavailableError,
)
from memex.engine.core.progress import PipelineStage

logger = logging.getLogger("chunking")

# Singleton httpx client for chunking API calls (reused across calls)
_chunking_client: httpx.Client | None = None
_chunking_client_lock = threading.Lock()

# Global cap on concurrent Docling chunking calls — shared with the convert
# path in loader.py. Keeps in-flight conversions within the server's worker
# pool so requests don't queue and trip DOCLING_SERVE_MAX_SYNC_WAIT.
_chunking_semaphore = threading.BoundedSemaphore(max(1, config.CONVERTER_MAX_CONCURRENT))


def _stop_transport_retry(retry_state) -> bool:
    """Stop when the configurable transport-retry attempt budget is exhausted.

    Read per attempt so config changes (and tests) take effect without
    re-importing the module.
    """
    return retry_state.attempt_number >= config.HTTP_TRANSPORT_MAX_RETRIES


def _wait_transport_retry(retry_state) -> float:
    """Exponential backoff, capped at 15s, base from config."""
    delay = config.HTTP_TRANSPORT_RETRY_BACKOFF * (2 ** (retry_state.attempt_number - 1))
    return min(delay, 15.0)


# ── Chunking API helpers ─────────────────────────────────────────────────────


def _get_chunking_url() -> str:
    """Build the chunking endpoint URL from the Docling base URL."""
    base = config.DOCLING_URL.split("/v1/convert")[0]
    return f"{base}/v1/chunk/hybrid/source"


def _get_chunking_client() -> httpx.Client:
    """Return a singleton httpx client for chunking API calls."""
    global _chunking_client
    if _chunking_client is not None:
        return _chunking_client
    with _chunking_client_lock:
        if _chunking_client is not None:
            return _chunking_client
        _chunking_client = httpx.Client(
            timeout=httpx.Timeout(config.DOCLING_TIMEOUT, connect=10.0),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )
    return _chunking_client


def _is_retryable_status(exc: Exception) -> bool:
    """Check if an HTTPStatusError is retryable (429, 502, 503, 504)."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 502, 503, 504)
    return False


@retry(
    retry=retry_if_exception_type(httpx.TransportError),
    # Transport errors (connection dropped, server restarting) get a longer
    # window — a container restart takes 30-60s, so a 2s window always fails.
    stop=_stop_transport_retry,
    wait=_wait_transport_retry,
    reraise=True,
)
def _post_chunking_transport(payload: dict) -> httpx.Response:
    """POST chunking payload, retrying connection-level failures."""
    url = _get_chunking_url()
    client = _get_chunking_client()
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if config.DOCLING_API_KEY:
        headers["X-Api-Key"] = config.DOCLING_API_KEY
    with _chunking_semaphore:
        resp = client.post(url, json=payload, headers=headers)
    return resp


@retry(
    retry=retry_if_exception_type(httpx.HTTPStatusError),
    stop=stop_after_attempt(config.HTTP_MAX_RETRIES),
    wait=wait_exponential(multiplier=config.HTTP_RETRY_BACKOFF, max=10),
    reraise=True,
)
def _post_chunking_status(resp: httpx.Response) -> dict:
    """Validate the chunking response, retrying transient HTTP statuses."""
    if resp.status_code in (429, 502, 503, 504):
        resp.raise_for_status()
    resp.raise_for_status()
    return resp.json()


def _post_chunking(payload: dict) -> dict:
    """POST to the Docling Serve chunking endpoint with layered retries."""
    resp = _post_chunking_transport(payload)
    return _post_chunking_status(resp)


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
    from memex.engine.ingestion.loader import build_docling_options

    return build_docling_options(to_formats=["md"])


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
        if exc.response.status_code == 504:
            raise ConversionTimeoutError(
                url, timeout_s=config.DOCLING_SERVE_MAX_SYNC_WAIT, cause=exc
            ) from exc
        raise ChunkingError(
            f"Docling chunking API error {exc.response.status_code} for {url}: "
            f"{exc.response.text[:200]}",
            cause=exc,
        ) from exc
    except httpx.TransportError as exc:
        raise ServiceUnavailableError(
            "Docling",
            f"cannot reach {config.DOCLING_URL}: {exc}",
            cause=exc,
        ) from exc

    return _parse_chunk_response(data, include_doc=include_doc)


def chunk_local_file(file_path: str, include_doc: bool = False) -> dict[str, Any]:
    """Convert and chunk a local file via Docling Serve chunking API.

    Args:
        file_path: Absolute path to the file (e.g., /mnt/docs/report.pdf)
        include_doc: If True, include the full markdown in the result.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file exceeds 200MB size limit.
    """
    import base64
    from pathlib import Path

    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Guard against excessive memory usage from base64 encoding
    file_size_mb = p.stat().st_size / (1024 * 1024)
    if file_size_mb > 200:
        raise ValueError(
            f"File too large for chunking API: {file_size_mb:.0f}MB (max 200MB). "
            "Consider splitting the file or using a streaming approach."
        )

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
        if exc.response.status_code == 504:
            raise ConversionTimeoutError(
                file_path,
                timeout_s=config.DOCLING_SERVE_MAX_SYNC_WAIT,
                hint=(
                    "This document is too large/slow for the conversion window. "
                    "Reduce converter.docling_max_concurrent, disable OCR "
                    "(converter.docling_ocr=false) for digital files, or increase "
                    "DOCLING_SERVE_MAX_SYNC_WAIT on the Docling server."
                ),
                cause=exc,
            ) from exc
        raise ChunkingError(
            f"Docling chunking API error {exc.response.status_code} for {p.name}: "
            f"{exc.response.text[:200]}",
            cause=exc,
        ) from exc
    except httpx.TransportError as exc:
        raise ServiceUnavailableError(
            "Docling",
            f"cannot reach {config.DOCLING_URL}: {exc}",
            cause=exc,
        ) from exc

    return _parse_chunk_response(data, include_doc=include_doc)


def chunk_file(file_path_or_url: str, include_doc: bool = False) -> dict[str, Any]:
    """Unified entry point: detect URL vs local path and route accordingly.

    When ``converter.engine`` is ``marker``, Marker converts the document but
    does not chunk — we return the markdown with no chunks so the caller runs
    the local recursive/fixed chunker (the RAG default).
    """
    from urllib.parse import urlparse

    if config.CONVERTER_ENGINE == "marker":
        from memex.engine.ingestion.loader import parse_file

        parse_result = parse_file(file_path_or_url)
        if not parse_result.ok:
            raise IngestionError(
                file_path_or_url,
                f"conversion failed: {parse_result.status} -- {parse_result.errors}",
                stage=PipelineStage.CONVERTING,
            )
        return {"chunks": [], "markdown": parse_result.markdown}

    parsed = urlparse(file_path_or_url)
    if parsed.scheme in ("http", "https"):
        return chunk_url(file_path_or_url, include_doc=include_doc)
    return chunk_local_file(file_path_or_url, include_doc=include_doc)


def _parse_chunk_response(data: dict, include_doc: bool = False) -> dict[str, Any]:
    """Parse the Docling Serve chunk response into our chunk format.

    Expects the Docling ``ChunkDocumentResponse`` shape:
      - chunks: list[ChunkedDocumentResultItem] (text: str, headings: list[str]|null)
      - documents: list[DocumentResultItem] (content.md_content: str|null)
      - processing_time: float

    Defensive isinstance checks handle Docling server version drift where
    fields may arrive as dicts instead of expected types.
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
        # Docling may return text as a dict in some formats — extract string
        if isinstance(text, dict):
            text = text.get("text", "") or text.get("content", "") or str(text)
        if not isinstance(text, str) or not text.strip():
            continue

        headings = item.get("headings") or []
        # Headings may be list[str] or list[dict] — normalize to string
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
        client = _get_chunking_client()
        resp = client.get(health_url, timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False
