"""MarkItDown document conversion client — simple HTTP, CPU-only.

Architecture:
    convert_markdown(file_bytes, filename)
      → POST /convert  (single request, no job polling)

No GPU lock needed — MarkItDown is CPU-only. No subprocess isolation needed —
MarkItDown is lightweight and safe to run in-process.

Error model (typed, from memex.engine.core.errors):
    ConversionError          — server reported failure
    CorruptedDocumentError   — empty output
    ServiceUnavailableError  — server unreachable
"""

from __future__ import annotations

import io
import logging
import threading
from dataclasses import dataclass, field
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
    ConversionError,
    CorruptedDocumentError,
    ServiceUnavailableError,
)

logger = logging.getLogger("markitdown-client")

_client: httpx.Client | None = None
_client_lock = threading.Lock()


@dataclass
class MarkItDownResult:
    """Structured output from MarkItDown conversion."""

    markdown: str
    metadata: dict[str, Any] = field(default_factory=dict)
    format: str = ""
    processing_time: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.markdown.strip())


# ── HTTP helpers ─────────────────────────────────────────────────────────────


def _base_url() -> str:
    return config.MARKITDOWN_URL.rstrip("/")


def _get_client() -> httpx.Client:
    global _client
    if _client is not None and not _client.is_closed:
        return _client
    with _client_lock:
        if _client is not None and not _client.is_closed:
            return _client
        _client = httpx.Client(
            timeout=httpx.Timeout(config.MARKITDOWN_TIMEOUT, connect=10.0),
            limits=httpx.Limits(
                max_connections=8,
                max_keepalive_connections=4,
                keepalive_expiry=30,
            ),
        )
    return _client


@retry(
    retry=retry_if_exception_type((httpx.ConnectError, httpx.ConnectTimeout)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=10),
    reraise=True,
)
def _post_convert(file_bytes: bytes, filename: str) -> httpx.Response:
    """POST file to MarkItDown, retrying only connection-level failures.

    Read timeouts are NOT retried — the server may still be converting the
    file in a worker thread. Retrying would queue the same heavy job again
    and double server load.
    """
    client = _get_client()
    url = f"{_base_url()}/convert"
    files = {"file": (filename, io.BytesIO(file_bytes), "application/octet-stream")}
    data = {"filename": filename}
    return client.post(url, files=files, data=data)


# ── Public API ───────────────────────────────────────────────────────────────


def convert_markdown(file_bytes: bytes, filename: str) -> MarkItDownResult:
    """Convert a document to Markdown via the MarkItDown service.

    Args:
        file_bytes: Raw file content.
        filename: Original filename (used for format detection).

    Raises:
        ConversionError: On conversion failure.
        CorruptedDocumentError: If output is empty.
        ServiceUnavailableError: If MarkItDown is unreachable.
    """
    try:
        resp = _post_convert(file_bytes, filename)
        resp.raise_for_status()
        body = resp.json()
    except httpx.HTTPStatusError as exc:
        raise ConversionError(
            filename,
            f"MarkItDown API error {exc.response.status_code}: {exc.response.text[:200]}",
            cause=exc,
        ) from exc
    except httpx.TransportError as exc:
        raise ServiceUnavailableError(
            "MarkItDown",
            f"cannot reach {_base_url()}: {exc}",
            hint="Check that the markitdown service is running (docker compose up -d markitdown).",
            cause=exc,
        ) from exc

    if not body.get("success", False):
        err = body.get("error", "unknown conversion error")
        raise ConversionError(
            filename,
            err,
            hint="MarkItDown failed to convert this document. Check the file format and content.",
        )

    markdown = body.get("output", "") or ""
    if not markdown.strip():
        raise CorruptedDocumentError(
            f"MarkItDown converted {filename} but returned empty markdown",
            component="conversion",
        )

    logger.info(
        "MarkItDown conversion complete — %d chars markdown, format=%s, time=%.1fs",
        len(markdown),
        body.get("format", "unknown"),
        body.get("processing_time", 0),
    )

    return MarkItDownResult(
        markdown=markdown,
        metadata=body.get("metadata", {}),
        format=body.get("format", ""),
        processing_time=body.get("processing_time", 0),
    )


def is_markitdown_available() -> bool:
    """Check whether the MarkItDown service is reachable."""
    try:
        client = _get_client()
        resp = client.get(f"{_base_url()}/health", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


def get_queue_status() -> dict[str, Any]:
    """Live MarkItDown queue state: which file is converting, which wait.

    Returns:
        {"current": str|None, "pending": [...], "queued": int, "busy": bool}
        or {"error": ...} when the service is unreachable.
    """
    try:
        resp = _get_client().get(f"{_base_url()}/queue", timeout=3.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"error": str(exc)}


def close() -> None:
    """Close the singleton client (process shutdown)."""
    global _client
    if _client is not None and not _client.is_closed:
        _client.close()
        _client = None


__all__ = ["MarkItDownResult", "close", "convert_markdown", "is_markitdown_available"]
