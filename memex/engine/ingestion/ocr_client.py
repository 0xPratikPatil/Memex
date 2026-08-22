"""OCR fallback client — calls the lightweight OCR Docker service.

Used when Marker fails with OOM on scanned PDFs. Sends the PDF to the
OCR service which runs PP-OCRv6 small (ONNX) for text extraction.

Architecture mirrors marker_client.py: HTTP client with retry, structured errors.
"""

from __future__ import annotations

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
    ServiceUnavailableError,
)

logger = logging.getLogger("ocr-client")

_client: httpx.Client | None = None
_client_lock = threading.Lock()


@dataclass
class OcrResult:
    """Structured output from OCR service conversion."""

    markdown: str
    pages: list[dict[str, Any]] = field(default_factory=list)
    model: str = "pp-ocrv6-small"
    status: str = "success"
    processing_time: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "success" and bool(self.markdown.strip())


# ── HTTP helpers ─────────────────────────────────────────────────────────────


def _base_url() -> str:
    return config.OCR_URL.rstrip("/")


def _get_client() -> httpx.Client:
    global _client
    if _client is not None and not _client.is_closed:
        return _client
    with _client_lock:
        if _client is not None and not _client.is_closed:
            return _client
        _client = httpx.Client(
            timeout=httpx.Timeout(config.OCR_TIMEOUT, connect=10.0),
            limits=httpx.Limits(
                max_connections=4,
                max_keepalive_connections=2,
                keepalive_expiry=30,
            ),
        )
    return _client


@retry(
    retry=retry_if_exception_type(httpx.ConnectError),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, max=5),
    reraise=True,
)
def _post(url: str, files: list[tuple[str, tuple[str, bytes, str]]]) -> httpx.Response:
    """POST files to OCR service, retrying only connection refusals.

    Read timeouts are NOT retried — the server may be mid-OCR on a long
    job and a retry would restart minutes of work.
    """
    client = _get_client()
    return client.post(url, files=files)


# ── Public API ──────────────────────────────────────────────────────────────


def is_ocr_available() -> bool:
    """Check if the OCR service is reachable."""
    try:
        resp = _get_client().get(f"{_base_url()}/health")
        return resp.status_code == 200
    except Exception:
        return False


def convert_with_ocr(file_bytes: bytes, filename: str) -> OcrResult:
    """Send a PDF to the OCR service for text extraction.

    The OCR service accepts PDF page images and returns extracted text
    as markdown. For full PDFs, we send the raw bytes and let the server
    handle page splitting.

    Args:
        file_bytes: Raw PDF file bytes.
        filename: Original filename (used for content-type detection).

    Returns:
        OcrResult with extracted markdown.

    Raises:
        ConversionError: OCR service returned an error.
        ServiceUnavailableError: OCR service is unreachable.
    """
    url = f"{_base_url()}/convert"

    try:
        resp = _post(
            url,
            files=[("files", (filename, file_bytes, "application/pdf"))],
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ConversionError(
            filename,
            f"OCR service error {exc.response.status_code}: {exc.response.text[:200]}",
            hint="Check OCR service logs: docker compose logs ocr",
            cause=exc,
        ) from exc
    except httpx.TransportError as exc:
        raise ServiceUnavailableError(
            "OCR",
            f"cannot reach OCR service at {_base_url()}: {exc}",
            hint="Check: docker compose ps | grep ocr",
            cause=exc,
        ) from exc

    data = resp.json()
    return OcrResult(
        markdown=data.get("markdown", ""),
        pages=data.get("pages", []),
        model=data.get("model", "unknown"),
        status="success",
        processing_time=data.get("processing_time", 0.0),
    )
