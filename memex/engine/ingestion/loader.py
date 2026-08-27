"""Document conversion — MarkItDown first, OCR fallback for scanned PDFs.

Flow:
  1. MarkItDown converts the document (PDF, DOCX, PPTX, XLSX, HTML, etc.)
  2. If MarkItDown fails or produces poor output (scanned PDF) → OCR fallback
  3. OCR uses PP-OCRv6 small (ONNX) for text extraction from page images
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from memex.engine.core import config
from memex.engine.core.errors import (
    ConversionError,
    CorruptedDocumentError,
    ServiceUnavailableError,
)

logger = logging.getLogger("converter-client")

_client: httpx.Client | None = None
_client_lock = threading.Lock()


# ── Structured result ────────────────────────────────────────────────────────


@dataclass
class ConversionResult:
    """Structured output from document conversion."""

    markdown: str
    json_content: dict[str, Any] = field(default_factory=dict)
    html_content: str = ""
    text_content: str = ""
    status: str = "success"
    processing_time: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in ("success", "partial_success")


# ── HTTP helpers ─────────────────────────────────────────────────────────────


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


# ── Quality detection ────────────────────────────────────────────────────────

# Only these formats can be OCR'd — OCR on anything else (docx, audio, xlsx…)
# is a guaranteed failure and wastes an OCR queue slot.
_OCRABLE_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _is_ocrable(filename: str) -> bool:
    return Path(filename).suffix.lower() in _OCRABLE_EXTENSIONS


def _is_poor_quality(result: ConversionResult, file_bytes: bytes, filename: str) -> bool:
    """Detect poor conversion quality (e.g., scanned PDF via MarkItDown).

    Returns True only for OCR-able formats (PDF/images) whose output
    suggests the document was scanned and would benefit from OCR fallback.
    """
    if not _is_ocrable(filename):
        return False
    text = (result.markdown or "").strip()
    if not text:
        return True
    # Short text for a moderate-sized file suggests scanned content
    if len(text) < 500 and len(file_bytes) > 5_000:
        return True
    # Low text-to-bytes ratio
    return len(text) / max(len(file_bytes), 1) < 0.005


def _ocr_to_conversion(ocr_result) -> ConversionResult:
    """Wrap OcrResult in ConversionResult."""
    return ConversionResult(
        markdown=ocr_result.markdown or "",
        status="success" if ocr_result.ok else "error",
        processing_time=ocr_result.processing_time,
        errors=[] if ocr_result.ok else ["OCR fallback failed"],
    )


# ── MarkItDown conversion ────────────────────────────────────────────────────


def _markitdown_convert(file_bytes: bytes, filename: str) -> ConversionResult:
    """Convert via MarkItDown service. Raises on transport errors."""
    from memex.engine.ingestion.markitdown_client import convert_markdown as md_convert

    md_result = md_convert(file_bytes, filename)
    return ConversionResult(
        markdown=md_result.markdown,
        json_content=md_result.metadata,
        status="success",
        processing_time=md_result.processing_time,
    )


def _ocr_fallback(file_bytes: bytes, filename: str) -> ConversionResult:
    """Run OCR fallback. Raises on transport errors."""
    from memex.engine.ingestion.ocr_client import convert_with_ocr, is_ocr_available

    if not is_ocr_available():
        logger.warning("OCR fallback skipped — OCR service not reachable")
        return ConversionResult(markdown="", status="error", errors=["OCR service unavailable"])

    ocr_result = convert_with_ocr(file_bytes, filename)
    if ocr_result.ok:
        logger.info("OCR succeeded for %s (%d chars)", filename, len(ocr_result.markdown))
        return _ocr_to_conversion(ocr_result)

    logger.warning("OCR returned no text for %s", filename)
    return ConversionResult(markdown="", status="error", errors=["OCR returned no text"])


# ── Public API ───────────────────────────────────────────────────────────────


def parse_url(url: str) -> ConversionResult:
    """Fetch a URL, convert via MarkItDown, OCR fallback if poor quality."""
    from memex.engine.utils.cache import cache_parse_result, get_cached_parse_result

    # Fetch the URL first — we need the content to compute a content-based cache key
    try:
        resp = httpx.get(url, timeout=config.HTTP_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ConversionError(
            url,
            f"URL fetch error {exc.response.status_code}: {exc.response.text[:200]}",
            hint="The URL is unreachable or returned an error status.",
            cause=exc,
        ) from exc
    except httpx.TransportError as exc:
        raise ServiceUnavailableError(
            "URL",
            f"cannot reach {url}: {exc}",
            hint="Check the URL is reachable from this host.",
            cause=exc,
        ) from exc

    # Content-based cache key — different content → different key → fresh conversion
    file_hash = hashlib.sha256(resp.content).hexdigest()[:16]
    cached = get_cached_parse_result(file_hash)
    if cached is not None:
        logger.info("Converter cache hit for URL: %s", url)
        return ConversionResult(
            markdown=cached["markdown"],
            json_content=cached.get("json_content", {}),
            status=cached.get("status", "success"),
            processing_time=cached.get("processing_time", 0.0),
            errors=cached.get("errors", []),
        )

    filename = url.split("/")[-1].split("?")[0] or "document"

    # MarkItDown conversion
    try:
        converted = _markitdown_convert(resp.content, filename)
    except ServiceUnavailableError:
        raise
    except (CorruptedDocumentError, ConversionError) as exc:
        logger.warning("MarkItDown failed for %s: %s — trying OCR", filename, exc)
        converted = ConversionResult(markdown="", status="success", errors=[str(exc)])

    # OCR fallback for poor quality (scanned PDFs / images)
    if _is_poor_quality(converted, resp.content, filename):
        try:
            ocr_result = _ocr_fallback(resp.content, filename)
            if ocr_result.ok:
                converted = ocr_result
        except ServiceUnavailableError:
            raise
        except Exception as e:
            logger.warning("OCR fallback failed for %s: %s", filename, e)

    if not converted.markdown.strip():
        raise CorruptedDocumentError(
            f"Conversion produced empty markdown for {filename} (MarkItDown + OCR both failed)",
            component="conversion",
        )

    cache_parse_result(
        file_hash,
        {
            "markdown": converted.markdown,
            "json_content": converted.json_content,
            "status": converted.status,
            "processing_time": converted.processing_time,
            "errors": converted.errors,
        },
    )
    return converted


def parse_local_file(file_path: str) -> ConversionResult:
    """Read a local file, convert via MarkItDown, OCR fallback if poor quality."""
    from memex.engine.utils.cache import cache_parse_result, get_cached_parse_result

    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    file_bytes = p.read_bytes()
    if len(file_bytes) > 200 * 1024 * 1024:
        raise ValueError(
            f"File too large ({len(file_bytes) / 1024 / 1024:.0f}MB > 200MB). Use chunking module for large files."
        )

    # Content-based cache key — different content → different key → fresh conversion
    file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
    cached = get_cached_parse_result(file_hash)
    if cached is not None:
        logger.info("Converter cache hit for local file: %s", file_path)
        return ConversionResult(
            markdown=cached["markdown"],
            json_content=cached.get("json_content", {}),
            status=cached.get("status", "success"),
            processing_time=cached.get("processing_time", 0.0),
            errors=cached.get("errors", []),
        )

    filename = p.name

    # MarkItDown conversion
    try:
        converted = _markitdown_convert(file_bytes, filename)
    except ServiceUnavailableError:
        raise
    except (CorruptedDocumentError, ConversionError) as exc:
        logger.warning("MarkItDown failed for %s: %s — trying OCR", filename, exc)
        converted = ConversionResult(markdown="", status="success", errors=[str(exc)])

    # OCR fallback for poor quality (scanned PDFs / images)
    if _is_poor_quality(converted, file_bytes, filename):
        try:
            ocr_result = _ocr_fallback(file_bytes, filename)
            if ocr_result.ok:
                converted = ocr_result
        except ServiceUnavailableError:
            raise
        except Exception as e:
            logger.warning("OCR fallback failed for %s: %s", filename, e)

    if not converted.markdown.strip():
        raise CorruptedDocumentError(
            f"Conversion produced empty markdown for {filename} (MarkItDown + OCR both failed)",
            component="conversion",
        )

    cache_parse_result(
        file_hash,
        {
            "markdown": converted.markdown,
            "json_content": converted.json_content,
            "status": converted.status,
            "processing_time": converted.processing_time,
            "errors": converted.errors,
        },
    )
    return converted


def parse_file(file_path_or_url: str) -> ConversionResult:
    """Unified entry point: detect URL vs local path and route accordingly."""
    parsed = urlparse(file_path_or_url)
    if parsed.scheme in ("http", "https"):
        return parse_url(file_path_or_url)
    return parse_local_file(file_path_or_url)


def close() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        _client.close()
        _client = None
