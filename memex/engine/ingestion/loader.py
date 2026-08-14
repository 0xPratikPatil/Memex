"""Docling document conversion client — v1 API.

Uses ``httpx`` for connection pooling and ``tenacity`` for automatic retries
with exponential back-off.  Works both inside Docker and on bare metal.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
    ConversionTimeoutError,
    CorruptedDocumentError,
    ServiceUnavailableError,
)

logger = logging.getLogger("docling-client")

_client: httpx.Client | None = None
_client_lock = threading.Lock()

# Global cap on concurrent Docling conversions. Prevents callers (sync,
# ingest) from overwhelming the single Docling instance — excess requests
# queue here instead of piling up inside the server and timing out.
# Must stay <= DOCLING_SERVE_ENG_LOC_NUM_WORKERS in docker-compose.yml.
_docling_semaphore = threading.BoundedSemaphore(max(1, config.CONVERTER_MAX_CONCURRENT))


# ── Structured result ────────────────────────────────────────────────────────


@dataclass
class ConversionResult:
    """Structured output from Docling conversion."""

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
            timeout=httpx.Timeout(config.DOCLING_TIMEOUT, connect=10.0),
            limits=httpx.Limits(
                max_connections=8,
                max_keepalive_connections=4,
                keepalive_expiry=30,
            ),
        )
    return _client


_RETRYABLE_STATUSES = (429, 502, 503, 504)


def _is_retryable(exc: Exception) -> bool:
    return (
        isinstance(exc, (httpx.TransportError, httpx.TimeoutException))
        or (isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in _RETRYABLE_STATUSES)
    )


def _stop_transport_retry(retry_state) -> bool:
    """Stop when the configurable transport-retry budget is exhausted (per-attempt)."""
    return retry_state.attempt_number >= config.HTTP_TRANSPORT_MAX_RETRIES


def _wait_transport_retry(retry_state) -> float:
    """Exponential backoff, capped at 15s, base from config."""
    delay = config.HTTP_TRANSPORT_RETRY_BACKOFF * (2 ** (retry_state.attempt_number - 1))
    return min(delay, 15.0)


@retry(
    retry=retry_if_exception_type(httpx.TransportError),
    # Connection-level failures (server restart, dropped keep-alive) get a
    # longer window — a container restart takes 30-60s.
    stop=_stop_transport_retry,
    wait=_wait_transport_retry,
    reraise=True,
)
def _post_transport(payload: dict) -> httpx.Response:
    """POST to Docling, retrying connection-level failures."""
    client = _get_client()
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if config.DOCLING_API_KEY:
        headers["X-Api-Key"] = config.DOCLING_API_KEY
    with _docling_semaphore:
        return client.post(config.DOCLING_URL, json=payload, headers=headers)


@retry(
    retry=retry_if_exception_type(httpx.HTTPStatusError),
    stop=stop_after_attempt(config.HTTP_MAX_RETRIES),
    wait=wait_exponential(multiplier=config.HTTP_RETRY_BACKOFF, max=10),
    reraise=True,
)
def _post_status(resp: httpx.Response) -> dict:
    """Validate the conversion response, retrying transient HTTP statuses."""
    if resp.status_code in _RETRYABLE_STATUSES:
        resp.raise_for_status()
    resp.raise_for_status()
    return resp.json()


def _post(payload: dict) -> dict:
    """POST to the Docling conversion endpoint with layered retries."""
    resp = _post_transport(payload)
    return _post_status(resp)


# ── Conversion options ───────────────────────────────────────────────────────


def build_docling_options(to_formats: list[str] | None = None) -> dict[str, Any]:
    """Build Docling conversion options from config. Single source of truth.

    Args:
        to_formats: Output formats. Defaults to ["md", "json", "html", "text"].
    """
    opts: dict[str, Any] = {
        "from_formats": ["docx", "pptx", "html", "image", "pdf", "md", "csv", "xlsx"],
        "to_formats": to_formats or ["md", "json", "html", "text"],
        "do_ocr": config.ENABLE_OCR,
        "table_mode": config.DOCLING_TABLE_MODE,
        "do_table_structure": config.DOCLING_TABLE_STRUCTURE,
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


def _build_options() -> dict[str, Any]:
    """Build Docling v1 conversion options from config."""
    return build_docling_options()


# ── Public API ───────────────────────────────────────────────────────────────


def _marker_result_to_conversion(result, url_or_path: str) -> ConversionResult:
    """Convert a MarkerResult into the shared ConversionResult shape."""
    return ConversionResult(
        markdown=result.markdown,
        json_content=result.metadata,
        html_content="",
        text_content="",
        status="success",
        processing_time=result.processing_time,
        errors=[],
    )


def parse_url(url: str) -> ConversionResult:
    """Fetch a URL via the configured converter and return structured result."""
    from memex.engine.utils.cache import cache_parse_result, get_cached_parse_result

    file_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    cached = get_cached_parse_result(file_hash)
    if cached is not None:
        logger.info("Converter cache hit for URL: %s", url)
        return ConversionResult(
            markdown=cached["markdown"],
            json_content=cached.get("json_content", {}),
            html_content=cached.get("html_content", ""),
            text_content=cached.get("text_content", ""),
            status=cached.get("status", "success"),
            processing_time=cached.get("processing_time", 0.0),
            errors=cached.get("errors", []),
        )

    if config.CONVERTER_ENGINE == "marker":
        # Marker's /marker/upload accepts file uploads only — fetch the URL
        # first, then convert the bytes.
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

        filename = url.split("/")[-1].split("?")[0] or "document"
        from memex.engine.ingestion.marker_client import convert_markdown

        result = convert_markdown(resp.content, filename)
        converted = _marker_result_to_conversion(result, url)
        cache_parse_result(
            file_hash,
            {
                "markdown": converted.markdown,
                "json_content": converted.json_content,
                "html_content": converted.html_content,
                "text_content": converted.text_content,
                "status": converted.status,
                "processing_time": converted.processing_time,
                "errors": converted.errors,
            },
        )
        return converted

    payload = {
        "options": _build_options(),
        "sources": [{"kind": "http", "url": url}],
    }

    try:
        data = _post(payload)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 504:
            raise ConversionTimeoutError(url, timeout_s=config.DOCLING_TIMEOUT, cause=exc) from exc
        raise ConversionError(
            url,
            f"Docling API error {exc.response.status_code}: {exc.response.text[:200]}",
            cause=exc,
        ) from exc
    except httpx.TransportError as exc:
        raise ServiceUnavailableError(
            "Docling",
            f"cannot reach {config.DOCLING_URL}: {exc}",
            cause=exc,
        ) from exc

    converted_result = _parse_response(data)
    cache_parse_result(
        file_hash,
        {
            "markdown": converted_result.markdown,
            "json_content": converted_result.json_content,
            "html_content": converted_result.html_content,
            "text_content": converted_result.text_content,
            "status": converted_result.status,
            "processing_time": converted_result.processing_time,
            "errors": converted_result.errors,
        },
    )
    return converted_result


def parse_local_file(file_path: str) -> ConversionResult:
    """Read a local file directly and convert via the configured converter.

    Args:
        file_path: Absolute path to the file (e.g., /mnt/docs/report.pdf)

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    from memex.engine.utils.cache import cache_parse_result, get_cached_parse_result

    file_hash = hashlib.sha256(file_path.encode()).hexdigest()[:16]
    cached = get_cached_parse_result(file_hash)
    if cached is not None:
        logger.info("Converter cache hit for local file: %s", file_path)
        return ConversionResult(
            markdown=cached["markdown"],
            json_content=cached.get("json_content", {}),
            html_content=cached.get("html_content", ""),
            text_content=cached.get("text_content", ""),
            status=cached.get("status", "success"),
            processing_time=cached.get("processing_time", 0.0),
            errors=cached.get("errors", []),
        )

    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    file_bytes = p.read_bytes()
    if len(file_bytes) > 200 * 1024 * 1024:  # 200MB limit
        raise ValueError(
            f"File too large ({len(file_bytes) / 1024 / 1024:.0f}MB > 200MB). Use chunking module for large files."
        )
    filename = p.name

    if config.CONVERTER_ENGINE == "marker":
        from memex.engine.ingestion.marker_client import convert_markdown

        result = convert_markdown(file_bytes, filename)
        converted = _marker_result_to_conversion(result, file_path)
        cache_parse_result(
            file_hash,
            {
                "markdown": converted.markdown,
                "json_content": converted.json_content,
                "html_content": converted.html_content,
                "text_content": converted.text_content,
                "status": converted.status,
                "processing_time": converted.processing_time,
                "errors": converted.errors,
            },
        )
        return converted

    b64 = base64.b64encode(file_bytes).decode("ascii")

    payload = {
        "options": _build_options(),
        "sources": [
            {
                "kind": "file",
                "base64_string": b64,
                "filename": filename,
            }
        ],
    }

    try:
        data = _post(payload)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 504:
            raise ConversionTimeoutError(
                file_path, timeout_s=config.DOCLING_TIMEOUT, cause=exc
            ) from exc
        raise ConversionError(
            file_path,
            f"Docling API error {exc.response.status_code}: {exc.response.text[:200]}",
            cause=exc,
        ) from exc
    except httpx.TransportError as exc:
        raise ServiceUnavailableError(
            "Docling",
            f"cannot reach {config.DOCLING_URL}: {exc}",
            cause=exc,
        ) from exc

    converted_result = _parse_response(data)
    cache_parse_result(
        file_hash,
        {
            "markdown": converted_result.markdown,
            "json_content": converted_result.json_content,
            "html_content": converted_result.html_content,
            "text_content": converted_result.text_content,
            "status": converted_result.status,
            "processing_time": converted_result.processing_time,
            "errors": converted_result.errors,
        },
    )
    return converted_result


def parse_file(file_path_or_url: str) -> ConversionResult:
    """Unified entry point: detect URL vs local path and route accordingly."""
    parsed = urlparse(file_path_or_url)
    if parsed.scheme in ("http", "https"):
        return parse_url(file_path_or_url)
    return parse_local_file(file_path_or_url)


def _parse_response(data: dict) -> ConversionResult:
    """Parse Docling v1 API response into ConversionResult."""
    status = data.get("status", "failure")
    errors = data.get("errors", [])
    processing_time = data.get("processing_time", 0.0)

    doc = data.get("document", {})
    markdown_text = doc.get("md_content") or doc.get("markdown", "")
    # Docling may return content as dict in some formats — extract string
    if isinstance(markdown_text, dict):
        markdown_text = markdown_text.get("text", "") or markdown_text.get("content", "") or str(markdown_text)
    if not isinstance(markdown_text, str):
        markdown_text = str(markdown_text)
    json_content = doc.get("json_content") or {}
    html_content = doc.get("html_content") or ""
    text_content = doc.get("text_content") or ""
    if isinstance(html_content, dict):
        html_content = html_content.get("text", "") or html_content.get("content", "") or str(html_content)
    if isinstance(text_content, dict):
        text_content = text_content.get("text", "") or text_content.get("content", "") or str(text_content)

    if not markdown_text and status != "failure":
        raise CorruptedDocumentError(
            f"Docling converted {doc.get('file_name', 'document')} but returned empty markdown. "
            f"Status: {status}, errors: {errors}"
        )

    if status == "failure":
        # Errors may be list[str] or list[dict] — normalize to strings
        error_parts = []
        for e in errors:
            if isinstance(e, dict):
                error_parts.append(e.get("message", "") or e.get("error", "") or str(e))
            else:
                error_parts.append(str(e))
        error_msg = "; ".join(error_parts) if error_parts else "Unknown error"
        raise ConversionError(
            doc.get("file_name", "document"),
            error_msg,
            hint="The document could not be parsed by Docling. Check it is a valid PDF/DOCX/HTML.",
        )

    if status == "partial_success":
        logger.warning("Docling partial success with errors: %s", errors)

    logger.info(
        "Docling conversion complete — status=%s, %d chars markdown, %.1fs",
        status,
        len(markdown_text),
        processing_time,
    )

    return ConversionResult(
        markdown=markdown_text,
        json_content=json_content,
        html_content=html_content,
        text_content=text_content,
        status=status,
        processing_time=processing_time,
        errors=errors,
    )


def close() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        _client.close()
        _client = None
