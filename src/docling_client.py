"""Docling document conversion client — v1 API.

Uses ``httpx`` for connection pooling and ``tenacity`` for automatic retries
with exponential back-off.  Works both inside Docker and on bare metal.
"""

from __future__ import annotations

import base64
import hashlib
import logging
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

from . import config

logger = logging.getLogger("docling-client")

_client: httpx.Client | None = None


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
    if _client is None or _client.is_closed:
        _client = httpx.Client(
            timeout=httpx.Timeout(config.DOCLING_TIMEOUT, connect=10.0),
            limits=httpx.Limits(
                max_connections=8,
                max_keepalive_connections=4,
                keepalive_expiry=30,
            ),
        )
    return _client


@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
    stop=stop_after_attempt(config.HTTP_MAX_RETRIES),
    wait=wait_exponential(multiplier=config.HTTP_RETRY_BACKOFF, max=10),
    reraise=True,
)
def _post(payload: dict) -> dict:
    client = _get_client()
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if config.DOCLING_API_KEY:
        headers["X-Api-Key"] = config.DOCLING_API_KEY
    resp = client.post(config.DOCLING_URL, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()


def fetch_file_from_server(file_path: str) -> tuple[bytes, str]:
    """Fetch a file from the host file server.

    Args:
        file_path: Absolute path on the host (e.g., /mnt/docs/report.pdf)

    Returns:
        Tuple of (file_bytes, filename)

    Raises:
        FileNotFoundError: If file not found on server
        RuntimeError: If server unreachable
    """
    # Strip leading slash for the URL path
    path = file_path.lstrip("/")
    url = f"{config.FILE_SERVER_URL}/{path}"

    try:
        client = _get_client()
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.content
        filename = Path(file_path).name
        return data, filename
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise FileNotFoundError(f"File not found on server: {file_path}") from None
        raise RuntimeError(f"File server returned HTTP {exc.response.status_code}") from exc
    except httpx.TransportError as exc:
        raise RuntimeError(f"Cannot reach file server at {config.FILE_SERVER_URL}: {exc}") from exc


# ── Conversion options ───────────────────────────────────────────────────────


def _build_options() -> dict[str, Any]:
    """Build Docling v1 conversion options from config."""
    return {
        "from_formats": ["docx", "pptx", "html", "image", "pdf", "md", "csv", "xlsx"],
        "to_formats": ["md", "json"],
        "do_ocr": config.ENABLE_OCR,
        "table_mode": "accurate",
        "do_table_structure": True,
        "include_images": False,
    }


# ── Public API ───────────────────────────────────────────────────────────────


def parse_url(url: str) -> ConversionResult:
    """Fetch a URL via Docling and return structured conversion result."""
    from .services.cache import cache_parse_result, get_cached_parse_result

    file_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    cached = get_cached_parse_result(file_hash)
    if cached is not None:
        logger.info("Docling cache hit for URL: %s", url)
        return ConversionResult(
            markdown=cached["markdown"],
            status=cached.get("status", "success"),
            processing_time=cached.get("processing_time", 0.0),
            errors=cached.get("errors", []),
        )

    payload = {
        "options": _build_options(),
        "sources": [{"kind": "http", "url": url}],
    }

    try:
        data = _post(payload)
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Docling server returned HTTP {exc.response.status_code}: {exc.response.text[:500]}"
        ) from exc
    except httpx.TransportError as exc:
        raise RuntimeError(f"Cannot reach Docling server at {config.DOCLING_URL}: {exc}") from exc

    result = _parse_response(data)
    cache_parse_result(file_hash, {
        "markdown": result.markdown,
        "status": result.status,
        "processing_time": result.processing_time,
        "errors": result.errors,
    })
    return result


def parse_local_file(file_path: str) -> ConversionResult:
    """Fetch a local file from the file server and convert via Docling.

    Args:
        file_path: Absolute path on the host (e.g., /mnt/docs/report.pdf)
    """
    from .services.cache import cache_parse_result, get_cached_parse_result

    file_hash = hashlib.sha256(file_path.encode()).hexdigest()[:16]
    cached = get_cached_parse_result(file_hash)
    if cached is not None:
        logger.info("Docling cache hit for local file: %s", file_path)
        return ConversionResult(
            markdown=cached["markdown"],
            status=cached.get("status", "success"),
            processing_time=cached.get("processing_time", 0.0),
            errors=cached.get("errors", []),
        )

    file_bytes, filename = fetch_file_from_server(file_path)
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
        raise RuntimeError(
            f"Docling server returned HTTP {exc.response.status_code}: {exc.response.text[:500]}"
        ) from exc
    except httpx.TransportError as exc:
        raise RuntimeError(f"Cannot reach Docling server at {config.DOCLING_URL}: {exc}") from exc

    result = _parse_response(data)
    cache_parse_result(file_hash, {
        "markdown": result.markdown,
        "status": result.status,
        "processing_time": result.processing_time,
        "errors": result.errors,
    })
    return result


def parse_file_content(
    file_content_b64: str,
    filename: str,
    source_identifier: str | None = None,
) -> ConversionResult:
    """Send base64-encoded file content to Docling and return structured result."""
    payload = {
        "options": _build_options(),
        "sources": [
            {
                "kind": "file",
                "base64_string": file_content_b64,
                "filename": filename,
            }
        ],
    }

    try:
        data = _post(payload)
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Docling server returned HTTP {exc.response.status_code}: {exc.response.text[:500]}"
        ) from exc
    except httpx.TransportError as exc:
        raise RuntimeError(f"Cannot reach Docling server at {config.DOCLING_URL}: {exc}") from exc

    return _parse_response(data)


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
    json_content = doc.get("json_content") or {}
    html_content = doc.get("html_content") or ""
    text_content = doc.get("text_content") or ""

    if not markdown_text and status != "failure":
        raise ValueError(f"Docling converted the file but returned empty markdown. Status: {status}, errors: {errors}")

    if status == "failure":
        error_msg = "; ".join(errors) if errors else "Unknown error"
        raise RuntimeError(f"Docling conversion failed: {error_msg}")

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
