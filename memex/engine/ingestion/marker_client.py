"""Marker document conversion client — async job-based, crash-proof.

Architecture (permanent fix for converter timeouts/crashes):

    convert_markdown(file_bytes, filename)
      → POST /jobs/{job_id}        (server saves file, spawns subprocess)
      → poll GET /jobs/{job_id}    with backoff until done/failed
      → GET /jobs/{job_id}/result  (markdown or error)

The marker server holds NO models — each conversion runs in an isolated
subprocess with a hard timeout. A crash/OOM kills only that job; the server
survives and the job can be retried. No more "server disconnected" storms.

OOM fallback: When Marker fails with CUDA OOM (common for scanned PDFs on
8GB GPUs), automatically retry via the lightweight OCR service (PP-OCRv6 small).

Error model (typed, from memex.engine.core.errors):
    ConversionTimeoutError   — job exceeded the server-side timeout
    ConversionError          — marker reported success=false
    CorruptedDocumentError   — empty output
    ServiceUnavailableError  — server unreachable/restarting
"""

from __future__ import annotations

import io
import logging
import threading
import time
import uuid
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
    ConversionTimeoutError,
    CorruptedDocumentError,
    ServiceUnavailableError,
)
from memex.engine.ingestion.ocr_client import OcrResult
from memex.engine.utils.gpu_lock import gpu_lock

logger = logging.getLogger("marker-client")

_client: httpx.Client | None = None
_client_lock = threading.Lock()

# Global cap on concurrent in-flight conversions (client-side queue).
_converter_semaphore = threading.BoundedSemaphore(max(1, config.CONVERTER_MAX_CONCURRENT))

# Server-side hard timeout (mirrors JOB_TIMEOUT in docker-compose.yml).
JOB_TIMEOUT_S: float = config.MARKER_TIMEOUT
JOB_POLL_INTERVAL: float = 2.0
JOB_POLL_MAX: float = 15.0


@dataclass
class MarkerResult:
    """Structured output from Marker conversion."""

    markdown: str
    metadata: dict[str, Any] = field(default_factory=dict)
    images: dict[str, str] = field(default_factory=dict)
    status: str = "success"
    processing_time: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "success"


# ── HTTP helpers ─────────────────────────────────────────────────────────────


def _base_url() -> str:
    return config.MARKER_URL.rstrip("/")


def _get_client() -> httpx.Client:
    global _client
    if _client is not None and not _client.is_closed:
        return _client
    with _client_lock:
        if _client is not None and not _client.is_closed:
            return _client
        _client = httpx.Client(
            timeout=httpx.Timeout(config.MARKER_TIMEOUT, connect=10.0),
            limits=httpx.Limits(
                max_connections=4,
                max_keepalive_connections=2,
                keepalive_expiry=30,
            ),
        )
    return _client


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
def _post_transport(url: str, files: dict[str, Any], data: dict[str, Any]) -> httpx.Response:
    """POST to Marker, retrying connection-level failures."""
    client = _get_client()
    with _converter_semaphore:
        return client.post(url, files=files, data=data)


@retry(
    retry=retry_if_exception_type(httpx.HTTPStatusError),
    stop=stop_after_attempt(config.HTTP_MAX_RETRIES),
    wait=wait_exponential(multiplier=config.HTTP_RETRY_BACKOFF, max=10),
    reraise=True,
)
def _post_status(resp: httpx.Response) -> httpx.Response:
    """Retry transient HTTP statuses from the server itself."""
    if resp.status_code in (429, 502, 503, 504):
        resp.raise_for_status()
    resp.raise_for_status()
    return resp


def _submit(file_bytes: bytes, filename: str, mode: str, force_ocr: bool) -> str:
    """Submit a conversion job, returning its job_id."""
    job_id = str(uuid.uuid4())
    url = f"{_base_url()}/jobs/{job_id}"
    files = {"file": (filename, io.BytesIO(file_bytes), "application/octet-stream")}
    data: dict[str, Any] = {
        "output_format": "markdown",
        "mode": mode,
        "force_ocr": "true" if force_ocr else "false",
    }
    try:
        resp = _post_transport(url, files, data)
        resp = _post_status(resp)
        body = resp.json()
        if body.get("status") not in ("pending", "running"):
            raise ConversionError(filename, f"job submission rejected: {body}", cause=None)
        return job_id
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 504:
            raise ConversionTimeoutError(
                filename,
                timeout_s=JOB_TIMEOUT_S,
                hint="Marker submission timed out. Check the marker service is healthy.",
                cause=exc,
            ) from exc
        raise ConversionError(
            filename,
            f"Marker API error {exc.response.status_code}: {exc.response.text[:200]}",
            cause=exc,
        ) from exc
    except httpx.TransportError as exc:
        raise ServiceUnavailableError(
            "Marker",
            f"cannot reach {_base_url()}: {exc}",
            hint="Check that the marker service is running (docker compose up -d marker).",
            cause=exc,
        ) from exc


def _poll(job_id: str, filename: str) -> dict[str, Any]:
    """Poll until the job finishes, with backoff. Returns the result body."""
    url = f"{_base_url()}/jobs/{job_id}"
    client = _get_client()

    deadline = time.monotonic() + JOB_TIMEOUT_S + 60  # server timeout + grace
    interval = JOB_POLL_INTERVAL
    while time.monotonic() < deadline:
        try:
            resp = client.get(url)
            resp.raise_for_status()
            status = resp.json().get("status", "pending")
        except httpx.TransportError as exc:
            # Server restarting mid-job — keep polling (the job is durable).
            logger.debug("poll interrupted for %s: %s", job_id, exc)
            time.sleep(interval)
            interval = min(interval * 1.5, JOB_POLL_MAX)
            continue

        if status == "done":
            result_url = f"{url}/result"
            result_resp = client.get(result_url)
            result_resp.raise_for_status()
            return result_resp.json()

        if status == "failed":
            # Fetch the error from the result file if present.
            try:
                result_resp = client.get(f"{url}/result")
                return result_resp.json()
            except Exception:
                return {"success": False, "error": "conversion failed"}

        time.sleep(interval)
        interval = min(interval * 1.5, JOB_POLL_MAX)

    raise ConversionTimeoutError(
        filename,
        timeout_s=JOB_TIMEOUT_S,
        hint=(
            "Marker conversion exceeded the job timeout. The document may be too "
            "large or the GPU service overloaded. Reduce converter.max_concurrent "
            "or switch converter.marker_mode to 'fast'."
        ),
    )


# ── OOM detection helpers ────────────────────────────────────────────────────

_OOM_PATTERNS = (
    "cuda",
    "out of memory",
    "oom",
    "insufficient memory",
    "allocate",
    "memory allocation failed",
    "runtimeerror",
    "torch.cuda",
)


def _is_oom_error(error_msg: str) -> bool:
    """Detect CUDA OOM or Surya OOM in marker error messages."""
    lower = error_msg.lower()
    return any(pat in lower for pat in _OOM_PATTERNS)


def _ocr_fallback(file_bytes: bytes, filename: str) -> OcrResult:
    """Retry conversion via the lightweight OCR service."""
    from memex.engine.ingestion.ocr_client import convert_with_ocr

    logger.info(
        "Marker OOM — falling back to OCR service for %s",
        filename,
        extra={"stage": "OcrFallback", "source": filename},
    )
    return convert_with_ocr(file_bytes, filename)


# ── Public API ───────────────────────────────────────────────────────────────


def convert_markdown(file_bytes: bytes, filename: str) -> MarkerResult:
    """Convert a document to Markdown via the Marker job API.

    Args:
        file_bytes: Raw file content.
        filename: Original filename (used for extension detection).

    Raises:
        ConversionTimeoutError: If the job exceeds the server timeout.
        ConversionError: On conversion failure (success=false).
        CorruptedDocumentError: If output is empty.
        ServiceUnavailableError: If Marker is unreachable.
    """
    gpu_lock.acquire("marker")
    try:
        job_id = _submit(file_bytes, filename, config.MARKER_MODE, config.MARKER_FORCE_OCR)
        body = _poll(job_id, filename)
    finally:
        gpu_lock.release("marker")

    if not body.get("success", False):
        err = body.get("error", "unknown conversion error")

        # OOM fallback: Marker OOM → retry via OCR service
        if config.OCR_FALLBACK and _is_oom_error(err):
            try:
                ocr_result = _ocr_fallback(file_bytes, filename)
                if ocr_result.ok and ocr_result.markdown.strip():
                    return MarkerResult(
                        markdown=ocr_result.markdown,
                        metadata={"source": "ocr_fallback", "ocr_model": ocr_result.model},
                        status="success",
                        processing_time=ocr_result.processing_time,
                    )
            except Exception as ocr_exc:
                logger.warning(
                    "OCR fallback also failed for %s: %s",
                    filename,
                    ocr_exc,
                    extra={"stage": "OcrFallback", "source": filename},
                )

        raise ConversionError(
            filename,
            err,
            hint=(
                "Marker failed to convert this document. If it is a scanned PDF, "
                "enable converter.marker_force_ocr=true; if it is corrupt, "
                "check the file."
            ),
        )

    markdown = body.get("output", "") or ""
    if not markdown.strip():
        raise CorruptedDocumentError(
            f"Marker converted {filename} but returned empty markdown",
            component="conversion",
        )

    logger.info(
        "Marker conversion complete — %d chars markdown, %d images",
        len(markdown),
        len(body.get("images", {})),
    )
    return MarkerResult(
        markdown=markdown,
        metadata=body.get("metadata", {}),
        images=body.get("images", {}),
    )


def is_marker_available() -> bool:
    """Check whether the Marker service is reachable (server is always thin)."""
    try:
        client = _get_client()
        resp = client.get(f"{_base_url()}/health", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


def close() -> None:
    """Close the singleton client (process shutdown)."""
    global _client
    if _client is not None and not _client.is_closed:
        _client.close()
        _client = None


__all__ = ["MarkerResult", "OcrResult", "close", "convert_markdown", "is_marker_available"]
