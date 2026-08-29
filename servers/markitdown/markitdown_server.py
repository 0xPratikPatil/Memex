"""MarkItDown conversion server — lightweight FastAPI service.

Converts documents to Markdown using Microsoft's MarkItDown library.
CPU-only, no GPU, no models to download. Runs in-process (safe — no GPU state).

Concurrency model:
  - MAX_CONCURRENT conversions at a time (default 2 — the container is
    limited to 2 CPUs). Extra requests wait in a FIFO queue exposed via
    GET /queue (current file + pending files), mirroring the OCR service.

Endpoints:
    POST /convert  — accept file bytes, return markdown
    GET  /health   — liveness check
    GET  /queue    — live queue state (current + pending files)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections import deque
from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="MarkItDown Converter")

# Lazy singleton — imported once, reused across requests.
_markitdown_instance = None

# Bound concurrent conversions: each holds a full document in memory and is
# CPU-heavy. Without this, concurrent requests OOM the container. The queue
# makes waits explicit instead of silently stacking threads.
MAX_CONCURRENT = int(os.environ.get("MARKITDOWN_MAX_CONCURRENT", "2"))
_convert_semaphore = asyncio.Semaphore(MAX_CONCURRENT)

# Live queue state (exposed via GET /queue).
# _active_files holds ALL files currently converting (MAX_CONCURRENT of
# them) — a single _current_file clobbers on the second concurrent job.
_active_files: deque[str] = deque()
_pending_files: deque[str] = deque()

# Recently completed conversions — ring buffer so the CLI poller can see
# work that finished between polls.  Each entry is (filename, end_time).
_RECENTLY_COMPLETED: deque[tuple[str, float]] = deque(maxlen=20)
_RECENTLY_COMPLETED_TTL_S = 5.0


def _get_markitdown():
    global _markitdown_instance
    if _markitdown_instance is None:
        from markitdown import MarkItDown

        _markitdown_instance = MarkItDown()
    return _markitdown_instance


def _convert_sync(content: bytes, name: str) -> str:
    """Run MarkItDown synchronously — executed in a worker thread."""
    md = _get_markitdown()
    result = md.convert(BytesIO(content), file_name=name)
    return result.text_content


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/queue")
async def queue_status() -> dict:
    """Live queue state: which files are converting now, which are waiting."""
    now = time.monotonic()
    recent = [
        name
        for name, ts in _RECENTLY_COMPLETED
        if now - ts < _RECENTLY_COMPLETED_TTL_S
    ]
    return {
        "current": _active_files[0] if _active_files else None,
        "active": list(_active_files),
        "pending": list(_pending_files),
        "queued": len(_pending_files),
        "busy": bool(_active_files),
        "max_concurrent": MAX_CONCURRENT,
        "recently_completed": recent,
    }


@app.post("/convert")
async def convert(
    file: UploadFile = File(...),  # noqa: B008
    filename: str = Form(default=""),
) -> JSONResponse:
    """Convert a document to Markdown (CPU work offloaded to threads)."""
    name = filename or file.filename or "document"

    if len(_active_files) >= MAX_CONCURRENT:
        _pending_files.append(name)
        logger.info(
            "conversion busy — queued %s (%d waiting)", name, len(_pending_files)
        )

    async with _convert_semaphore:
        _active_files.append(name)
        with contextlib.suppress(ValueError):
            _pending_files.remove(name)

        start = time.monotonic()
        try:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="empty file") from None

            text = await asyncio.to_thread(_convert_sync, content, name)
            elapsed = time.monotonic() - start

            logger.info(
                "conversion complete",
                extra={"source": name, "chars": len(text), "time": f"{elapsed:.1f}s"},
            )

            return JSONResponse(
                {
                    "success": True,
                    "output": text,
                    "format": _detect_format(name),
                    "processing_time": round(elapsed, 2),
                    "metadata": {},
                }
            )
        except HTTPException:
            raise
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error("conversion failed", extra={"source": name, "error": str(exc)})
            return JSONResponse(
                {"success": False, "error": str(exc), "processing_time": round(elapsed, 2)},
                status_code=500,
            )
        finally:
            with contextlib.suppress(ValueError):
                _active_files.remove(name)
            _RECENTLY_COMPLETED.append((name, time.monotonic()))


def _detect_format(filename: str) -> str:
    """Detect file format from extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext or "unknown"
