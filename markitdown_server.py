"""MarkItDown conversion server — lightweight FastAPI service.

Converts documents to Markdown using Microsoft's MarkItDown library.
CPU-only, no GPU, no models to download. Runs in-process (safe — no GPU state).

Endpoints:
    POST /convert  — accept file bytes, return markdown
    GET  /health   — liveness check
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="MarkItDown Converter")

# Lazy singleton — imported once, reused across requests.
_markitdown_instance = None

# Bound concurrent conversions: each holds a full document in memory and is
# CPU-heavy. Without this, 8+ concurrent requests OOM the container. Also
# keeps the event loop responsive so /health never stalls.
MAX_CONCURRENT = int(os.environ.get("MARKITDOWN_MAX_CONCURRENT", "4"))
_convert_semaphore = asyncio.Semaphore(MAX_CONCURRENT)


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


@app.post("/convert")
async def convert(
    file: UploadFile = File(...),  # noqa: B008
    filename: str = Form(default=""),
) -> JSONResponse:
    """Convert a document to Markdown (CPU work offloaded to threads)."""
    start = time.monotonic()
    name = filename or file.filename or "document"

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="empty file") from None

        async with _convert_semaphore:
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


def _detect_format(filename: str) -> str:
    """Detect file format from extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext or "unknown"
