"""MarkItDown conversion server — lightweight FastAPI service.

Converts documents to Markdown using Microsoft's MarkItDown library.
CPU-only, no GPU, no models to download. Runs in-process (safe — no GPU state).

Endpoints:
    POST /convert  — accept file bytes, return markdown
    GET  /health   — liveness check
"""

from __future__ import annotations

import logging
import time
from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="MarkItDown Converter")

# Lazy singleton — imported once, reused across requests.
_markitdown_instance = None


def _get_markitdown():
    global _markitdown_instance
    if _markitdown_instance is None:
        from markitdown import MarkItDown

        _markitdown_instance = MarkItDown()
    return _markitdown_instance


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/convert")
async def convert(
    file: UploadFile = File(...),  # noqa: B008
    filename: str = Form(default=""),
) -> JSONResponse:
    """Convert a document to Markdown."""
    start = time.monotonic()
    name = filename or file.filename or "document"

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="empty file") from None

        md = _get_markitdown()
        result = md.convert(BytesIO(content), file_name=name)
        elapsed = time.monotonic() - start

        logger.info(
            "conversion complete",
            extra={"source": name, "chars": len(result.text_content), "time": f"{elapsed:.1f}s"},
        )

        return JSONResponse(
            {
                "success": True,
                "output": result.text_content,
                "format": _detect_format(name),
                "processing_time": round(elapsed, 2),
                "metadata": getattr(result, "metadata", {}) or {},
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
