"""MarkItDown conversion service for Memex RAG."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Memex MarkItDown Service", version="0.1.0")


class ConvertResponse(BaseModel):
    markdown: str
    filename: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/convert", response_model=ConvertResponse)
async def convert_file(file: UploadFile = File(...)):  # noqa: B008
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    try:
        from markitdown import MarkItDown

        md = MarkItDown()
    except ImportError as err:
        raise HTTPException(status_code=503, detail="MarkItDown library not available") from err

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = md.convert(tmp_path)
        logger.info("Converted %s (%d chars)", file.filename, len(result.text_content))
        return ConvertResponse(markdown=result.text_content, filename=file.filename)
    except Exception as exc:
        logger.error("Conversion failed for %s: %s", file.filename, exc)
        raise HTTPException(status_code=500, detail="Conversion failed") from exc
    finally:
        os.unlink(tmp_path)
