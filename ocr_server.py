"""Lightweight OCR service — RapidOCR (PP-OCRv6 ONNX, no PaddlePaddle)."""

from __future__ import annotations

import io
import logging
import os
import time

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger("ocr-server")

app = FastAPI(title="Memex OCR Service", version="0.1.0")

# ── Model registry ──────────────────────────────────────────────────────────
ACTIVE_MODEL: str = os.environ.get("OCR_MODEL", "pp-ocrv6-small")
_models: dict[str, object] = {}


class ModelInfo(BaseModel):
    name: str
    loaded: bool
    vram_mb: int = 0


# ── RapidOCR (PP-OCRv6 ONNX) ───────────────────────────────────────────────
def _load_rapidocr():
    """Load RapidOCR (uses PaddleOCR ONNX models, no PaddlePaddle needed)."""
    try:
        from rapidocr_onnxruntime import RapidOCR

        ocr = RapidOCR()
        logger.info("RapidOCR loaded successfully")
        return ocr
    except Exception as e:
        logger.error("Failed to load RapidOCR: %s", e)
        raise


def _ocr_rapid(image_bytes: bytes) -> dict:
    """Run RapidOCR on image bytes, return structured result."""
    model = _models.get("pp-ocrv6-small")
    if model is None:
        raise RuntimeError("RapidOCR not loaded")

    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)

    result, _elapse = model(img_array)

    texts = []
    total_conf = 0.0
    count = 0
    if result:
        for line in result:
            if len(line) >= 2:
                text, conf = line[1], line[2]
                texts.append(text)
                total_conf += conf
                count += 1

    avg_conf = total_conf / count if count > 0 else 0.0
    return {
        "text": "\n".join(texts),
        "confidence": avg_conf,
        "lines": count,
    }


# ── Health / model management ───────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": ACTIVE_MODEL,
        "loaded_models": list(_models.keys()),
    }


class ModelSwapRequest(BaseModel):
    model: str


@app.post("/model/swap")
async def swap_model(req: ModelSwapRequest):
    global ACTIVE_MODEL
    old = ACTIVE_MODEL
    if req.model not in ("pp-ocrv6-small",):
        raise HTTPException(400, f"Unknown model: {req.model}")
    ACTIVE_MODEL = req.model
    return {"status": "ok", "previous": old, "current": ACTIVE_MODEL}


# ── Convert endpoint ────────────────────────────────────────────────────────
@app.post("/convert")
async def convert(files: list[UploadFile] = File(...)):  # noqa: B008
    start = time.time()
    pages = []

    for i, f in enumerate(files):
        image_bytes = await f.read()
        try:
            result = _ocr_rapid(image_bytes)
            pages.append({"page": i + 1, **result})
        except Exception as e:
            logger.error("OCR failed on page %d: %s", i + 1, e)
            pages.append({"page": i + 1, "text": "", "confidence": 0, "error": str(e)})

    # Build markdown from pages
    markdown_parts = []
    for p in pages:
        if p.get("text"):
            markdown_parts.append(p["text"])
    markdown = "\n\n---\n\n".join(markdown_parts)

    elapsed = time.time() - start
    return {
        "markdown": markdown,
        "pages": pages,
        "model": ACTIVE_MODEL,
        "processing_time": round(elapsed, 2),
    }


# ── Startup: load default model ─────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global _models
    logger.info("Loading OCR model: %s", ACTIVE_MODEL)
    try:
        _models["pp-ocrv6-small"] = _load_rapidocr()
        logger.info("OCR service ready with model: %s", ACTIVE_MODEL)
    except Exception as e:
        logger.error("Failed to load OCR model: %s", e)
        # Service starts anyway — /health will show not-loaded
