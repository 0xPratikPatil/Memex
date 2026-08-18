"""Lightweight OCR service — PP-OCRv6 small (ONNX) with optional LightOnOCR-2-1B."""

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


# ── PP-OCRv6 small (ONNX) ──────────────────────────────────────────────────
def _load_pp_ocrv6_small():
    """Load PP-OCRv6 detection + recognition models via PaddleOCR ONNX."""
    try:
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=False,  # CPU in Docker, GPU via onnxruntime-gpu if available
            show_log=False,
        )
        logger.info("PP-OCRv6 small loaded successfully")
        return ocr
    except Exception as e:
        logger.error("Failed to load PP-OCRv6 small: %s", e)
        raise


def _ocr_pp_ocrv6(image_bytes: bytes) -> dict:
    """Run PP-OCRv6 on image bytes, return structured result."""
    model = _models.get("pp-ocrv6-small")
    if model is None:
        raise RuntimeError("PP-OCRv6 small not loaded")

    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)

    result = model.ocr(img_array, cls=True)  # type: ignore[attr-defined]

    texts = []
    total_conf = 0.0
    count = 0
    if result and result[0]:
        for line in result[0]:
            if line[1]:
                text, conf = line[1]
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
    if req.model not in ("pp-ocrv6-small", "lightonocr-2-1b"):
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
            result = _ocr_pp_ocrv6(image_bytes)
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
        _models["pp-ocrv6-small"] = _load_pp_ocrv6_small()
        logger.info("OCR service ready with model: %s", ACTIVE_MODEL)
    except Exception as e:
        logger.error("Failed to load OCR model: %s", e)
        # Service starts anyway — /health will show not-loaded
