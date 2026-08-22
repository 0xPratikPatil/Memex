"""Lightweight OCR service — RapidOCR (PP-OCRv6 ONNX, no PaddlePaddle).

Accepts both images and PDFs. PDFs are rendered to page images with
pypdfium2 (pure-C types, no GPU) before OCR runs on each page.
"""

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

PDF_MAGIC = b"%PDF-"


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


def _is_pdf(data: bytes) -> bool:
    """Detect PDF by magic bytes (allows leading whitespace)."""
    return data.lstrip()[:5] == PDF_MAGIC


def _pdf_to_pil_pages(pdf_bytes: bytes, scale: float = 2.0):
    """Render each PDF page to a PIL image (scale 2 = ~144 DPI)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_bytes)
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=scale)
            yield bitmap.to_pil()
    finally:
        pdf.close()


def _ocr_pil_image(pil_img) -> dict:
    """Run RapidOCR on a PIL image, return structured result."""
    model = _models.get("pp-ocrv6-small")
    if model is None:
        raise RuntimeError("RapidOCR not loaded")

    img_array = np.array(pil_img.convert("RGB"))

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

    page_no = 0
    for i, f in enumerate(files):
        data = await f.read()
        if not data:
            pages.append({"page": i + 1, "text": "", "confidence": 0, "error": "empty upload"})
            continue

        if _is_pdf(data):
            logger.info("PDF detected (%d bytes, %s) — rendering pages", len(data), f.filename)
            try:
                for pil_img in _pdf_to_pil_pages(data):
                    page_no += 1
                    try:
                        result = _ocr_pil_image(pil_img)
                        pages.append({"page": page_no, **result})
                    except Exception as e:
                        logger.error("OCR failed on page %d: %s", page_no, e)
                        pages.append({"page": page_no, "text": "", "confidence": 0, "error": str(e)})
            except Exception as e:
                logger.error("PDF rendering failed for %s: %s", f.filename, e)
                pages.append({"page": i + 1, "text": "", "confidence": 0, "error": f"PDF rendering failed: {e}"})
        else:
            page_no += 1
            try:
                from PIL import Image

                img = Image.open(io.BytesIO(data)).convert("RGB")
                result = _ocr_pil_image(img)
                pages.append({"page": page_no, **result})
            except Exception as e:
                logger.error("OCR failed on image %d: %s", i + 1, e)
                pages.append({"page": page_no, "text": "", "confidence": 0, "error": str(e)})

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
