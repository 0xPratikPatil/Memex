"""OCR service — multi-model backends with GPU/CPU auto-detection.

Backends, exactly one active at a time:
  pp-ocrv6-small       RapidOCR PP-OCRv6 small  (det+rec ONNX, ~90MB)
  pp-ocrv6-medium      RapidOCR PP-OCRv6 medium (det+rec ONNX, ~130MB)
  granite-docling-258m Granite-Docling-258M VLM (~600MB fp16)
  lightonocr-2-1b      LightOnOCR-2-1B VLM (~4.2GB fp16)

The RapidOCR tiers load the REAL PP-OCRv6 models (tiny/small/medium)
from the ``rapidocr`` package — pre-cached in the Docker image at build
time, no network needed at runtime. CUDA is preferred, with automatic
CPU fallback.

PDFs are rendered to page images via pypdfium2, then the active backend
reads each page image and returns extracted text.

One OCR job runs at a time — further requests wait in a queue that is
exposed via GET /queue (current file + pending files).

Config via environment:
  OCR_MODEL            initial model name (default: pp-ocrv6-medium)
  OCR_RENDER_SCALE     PDF render scale (default: 1.5)
  OCR_LIMIT_SIDE_LEN   max page-image side in px (default: 1280)
  OCR_IDLE_UNLOAD_S    seconds of idle before unloading the model (default: 300)
  RAPIDOCR_MODELS_DIR  model cache dir (default: /models/rapidocr)
"""

from __future__ import annotations

import abc
import asyncio
import contextlib
import io
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger("ocr-server")

app = FastAPI(title="Memex OCR Service", version="0.3.0")

# ── Env config ───────────────────────────────────────────────────────────────
ACTIVE_MODEL: str = os.environ.get("OCR_MODEL", "pp-ocrv6-medium")
RENDER_SCALE: float = float(os.environ.get("OCR_RENDER_SCALE", "1.5"))
LIMIT_SIDE_LEN: int = int(os.environ.get("OCR_LIMIT_SIDE_LEN", "1280"))
IDLE_UNLOAD_S: float = float(os.environ.get("OCR_IDLE_UNLOAD_S", "300"))
RAPIDOCR_MODELS_DIR: str = os.environ.get("RAPIDOCR_MODELS_DIR", "/models/rapidocr")

PDF_MAGIC = b"%PDF-"
_last_request_time: float = 0.0

# ── Concurrency: 1 request at a time (GPU OCR is not thread-safe) ────────────
# Requests beyond the first wait in a queue. MarkItDown (separate service)
# stays fully parallel — only this OCR container is serialized.
_ocr_semaphore = asyncio.Semaphore(1)
_current_file: str | None = None
_pending_files: deque[str] = deque()

# Recently completed conversions — ring buffer so the CLI poller can see
# work that finished between polls.  Each entry is (filename, end_time).
_RECENTLY_COMPLETED: deque[tuple[str, float]] = deque(maxlen=20)
_RECENTLY_COMPLETED_TTL_S = 5.0


# ── Backend protocol ────────────────────────────────────────────────────────
class OcrBackend(abc.ABC):
    @abc.abstractmethod
    def load(self) -> None: ...

    @abc.abstractmethod
    def ocr_pil_image(self, pil_img) -> dict: ...

    @abc.abstractmethod
    def unload(self) -> None: ...

    @property
    @abc.abstractmethod
    def vram_mb(self) -> int: ...

    @property
    @abc.abstractmethod
    def provider(self) -> str: ...


@dataclass
class BackendState:
    backend: OcrBackend
    name: str
    loaded: bool = False


_backend_registry: dict[str, OcrBackend] = {}
_current: BackendState | None = None


# ── RapidOCR backend (PP-OCRv6 ONNX) ────────────────────────────────────────
# Uses the ``rapidocr`` package (>=3.9) which ships PP-OCRv6 det/rec tiers
# (tiny/small/medium). Models are pre-cached in the Docker image under
# RAPIDOCR_MODELS_DIR — first load never touches the network.
class RapidOcrBackend(OcrBackend):
    def __init__(self, tier: str):
        self._tier = tier
        self._ocr = None
        self._provider = "cpu"

    def load(self) -> None:
        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                self._provider = "cuda"
                logger.info("CUDA available — using GPU for OCR")
            else:
                self._provider = "cpu"
                logger.info("CUDA not available — using CPU for OCR")
        except ImportError:
            self._provider = "cpu"

        from rapidocr import ModelType, RapidOCR

        tier_enum = {
            "tiny": ModelType.TINY,
            "small": ModelType.SMALL,
            "medium": ModelType.MEDIUM,
        }[self._tier]

        params = {
            "Det.model_type": tier_enum,
            "Rec.model_type": tier_enum,
            "EngineConfig.onnxruntime.use_cuda": self._provider == "cuda",
        }
        if RAPIDOCR_MODELS_DIR:
            params["Global.model_root_dir"] = str(Path(RAPIDOCR_MODELS_DIR))

        try:
            self._ocr = RapidOCR(params=params)
            self._provider = self._detect_actual_provider()
            logger.info(
                "RapidOCR loaded: PP-OCRv6 %s (provider=%s)", self._tier, self._provider
            )
        except Exception as exc:
            if self._provider == "cuda":
                logger.warning("GPU init failed (%s), falling back to CPU", exc)
                self._provider = "cpu"
                params["EngineConfig.onnxruntime.use_cuda"] = False
                self._ocr = RapidOCR(params=params)
                self._provider = self._detect_actual_provider()
                logger.info(
                    "RapidOCR loaded: PP-OCRv6 %s (provider=cpu)", self._tier
                )
            else:
                raise

    def _detect_actual_provider(self) -> str:
        """Read the provider from the live ORT session (handles missing
        CUDA libs where CUDA EP is listed but fails to initialize)."""
        try:
            session = self._ocr.text_det.session
            providers = session.session.get_providers()
            if any("CUDA" in p for p in providers):
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def ocr_pil_image(self, pil_img) -> dict:
        img_array = np.array(pil_img.convert("RGB"))
        result = self._ocr(img_array)
        texts = list(result.txts or [])
        scores = list(result.scores or [])
        avg_conf = sum(scores) / len(scores) if scores else 0.0
        return {"text": "\n".join(texts), "confidence": avg_conf, "lines": len(texts)}

    def unload(self) -> None:
        self._ocr = None
        self._provider = "cpu"

    @property
    def vram_mb(self) -> int:
        return 700 if self._provider == "cuda" else 0

    @property
    def provider(self) -> str:
        return self._provider


# ── Granite-Docling-258M VLM backend ────────────────────────────────────────
class GraniteDoclingBackend(OcrBackend):
    def __init__(self):
        self._model = None
        self._processor = None
        self._device = "cpu"

    def load(self) -> None:
        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        from transformers import AutoModelForCausalLM, AutoProcessor

        model_id = "ibm-granite/granite-docling-258m"
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda:0" if self._device == "cuda" else "cpu",
            trust_remote_code=True,
        )
        logger.info("Granite-Docling-258M loaded (device=%s)", self._device)

    def ocr_pil_image(self, pil_img) -> dict:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": "Transcribe the text in this document image."},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=4096)

        new_tokens = output[0, inputs["input_ids"].shape[-1] :]
        text = self._processor.decode(new_tokens, skip_special_tokens=True)
        return {"text": text.strip(), "confidence": 0.9, "lines": text.count("\n") + 1}

    def unload(self) -> None:
        import torch

        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._device = "cpu"

    @property
    def vram_mb(self) -> int:
        return 600 if self._device == "cuda" else 0

    @property
    def provider(self) -> str:
        return self._device


# ── LightOnOCR-2-1B VLM backend ─────────────────────────────────────────────
class LightOnOcrBackend(OcrBackend):
    def __init__(self):
        self._model = None
        self._processor = None
        self._device = "cpu"

    def load(self) -> None:
        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        from transformers import AutoModelForCausalLM, AutoProcessor

        model_id = "LightOnAI/lighton-ocr-2-1b"
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda:0" if self._device == "cuda" else "cpu",
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        dtype = "fp16" if self._device == "cuda" else "fp32"
        logger.info("LightOnOCR-2-1B loaded (device=%s, dtype=%s)", self._device, dtype)

    def ocr_pil_image(self, pil_img) -> dict:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": "Transcribe the text in this document image."},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=4096)

        new_tokens = output[0, inputs["input_ids"].shape[-1] :]
        text = self._processor.decode(new_tokens, skip_special_tokens=True)
        return {"text": text.strip(), "confidence": 0.85, "lines": text.count("\n") + 1}

    def unload(self) -> None:
        import torch

        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._device = "cpu"

    @property
    def vram_mb(self) -> int:
        return 4200 if self._device == "cuda" else 0

    @property
    def provider(self) -> str:
        return self._device


# ── Registry initialization ──────────────────────────────────────────────────
_backend_registry = {
    "pp-ocrv6-small": RapidOcrBackend(tier="small"),
    "pp-ocrv6-medium": RapidOcrBackend(tier="medium"),
    "granite-docling-258m": GraniteDoclingBackend(),
    "lightonocr-2-1b": LightOnOcrBackend(),
}


def _load_model(name: str) -> None:
    global _current
    if name not in _backend_registry:
        raise ValueError(f"Unknown model: {name}")
    backend = _backend_registry[name]
    backend.load()
    _current = BackendState(backend=backend, name=name, loaded=True)
    logger.info("Loaded model: %s", name)


def _unload_model() -> None:
    global _current
    if _current is not None and _current.loaded:
        _current.backend.unload()
        logger.info("Unloaded model: %s", _current.name)
        _current = None


# ── PDF / image helpers ──────────────────────────────────────────────────────
def _is_pdf(data: bytes) -> bool:
    return data.lstrip()[:5] == PDF_MAGIC


def _pdf_to_pil_pages(pdf_bytes: bytes):
    """Render each PDF page to a PIL image with scale + side-length cap."""
    import pypdfium2 as pdfium
    from PIL import Image

    pdf = pdfium.PdfDocument(pdf_bytes)
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=RENDER_SCALE)
            pil = bitmap.to_pil()

            # Cap side length to limit VRAM / CPU memory
            w, h = pil.size
            max_side = max(w, h)
            if max_side > LIMIT_SIDE_LEN:
                ratio = LIMIT_SIDE_LEN / max_side
                pil = pil.resize(
                    (int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS
                )

            yield pil
    finally:
        pdf.close()


def _ocr_pil_image(pil_img) -> dict:
    """Run the active backend on a PIL image."""
    if _current is None or not _current.loaded:
        raise RuntimeError("No OCR model loaded")
    return _current.backend.ocr_pil_image(pil_img)


def _process_pdf_bytes(data: bytes) -> list[dict]:
    """Render PDF pages to images and OCR each — CPU/GPU-bound, runs in a thread."""
    pages: list[dict] = []
    for page_no, pil_img in enumerate(_pdf_to_pil_pages(data), start=1):
        try:
            result = _ocr_pil_image(pil_img)
            pages.append({"page": page_no, **result})
        except Exception as e:
            logger.error("OCR failed on page %d: %s", page_no, e)
            pages.append({"page": page_no, "text": "", "confidence": 0, "error": str(e)})
    return pages


def _process_image_bytes(data: bytes) -> dict:
    """OCR a single image — runs in a thread."""
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("RGB")
    return _ocr_pil_image(img)


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    model_name = _current.name if _current else ACTIVE_MODEL
    provider = _current.backend.provider if _current and _current.loaded else "none"
    loaded = _current.loaded if _current else False
    vram = _current.backend.vram_mb if _current and _current.loaded else 0
    return {
        "status": "ok",
        "model": model_name,
        "provider": provider,
        "loaded": loaded,
        "vram_mb": vram,
    }


class ModelSwapRequest(BaseModel):
    model: str


@app.post("/model/swap")
async def swap_model(req: ModelSwapRequest):
    global _current
    if req.model not in _backend_registry:
        raise HTTPException(400, f"Unknown model: {req.model}. Valid: {list(_backend_registry.keys())}")

    old_name = _current.name if _current else None
    _unload_model()
    try:
        _load_model(req.model)
    except Exception as exc:
        raise HTTPException(500, f"Failed to load {req.model}: {exc}") from exc

    return {
        "status": "ok",
        "previous": old_name,
        "current": req.model,
        "provider": _current.backend.provider,
        "vram_mb": _current.backend.vram_mb,
    }


@app.get("/queue")
async def queue_status():
    """Live queue state: which file is being OCR'd now, which are waiting."""
    now = time.monotonic()
    recent = [
        name
        for name, ts in _RECENTLY_COMPLETED
        if now - ts < _RECENTLY_COMPLETED_TTL_S
    ]
    return {
        "current": _current_file,
        "pending": list(_pending_files),
        "queued": len(_pending_files),
        "busy": _ocr_semaphore.locked(),
        "model": _current.name if _current and _current.loaded else ACTIVE_MODEL,
        "provider": _current.backend.provider if _current and _current.loaded else "none",
        "recently_completed": recent,
    }


@app.post("/convert")
async def convert(files: list[UploadFile] = File(...)):  # noqa: B008
    global _last_request_time, _current_file

    if _current is None or not _current.loaded:
        raise HTTPException(503, f"No OCR model loaded (active: {ACTIVE_MODEL})")

    filenames = [f.filename or f"file{i}" for i, f in enumerate(files)]
    label = ", ".join(filenames)

    if _ocr_semaphore.locked():
        _pending_files.extend(filenames)
        logger.info(
            "OCR busy — queued [%s] (%d waiting behind current job)",
            label,
            len(_pending_files),
        )

    async with _ocr_semaphore:
        _current_file = filenames[0] if len(filenames) == 1 else label
        for fname in filenames:
            with contextlib.suppress(ValueError):
                _pending_files.remove(fname)
        _last_request_time = time.time()

        start = time.time()
        pages: list[dict] = []

        page_no = 0
        for i, f in enumerate(files):
            data = await f.read()
            if not data:
                pages.append({"page": i + 1, "text": "", "confidence": 0, "error": "empty upload"})
                continue

            if _is_pdf(data):
                logger.info("PDF detected (%d bytes, %s) — rendering pages", len(data), f.filename)
                try:
                    pdf_pages = await asyncio.to_thread(_process_pdf_bytes, data)
                    for p in pdf_pages:
                        p["page"] = page_no + p["page"]
                    page_no += len(pdf_pages)
                    pages.extend(pdf_pages)
                except Exception as e:
                    logger.error("PDF rendering failed for %s: %s", f.filename, e)
                    page_no += 1
                    pages.append({"page": page_no, "text": "", "confidence": 0, "error": f"PDF rendering failed: {e}"})
            else:
                page_no += 1
                try:
                    result = await asyncio.to_thread(_process_image_bytes, data)
                    pages.append({"page": page_no, **result})
                except Exception as e:
                    logger.error("OCR failed on image %d: %s", i + 1, e)
                    pages.append({"page": page_no, "text": "", "confidence": 0, "error": str(e)})

        markdown_parts = []
        for p in pages:
            if p.get("text"):
                markdown_parts.append(p["text"])
        markdown = "\n\n---\n\n".join(markdown_parts)

        elapsed = time.time() - start
        _current_file = None
        _RECENTLY_COMPLETED.append((filenames[0] if len(filenames) == 1 else label, time.monotonic()))
        return {
            "markdown": markdown,
            "pages": pages,
            "model": _current.name if _current else ACTIVE_MODEL,
            "provider": _current.backend.provider if _current else "none",
            "processing_time": round(elapsed, 2),
        }


# ── Idle unload background task ──────────────────────────────────────────────
async def _idle_unload_loop():
    """Unload the active model after IDLE_UNLOAD_S of inactivity.

    pp-ocrv6-small stays resident (≤700MB, always fits alongside Ollama).
    """
    global _last_request_time
    while True:
        await asyncio.sleep(30)
        if _current is None or not _current.loaded:
            continue
        if _current.name in ("pp-ocrv6-small", "pp-ocrv6-medium"):
            continue
        if _last_request_time == 0.0:
            continue
        idle = time.time() - _last_request_time
        if idle >= IDLE_UNLOAD_S:
            logger.info("Model %s idle for %.0fs — unloading", _current.name, idle)
            _unload_model()


_idle_task: asyncio.Task | None = None


@app.on_event("startup")
async def startup():
    global _idle_task
    logger.info(
        "Loading OCR model: %s (render_scale=%.1f, limit_side=%d, idle_unload=%ds)",
        ACTIVE_MODEL,
        RENDER_SCALE,
        LIMIT_SIDE_LEN,
        IDLE_UNLOAD_S,
    )
    try:
        _load_model(ACTIVE_MODEL)
    except Exception as e:
        logger.error("Failed to load OCR model %s: %s — service starts with no model", ACTIVE_MODEL, e)

    _idle_task = asyncio.create_task(_idle_unload_loop())
