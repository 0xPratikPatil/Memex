# syntax=docker/dockerfile:1
# ═══════════════════════════════════════════════════════════════════════════════════
# Memex — OCR service (multi-model, auto-detect GPU)
#
# Two build targets, one Dockerfile:
#   rapidocr — pp-ocrv6-medium (RapidOCR + onnxruntime-gpu, auto CPU fallback)
#   vlm      — all 3 models (RapidOCR + Granite-Docling + LightOnOCR, ~8GB)
#
# Build:
#   docker build --target rapidocr -t memex-ocr -f ocr.Dockerfile .
#   docker build --target vlm      -t memex-ocr:vlm -f ocr.Dockerfile .
#
# GPU auto-detection: onnxruntime-gpu uses CUDA when available, falls back to
# CPU automatically. No config needed — just mount GPU with NVIDIA Container Toolkit.
#
# Multi-stage rules:
#   • Stages ordered least→most volatile: tooling → base → builders → runtime
#   • BuildKit cache mounts for fast incremental rebuilds.
#   • Models pre-cached at build time → instant container startup.
#   • Non-root runtime user (1001:1001).
# ═══════════════════════════════════════════════════════════════════════════════════

ARG UV_VERSION=0.6.0

# ═══════════════════════════════════════════════════════════════════════════════════
# Stage 0 — uv-tool : Pinned uv package manager
# ═══════════════════════════════════════════════════════════════════════════════════
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-tool

# ═══════════════════════════════════════════════════════════════════════════════════
# Stage 1 — base : CUDA + cuDNN + Python + system deps
# ═══════════════════════════════════════════════════════════════════════════════════
# cudnn-runtime tag bundles CUDA 12.4 runtime + cuBLAS + cuDNN 9 — the exact
# libs onnxruntime-gpu needs (libcublasLt.so.12, libcudnn.so.9).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      gcc \
      g++ \
      libgl1 \
      libglib2.0-0 \
      python3-full \
      python3-pip \
      python3-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=uv-tool /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN uv venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# ═══════════════════════════════════════════════════════════════════════════════════
# Stage 2a — rapidocr-builder : onnxruntime-gpu + RapidOCR (auto CPU fallback)
# ═══════════════════════════════════════════════════════════════════════════════════
FROM base AS rapidocr-builder

# onnxruntime-gpu auto-detects CUDA at runtime.
# If no GPU → falls back to CPU automatically. Zero config.
# ``rapidocr`` (>=3.9) ships REAL PP-OCRv6 models (det/rec tiers:
# tiny/small/medium). Models are downloaded and pre-cached below.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
      "uvicorn[standard]>=0.30,<1" \
      "fastapi[standard]>=0.115,<1" \
      "python-multipart>=0.0.9,<1" \
      "httpx>=0.27,<1" \
      "pypdfium2>=4,<5" \
      "Pillow>=10,<11" \
      "numpy>=1.26,<3" \
      "onnxruntime-gpu>=1.21,<2" \
      "rapidocr>=3.9,<4" \
    && pip uninstall -y onnxruntime 2>/dev/null || true

# Pre-cache PP-OCRv6 small + medium (det + rec + cls) at build time —
# first load at runtime never touches the network.
ENV RAPIDOCR_MODELS_DIR=/models/rapidocr

RUN python - <<'PYEOF' \
    || echo "PP-OCRv6 pre-cache skipped"
import os
from rapidocr import ModelType, RapidOCR

os.makedirs("/models/rapidocr", exist_ok=True)
for tier in (ModelType.SMALL, ModelType.MEDIUM):
    RapidOCR(
        params={
            "Det.model_type": tier,
            "Rec.model_type": tier,
            "Global.model_root_dir": "/models/rapidocr",
        }
    )
print("PP-OCRv6 small+medium cached")
PYEOF

# ═══════════════════════════════════════════════════════════════════════════════════
# Stage 2b — vlm-builder : PyTorch + transformers + all 3 OCR backends
# ═══════════════════════════════════════════════════════════════════════════════════
FROM base AS vlm-builder

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
      --extra-index-url https://download.pytorch.org/whl/cu124 \
      "torch>=2.4,<3" \
      "onnxruntime-gpu>=1.21,<2" \
    && uv pip install \
      "uvicorn[standard]>=0.30,<1" \
      "fastapi[standard]>=0.115,<1" \
      "python-multipart>=0.0.9,<1" \
      "httpx>=0.27,<1" \
      "pypdfium2>=4,<5" \
      "Pillow>=10,<11" \
      "numpy>=1.26,<3" \
      "rapidocr>=3.9,<4" \
      "transformers>=4.45,<5" \
      "accelerate>=1.0,<2" \
    && pip uninstall -y onnxruntime 2>/dev/null || true

# Pre-cache PP-OCRv6 small + medium (det + rec + cls) at build time.
ENV RAPIDOCR_MODELS_DIR=/models/rapidocr

RUN python - <<'PYEOF' \
    || echo "PP-OCRv6 pre-cache skipped"
import os
from rapidocr import ModelType, RapidOCR

os.makedirs("/models/rapidocr", exist_ok=True)
for tier in (ModelType.SMALL, ModelType.MEDIUM):
    RapidOCR(
        params={
            "Det.model_type": tier,
            "Rec.model_type": tier,
            "Global.model_root_dir": "/models/rapidocr",
        }
    )
print("PP-OCRv6 small+medium cached")
PYEOF

ENV HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface

RUN python -c \
    "from transformers import AutoModelForCausalLM, AutoProcessor; \
     AutoProcessor.from_pretrained('ibm-granite/granite-docling-258m', trust_remote_code=True); \
     AutoModelForCausalLM.from_pretrained('ibm-granite/granite-docling-258m', trust_remote_code=True); \
     print('Granite-Docling-258M cached')" || echo "Granite-Docling pre-cache skipped"

RUN python -c \
    "from transformers import AutoModelForCausalLM, AutoProcessor; \
     AutoProcessor.from_pretrained('LightOnAI/lighton-ocr-2-1b', trust_remote_code=True); \
     AutoModelForCausalLM.from_pretrained('LightOnAI/lighton-ocr-2-1b', torch_dtype='auto', trust_remote_code=True); \
     print('LightOnOCR-2-1B cached')" || echo "LightOnOCR pre-cache skipped"

# ═══════════════════════════════════════════════════════════════════════════════════
# Stage 3a — rapidocr runtime : GPU-auto (~1.2GB)
# Target: docker build --target rapidocr -t memex-ocr -f ocr.Dockerfile .
# ═══════════════════════════════════════════════════════════════════════════════════
FROM base AS rapidocr

ENV OCR_MODEL=pp-ocrv6-medium \
    OCR_RENDER_SCALE=1.5 \
    OCR_LIMIT_SIDE_LEN=1280 \
    OCR_IDLE_UNLOAD_S=300 \
    RAPIDOCR_MODELS_DIR=/models/rapidocr

RUN groupadd -g 1001 -r appgroup && \
    useradd -u 1001 -r -g appgroup -d /app -s /sbin/nologin appuser && \
    mkdir -p /app /models/rapidocr && \
    chown -R appuser:appgroup /app /models/rapidocr

COPY --from=rapidocr-builder --chown=1001:1001 /opt/venv /opt/venv
COPY --from=rapidocr-builder --chown=1001:1001 /models/rapidocr /models/rapidocr
COPY --chown=1001:1001 servers/ocr/ocr_server.py /app/ocr_server.py

USER 1001

EXPOSE 5004

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD ["curl", "-sf", "http://localhost:5004/health"]

STOPSIGNAL SIGTERM

LABEL org.opencontainers.image.title="Memex OCR" \
      org.opencontainers.image.description="PP-OCRv6 small/medium via RapidOCR + onnxruntime-gpu (auto CPU/GPU)" \
      org.opencontainers.image.source="https://github.com/0xPratikPatil/Memex"

CMD ["uvicorn", "ocr_server:app", "--host", "0.0.0.0", "--port", "5004"]

# ═══════════════════════════════════════════════════════════════════════════════════
# Stage 3b — vlm runtime : Full 3-model image (~8GB)
# Target: docker build --target vlm -t memex-ocr:vlm -f ocr.Dockerfile .
# ═══════════════════════════════════════════════════════════════════════════════════
FROM base AS vlm

ENV OCR_MODEL=pp-ocrv6-medium \
    OCR_RENDER_SCALE=1.5 \
    OCR_LIMIT_SIDE_LEN=1280 \
    OCR_IDLE_UNLOAD_S=300 \
    RAPIDOCR_MODELS_DIR=/models/rapidocr \
    HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface

RUN groupadd -g 1001 -r appgroup && \
    useradd -u 1001 -r -g appgroup -d /app -s /sbin/nologin appuser && \
    mkdir -p /app /models/huggingface /models/rapidocr && \
    chown -R appuser:appgroup /app /models/huggingface /models/rapidocr

COPY --from=vlm-builder --chown=1001:1001 /opt/venv /opt/venv
COPY --from=vlm-builder --chown=1001:1001 /models/huggingface /models/huggingface
COPY --from=vlm-builder --chown=1001:1001 /models/rapidocr /models/rapidocr
COPY --chown=1001:1001 servers/ocr/ocr_server.py /app/ocr_server.py

USER 1001

EXPOSE 5004

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["curl", "-sf", "http://localhost:5004/health"]

STOPSIGNAL SIGTERM

LABEL org.opencontainers.image.title="Memex OCR (VLM)" \
      org.opencontainers.image.description="Multi-model OCR: RapidOCR + Granite-Docling + LightOnOCR" \
      org.opencontainers.image.source="https://github.com/0xPratikPatil/Memex"

CMD ["uvicorn", "ocr_server:app", "--host", "0.0.0.0", "--port", "5004"]
