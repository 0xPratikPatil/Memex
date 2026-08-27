# syntax=docker/dockerfile:1
# ═══════════════════════════════════════════════════════════════════════════════════
# Memex — Marker document conversion service
#
# Replaces Docling: marker-pdf (Surya OCR + layout) served via the official
# marker FastAPI server (/marker/upload) on GPU.
#
# Multi-stage build rules (mirrors the ml-services Dockerfile):
#   • Stages ordered least→most volatile: tooling → deps → model cache → app code
#   • Pinned base images and tools for reproducible builds.
#   • BuildKit cache mounts for fast incremental rebuilds.
#   • Models pre-cached at build time → instant container startup.
#   • Non-root runtime user with explicit UID/GID (1001:1001).
#   • GPU support via NVIDIA Container Toolkit.
# ═══════════════════════════════════════════════════════════════════════════════════

ARG UV_VERSION=0.6.0

# ═══════════════════════════════════════════════════════════════════════════════════
# Stage 0 — uv-tool : Pinned uv package manager (most stable, rarely changes)
# ═══════════════════════════════════════════════════════════════════════════════════
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-tool

# ═══════════════════════════════════════════════════════════════════════════════════
# Stage 1 — python-base : OS + system deps + Python virtual environment
# ═══════════════════════════════════════════════════════════════════════════════════
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime AS python-base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# ── System dependencies ──────────────────────────────────────────────────────
# curl = healthcheck, ca-certificates = TLS for HuggingFace model downloads,
# libgl/libglib = OpenCV (marker image handling).
# gcc/g++ ARE runtime deps — required by Triton for CUDA kernel JIT compilation
# when the models load on GPU (same requirement as the ml-services image).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      gcc \
      g++ \
      libgl1 \
      libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── uv package manager (pinned version) ──────────────────────────────────────
COPY --from=uv-tool /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# ── Python virtual environment ───────────────────────────────────────────────
RUN uv venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# ── Install Python packages (BuildKit cache mount for uv) ────────────────────
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
      "marker-pdf>=1.0,<2" \
      "uvicorn[standard]>=0.30,<1" \
      "fastapi[standard]>=0.115,<1" \
      "python-multipart>=0.0.9,<1" \
      "httpx>=0.27,<1"

# ═══════════════════════════════════════════════════════════════════════════════════
# Stage 2 — model-cache : Pre-cache marker models at build time
# ═══════════════════════════════════════════════════════════════════════════════════
FROM python-base AS model-cache

ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    # Surya caches models under $HOME/.cache by default — pin to /app/.cache
    # so the runtime COPY --from finds them in one place.
    HOME=/app

# ── Pre-cache marker models ─────────────────────────────────────────────────
# Runs with host networking so DNS works during build (host uses the
# systemd-resolved stub 127.0.0.53, unreachable inside the default build
# network). Models download to /app/.cache and are copied into runtime —
# without this the container downloads 1.35GB+ on every fresh start.
# NOTE: this stage must NOT fail silently — a failed pre-cache defeats the
# entire purpose (instant startup, no runtime downloads).
RUN python -c \
    "from marker.models import create_model_dict; \
     m = create_model_dict(); \
     print('Marker models cached: %s' % sorted(m.keys()))"

# ── Pre-cache the rendering font (marker downloads it at runtime otherwise) ──
RUN sh -c "mkdir -p /opt/venv/lib/python3.11/site-packages/static/fonts && \
    curl -sfL -o /opt/venv/lib/python3.11/site-packages/static/fonts/GoNotoCurrent-Regular.ttf \
      https://models.datalab.to/artifacts/GoNotoCurrent-Regular.ttf && \
    echo 'Font pre-cached: \$(ls -la /opt/venv/lib/python3.11/site-packages/static/fonts/ | tail -1)'"

# ═══════════════════════════════════════════════════════════════════════════════════
# Stage 3 — runtime : Minimal production image (most volatile, changes with code)
# ═══════════════════════════════════════════════════════════════════════════════════
FROM python-base AS runtime

ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HOME=/app \
    TORCH_DEVICE=cuda \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Carry the pre-cached models from the build stage ─────────────────────────
COPY --from=model-cache --chown=1001:1001 /app/.cache /app/.cache
# ── Carry the pre-cached font ────────────────────────────────────────────────
COPY --from=model-cache --chown=1001:1001 /opt/venv/lib/python3.11/site-packages/static /opt/venv/lib/python3.11/site-packages/static

# ── Non-root user (explicit UID/GID for host volume compatibility) ───────────
RUN groupadd -g 1001 -r appgroup && \
    useradd -u 1001 -r -g appgroup -d /app -s /sbin/nologin appuser && \
    mkdir -p /app/.cache /app/uploads && \
    # Marker writes fonts/artifacts into the package static dir at runtime —
    # pre-create and chown so the non-root user can write there.
    mkdir -p /opt/venv/lib/python3.11/site-packages/static/fonts \
             /opt/venv/lib/python3.11/site-packages/conversion_results \
             /opt/venv/lib/python3.11/site-packages/debug_data && \
    chown -R appuser:appgroup /app /opt/venv/lib/python3.11/site-packages/static /opt/venv/lib/python3.11/site-packages/conversion_results /opt/venv/lib/python3.11/site-packages/debug_data

# ── Job server + conversion worker (thin server, isolated subprocess worker) ──
COPY servers/marker/marker_server.py /app/marker_server.py
COPY servers/marker/convert_one.py /app/convert_one.py
COPY servers/marker/converter_helpers.py /app/converter_helpers.py
RUN mkdir -p /app/jobs && chown -R appuser:appgroup /app/jobs

USER appuser

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD ["curl", "-sf", "http://localhost:5001/health"] || exit 1

ENTRYPOINT ["python", "-m", "uvicorn", "marker_server:app", "--host", "0.0.0.0", "--port", "5001", "--workers", "1"]
