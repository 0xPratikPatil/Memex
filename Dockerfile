# syntax=docker/dockerfile:1
# ═══════════════════════════════════════════════════════════════════════════════════
# Memex — ML Services (sparse BM25 embeddings + cross-encoder / causal-LM reranker)
#
# Multi-stage build rules:
#   • Stages ordered least→most volatile: tooling → deps → model cache → app code
#   • Pinned base images and tools for reproducible builds.
#   • BuildKit cache mounts for fast incremental rebuilds.
#   • Model pre-caching at build time → instant container startup.
#   • Non-root runtime user with explicit UID/GID (1001:1001).
#   • COPY --link on all cross-stage copies (no intermediate layers in final image).
#   • GPU support via NVIDIA Container Toolkit.
# ═══════════════════════════════════════════════════════════════════════════════════

# ── Global ARGs (available in FROM but re-declared per stage for RUN access) ──────
ARG SPARSE_MODEL=Qdrant/bm25
ARG RERANK_MODEL=Qwen/Qwen3-Reranker-0.6B
ARG RERANK_MODEL_FALLBACK=BAAI/bge-reranker-base
ARG RERANK_TYPE=auto
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
# gcc/g++ ARE runtime deps — required by Triton for CUDA kernel JIT compilation
# when the causal-LM reranker loads on GPU. Do NOT remove them.
# curl = healthcheck, ca-certificates = TLS for HuggingFace downloads.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      gcc \
      g++ \
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
      "fastembed>=0.4,<1" \
      "sentence-transformers>=3,<5" \
      "fastapi[standard]>=0.115,<1" \
      "uvicorn[standard]>=0.30,<1" \
      "pydantic>=2,<3" \
      "httpx>=0.27,<1" \
      "accelerate>=1.0,<2"

# ═══════════════════════════════════════════════════════════════════════════════════
# Stage 2 — model-cache : Pre-cache HuggingFace models at build time
# ═══════════════════════════════════════════════════════════════════════════════════
FROM python-base AS model-cache

ARG SPARSE_MODEL
ARG RERANK_MODEL
ARG RERANK_MODEL_FALLBACK
ARG RERANK_TYPE

ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

# ── Pre-cache sparse model (BM25) ────────────────────────────────────────────
RUN python -c \
    "from fastembed import SparseTextEmbedding; \
     m = SparseTextEmbedding(model_name='${SPARSE_MODEL}'); \
     list(m.embed(['warmup'])); \
     print('Sparse model cached: ${SPARSE_MODEL}')"

# ── Pre-cache primary reranker (graceful fallback on failure) ────────────────
RUN ( \
  if [ "${RERANK_TYPE}" = "causal-lm" ] || (echo "${RERANK_MODEL}" | grep -qi "qwen3-reranker"); then \
    echo "Pre-caching causal-LM reranker: ${RERANK_MODEL}"; \
    python -c \
      "from transformers import AutoModelForCausalLM, AutoTokenizer; \
       AutoTokenizer.from_pretrained('${RERANK_MODEL}', trust_remote_code=True); \
       AutoModelForCausalLM.from_pretrained('${RERANK_MODEL}', trust_remote_code=True); \
       print('Causal-LM reranker cached')"; \
  else \
    echo "Pre-caching cross-encoder reranker: ${RERANK_MODEL}"; \
    python -c \
      "from sentence_transformers import CrossEncoder; \
       m = CrossEncoder('${RERANK_MODEL}', device='cpu'); \
       print('Cross-encoder reranker cached')"; \
  fi \
  ) || echo "Primary reranker pre-cache skipped — fallback will load at runtime"

# ── Always pre-cache the fallback reranker ───────────────────────────────────
RUN python -c \
    "from sentence_transformers import CrossEncoder; \
     m = CrossEncoder('${RERANK_MODEL_FALLBACK}', device='cpu'); \
     print('Fallback reranker cached: ${RERANK_MODEL_FALLBACK}')"

# ═══════════════════════════════════════════════════════════════════════════════════
# Stage 3 — runtime : Minimal production image (most volatile, changes with code)
# ═══════════════════════════════════════════════════════════════════════════════════
FROM python-base AS runtime

ARG SPARSE_MODEL
ARG RERANK_MODEL
ARG RERANK_MODEL_FALLBACK
ARG RERANK_TYPE

ENV SPARSE_MODEL=${SPARSE_MODEL} \
    RERANK_MODEL=${RERANK_MODEL} \
    RERANK_MODEL_FALLBACK=${RERANK_MODEL_FALLBACK} \
    RERANK_TYPE=${RERANK_TYPE} \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

# ── Non-root user (explicit UID/GID for host volume compatibility) ───────────
RUN groupadd -g 1001 -r appgroup && \
    useradd -u 1001 -r -g appgroup -d /app -s /sbin/nologin appuser && \
    mkdir -p /app/.cache && \
    chown -R appuser:appgroup /app

# ── Copy pre-cached models and application code ──────────────────────────────
# --link avoids preserving intermediate layers (valid on --from copies).
# Use numeric UID:GID below — --link with named user/group hits "invalid user index".
COPY --from=model-cache --chown=1001:1001 --link /app/.cache /app/.cache
COPY --chown=1001:1001 memex/engine/retrieval/reranker.py /app/server.py

USER 1001

EXPOSE 5002

# ── Healthcheck (curl inside container) ──────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD ["curl", "-sf", "http://localhost:5002/health"]

STOPSIGNAL SIGTERM

# ── OCI labels ───────────────────────────────────────────────────────────────
LABEL org.opencontainers.image.title="Memex ML Services" \
      org.opencontainers.image.description="Sparse BM25 embeddings + cross-encoder/causal-LM reranker" \
      org.opencontainers.image.source="https://github.com/0xPratikPatil/Memex" \
      org.opencontainers.image.authors="0xPratikPatil" \
      org.opencontainers.image.documentation="https://github.com/0xPratikPatil/Memex/blob/main/DOCKER.md"

CMD ["/opt/venv/bin/python", "/app/server.py"]
