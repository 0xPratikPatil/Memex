# syntax=docker/dockerfile:1
# ══════════════════════════════════════════════════════════════════════════════
# Memex — ML Services (sparse embeddings + cross-encoder reranker)
#
# Optimized for: layer caching, reproducible builds, GPU inference.
# Models are pre-cached at build time for instant startup.
# ══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: Dependencies (cached) ───────────────────────────────────────────
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime AS deps

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN uv pip install --no-cache \
    "fastembed>=0.4,<1" \
    "sentence-transformers>=3,<5" \
    "fastapi[standard]>=0.115,<1" \
    "uvicorn[standard]>=0.30,<1" \
    "pydantic>=2,<3" \
    "httpx>=0.27,<1"

# ── Stage 2: Model pre-caching (optional, build-time speedup) ────────────────
FROM deps AS preload
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ARG SPARSE_MODEL=Qdrant/bm25
ARG RERANK_MODEL=Qwen3-Reranker-0.6B
ARG RERANK_MODEL_FALLBACK=BAAI/bge-reranker-base
ARG RERANK_TYPE=auto
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

RUN python -c "\
from fastembed import SparseTextEmbedding; \
m = SparseTextEmbedding(model_name='${SPARSE_MODEL}'); \
list(m.embed(['warmup'])); \
print('Sparse model cached')"

# Pre-cache reranker based on type (auto-detect or explicit)
# Skip on failure - fallback happens at runtime
RUN (if [ "$RERANK_TYPE" = "causal-lm" ] || (echo "$RERANK_MODEL" | grep -qi "qwen3-reranker"); then \
      echo "Pre-caching causal-LM reranker: ${RERANK_MODEL}"; \
      python -c "\
from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoTokenizer.from_pretrained('${RERANK_MODEL}', trust_remote_code=True); \
AutoModelForCausalLM.from_pretrained('${RERANK_MODEL}', trust_remote_code=True); \
print('Causal-LM reranker cached')" ; \
    else \
      echo "Pre-caching cross-encoder reranker: ${RERANK_MODEL}"; \
      python -c "\
from sentence_transformers import CrossEncoder; \
m = CrossEncoder('${RERANK_MODEL}', device='cpu'); \
print('Cross-encoder reranker cached')" ; \
    fi) || echo "Reranker pre-cache failed, will fallback at runtime"

# Also pre-cache the fallback reranker
RUN python -c "\
from sentence_transformers import CrossEncoder; \
m = CrossEncoder('${RERANK_MODEL_FALLBACK}', device='cpu'); \
print('Fallback reranker cached')"

# ── Stage 3: Runtime (minimal) ───────────────────────────────────────────────
FROM deps AS ml
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN groupadd -g 1001 -r appgroup && \
    useradd -u 1001 -r -g appgroup -d /app -s /sbin/nologin appuser && \
    mkdir -p /app/.cache && chown -R appuser:appgroup /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=preload --chown=appuser:appgroup /app/.cache /app/.cache

COPY --chown=appuser:appgroup rag/ml_server.py /app/server.py

ARG SPARSE_MODEL=Qdrant/bm25
ARG RERANK_MODEL=Qwen3-Reranker-0.6B
ARG RERANK_MODEL_FALLBACK=BAAI/bge-reranker-base
ARG RERANK_TYPE=auto
ENV SPARSE_MODEL=${SPARSE_MODEL}
ENV RERANK_MODEL=${RERANK_MODEL}
ENV RERANK_MODEL_FALLBACK=${RERANK_MODEL_FALLBACK}
ENV RERANK_TYPE=${RERANK_TYPE}
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

USER 1001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:5002/health || exit 1

EXPOSE 5002

CMD ["/opt/venv/bin/python", "/app/server.py"]
