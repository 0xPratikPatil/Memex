# syntax=docker/dockerfile:1
# ══════════════════════════════════════════════════════════════════════════════
# Memex - ML Services Dockerfile
# ══════════════════════════════════════════════════════════════════════════════

# TARGET: ml — Sparse embeddings + Cross-encoder reranker (GPU)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS ml

WORKDIR /app

# Install system deps + Python packages
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir \
        fastembed \
        sentence-transformers \
        fastapi \
        uvicorn \
        pydantic

COPY rag/ml_server.py /app/server.py

# Models configurable via env vars
ARG SPARSE_MODEL=Qdrant/bm25
ARG RERANK_MODEL=BAAI/bge-reranker-base
ENV SPARSE_MODEL=${SPARSE_MODEL}
ENV RERANK_MODEL=${RERANK_MODEL}

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:5002/health || exit 1

EXPOSE 5002

CMD ["python", "/app/server.py"]
