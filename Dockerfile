# syntax=docker/dockerfile:1
# ── Stage 1: build deps ──────────────────────────────────────────────────────
FROM python:3.12.8-slim AS builder

WORKDIR /app

# Install build dependencies with cache mounts
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./

# Install Python dependencies with pip cache mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -e . 2>/dev/null || \
    pip install --no-cache-dir httpx tenacity mcp qdrant-client fastembed sentence-transformers

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12.8-slim AS runtime

LABEL org.opencontainers.image.title="Personal RAG MCP Server"
LABEL org.opencontainers.image.description="Production MCP server for RAG with Docling + Qdrant + Ollama"

WORKDIR /app

# Install runtime-only dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user with specific UID/GID and writable cache dir
RUN groupadd -g 10001 -r appgroup && \
    useradd -u 10001 -r -g appgroup -d /app -s /sbin/nologin appuser && \
    mkdir -p /app/.cache && chown -R appuser:appgroup /app/.cache

# Copy application code with proper ownership
COPY --chown=appuser:appgroup config.py run.py ./
COPY --chown=appuser:appgroup src/ src/

# Switch to non-root user
USER 10001

# Environment — Docker-friendly defaults point to compose service names
ENV DOCLING_URL="http://docling:5001/v1/convert/source" \
    OLLAMA_EMBED_URL="http://ollama:11434/api/embeddings" \
    QDRANT_URL="http://qdrant:6333" \
    MCP_HOST="0.0.0.0" \
    MCP_PORT="8080" \
    HF_HOME="/app/.cache/huggingface" \
    TORCH_HOME="/app/.cache/torch" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["python", "run.py"]
CMD ["--http"]
