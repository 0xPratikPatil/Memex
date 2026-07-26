# syntax=docker/dockerfile:1
# ══════════════════════════════════════════════════════════════════════════════
# Memex - Multi-stage Dockerfile
# Produces two images via build targets:
#   docker build --target mcp -t memex-mcp .
#   docker build --target fileserver -t memex-fileserver .
# ══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: Python builder (shared base) ────────────────────────────────────
FROM python:3.12-slim AS python-builder

WORKDIR /app

# Install build dependencies with cache mounts
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies with pip cache mount
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -e . 2>/dev/null || \
    pip install --no-cache-dir \
        httpx \
        tenacity \
        mcp \
        qdrant-client \
        fastembed \
        sentence-transformers \
        redis

# ══════════════════════════════════════════════════════════════════════════════
# TARGET: mcp - Full MCP Server with ML dependencies
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.12-slim AS mcp

LABEL org.opencontainers.image.title="Memex MCP Server"
LABEL org.opencontainers.image.description="Production MCP server for RAG with Docling + Qdrant + Ollama"
LABEL org.opencontainers.image.version="0.3.0"

WORKDIR /app

# Install runtime-only dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=python-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin

# Create non-root user with specific UID/GID and writable cache dir
RUN groupadd -g 10001 -r appgroup && \
    useradd -u 10001 -r -g appgroup -d /app -s /sbin/nologin appuser && \
    mkdir -p /app/.cache && chown -R appuser:appgroup /app/.cache

# Copy application code with proper ownership
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

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["--http"]

# ══════════════════════════════════════════════════════════════════════════════
# TARGET: fileserver - Lightweight file server (no ML dependencies)
# ══════════════════════════════════════════════════════════════════════════════
FROM python:3.12-slim AS fileserver

LABEL org.opencontainers.image.title="Memex File Server"
LABEL org.opencontainers.image.description="Lightweight file server for MCP Docker access"
LABEL org.opencontainers.image.version="0.3.0"

WORKDIR /app

# Install curl for healthcheck
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -g 10001 -r appgroup && \
    useradd -u 10001 -r -g appgroup -d /app -s /sbin/nologin appuser

# Copy only the file server script
COPY --chown=appuser:appgroup src/services/file_server.py .

# Switch to non-root user
USER 10001

EXPOSE 9900

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9900/health || exit 1

ENTRYPOINT ["python", "file_server.py"]
CMD ["--port", "9900", "--roots", "/mnt", "/home"]
