"""Central configuration — all values are overridable via environment variables.

Docker-friendly defaults use service names (e.g. "http://qdrant:6333") so the
MCP server works inside Docker Compose without any changes.  When run locally
the defaults fall back to localhost.
"""

from __future__ import annotations

import os


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))  # type: ignore[return-value]
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))  # type: ignore[return-value]
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ("1", "true", "yes")


# ── Service URLs ──────────────────────────────────────────────────────────────
# Docker Compose uses service names; local dev uses localhost.
DOCLING_URL: str = _env(
    "DOCLING_URL",
    _env("DOCLING_SERVER_URL", "http://localhost:5001/v1/convert/source"),
)
OLLAMA_EMBED_URL: str = _env(
    "OLLAMA_EMBED_URL",
    _env("OLLAMA_SERVER_URL", "http://localhost:11434/api/embeddings"),
)
QDRANT_URL: str = _env(
    "QDRANT_URL",
    _env("QDRANT_SERVER_URL", "http://localhost:6333"),
)
FILE_SERVER_URL: str = _env(
    "FILE_SERVER_URL",
    "http://localhost:9900",
)

# ── API Keys ──────────────────────────────────────────────────────────────────
DOCLING_API_KEY: str = _env("DOCLING_API_KEY", "")

# ── Model / collection names ──────────────────────────────────────────────────
COLLECTION_NAME: str = _env("COLLECTION_NAME", "personal_rag")
EMBED_MODEL: str = _env("EMBED_MODEL", "bge-m3")
RERANK_MODEL: str = _env("RERANK_MODEL", "BAAI/bge-reranker-base")
SPARSE_MODEL: str = _env("SPARSE_MODEL", "Qdrant/bm25")
DENSE_DIM: int = _env_int("DENSE_DIM", 1024)

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = _env_int("CHUNK_SIZE", 512)        # tokens (target)
CHUNK_OVERLAP: int = _env_int("CHUNK_OVERLAP", 50)   # tokens (overlap)
MIN_CHUNK_LEN: int = _env_int("MIN_CHUNK_LEN", 30)   # min chars to keep
CHUNK_STRATEGY: str = _env("CHUNK_STRATEGY", "recursive")  # recursive | paragraph | fixed

# ── HTTP client settings ──────────────────────────────────────────────────────
HTTP_TIMEOUT: float = _env_float("HTTP_TIMEOUT", 60.0)           # seconds per request
DOCLING_TIMEOUT: float = _env_float("DOCLING_TIMEOUT", 300.0)    # doc conversion can be slow
HTTP_MAX_RETRIES: int = _env_int("HTTP_MAX_RETRIES", 3)
HTTP_RETRY_BACKOFF: float = _env_float("HTTP_RETRY_BACKOFF", 0.5)  # exponential base

# ── Qdrant client settings ────────────────────────────────────────────────────
QDRANT_TIMEOUT: float = _env_float("QDRANT_TIMEOUT", 10.0)
QDRANT_MAX_RETRIES: int = _env_int("QDRANT_MAX_RETRIES", 3)

# ── Search settings ──────────────────────────────────────────────────────────
SEARCH_FUSION: str = _env("SEARCH_FUSION", "rrf")  # rrf | weighted
RERANK_ENABLED: bool = _env_bool("RERANK_ENABLED", True)
SEARCH_TOP_K: int = _env_int("SEARCH_TOP_K", 20)  # candidates before rerank

# ── MCP server settings ───────────────────────────────────────────────────────
MCP_HOST: str = _env("MCP_HOST", "0.0.0.0")
MCP_PORT: int = _env_int("MCP_PORT", 8080)

# ── Response limits ────────────────────────────────────────────────────────────
CHARACTER_LIMIT: int = _env_int("CHARACTER_LIMIT", 25000)

# ── Feature toggles ──────────────────────────────────────────────────────────
ENABLE_OCR: bool = _env_bool("ENABLE_OCR", True)
ENABLE_RERANKING: bool = _env_bool("ENABLE_RERANKING", True)

# ── Embedding batch size ─────────────────────────────────────────────────────
EMBED_BATCH_SIZE: int = _env_int("EMBED_BATCH_SIZE", 32)
