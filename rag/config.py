"""Central configuration — all values overridable via environment variables.

Priority: env var > .env file > hardcoded default.

Loads .env automatically so ``uv run memex`` picks up your settings.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Auto-load .env ──────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).resolve().parent.parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=False)  # don't override existing env vars
except ImportError:
    pass


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


# ── Service ports (single source of truth) ──────────────────────────────────
QDRANT_PORT: int = _env_int("QDRANT_PORT", 6333)
OLLAMA_PORT: int = _env_int("OLLAMA_PORT", 11434)
DOCLING_PORT: int = _env_int("DOCLING_PORT", 5001)
ML_SERVICES_PORT: int = _env_int("ML_SERVICES_PORT", 5002)
REDIS_PORT: int = _env_int("REDIS_PORT", 6379)
MCP_PORT: int = _env_int("MCP_PORT", 8080)

# ── Service URLs (constructed from ports, overridable via explicit *_URL env) ─
DOCLING_URL: str = _env("DOCLING_URL", f"http://localhost:{DOCLING_PORT}/v1/convert/source")
OLLAMA_EMBED_URL: str = _env("OLLAMA_EMBED_URL", f"http://localhost:{OLLAMA_PORT}/api/embeddings")
QDRANT_URL: str = _env("QDRANT_URL", f"http://localhost:{QDRANT_PORT}")
ML_SERVICES_URL: str = _env("ML_SERVICES_URL", f"http://localhost:{ML_SERVICES_PORT}")
REDIS_URL: str = _env("REDIS_URL", f"redis://localhost:{REDIS_PORT}/0")


# ── API Keys ──────────────────────────────────────────────────────────────────
DOCLING_API_KEY: str = _env("DOCLING_API_KEY", "")

# ── Model / collection names ──────────────────────────────────────────────────
COLLECTION_NAME: str = _env("COLLECTION_NAME", "memex")
EMBED_MODEL: str = _env("EMBED_MODEL", "bge-m3")
RERANK_MODEL: str = _env("RERANK_MODEL", "BAAI/bge-reranker-base")
SPARSE_MODEL: str = _env("SPARSE_MODEL", "Qdrant/bm25")
CHAT_MODEL: str = _env("CHAT_MODEL", "qwen2.5:0.5b")  # for context/query/metadata generation

# Provider: "http" (Docker ML service) or "local" (load in-process)
SPARSE_PROVIDER: str = _env("SPARSE_PROVIDER", "http")
RERANK_PROVIDER: str = _env("RERANK_PROVIDER", "http")
DENSE_DIM: int = _env_int("DENSE_DIM", 1024)

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = _env_int("CHUNK_SIZE", 512)  # tokens (target)
CHUNK_OVERLAP: int = _env_int("CHUNK_OVERLAP", 50)  # tokens (overlap)
MIN_CHUNK_LEN: int = _env_int("MIN_CHUNK_LEN", 30)  # min chars to keep
CHUNK_STRATEGY: str = _env("CHUNK_STRATEGY", "recursive")  # recursive | paragraph | fixed

# ── HTTP client settings ──────────────────────────────────────────────────────
HTTP_TIMEOUT: float = _env_float("HTTP_TIMEOUT", 60.0)  # seconds per request
DOCLING_TIMEOUT: float = _env_float("DOCLING_TIMEOUT", 300.0)  # doc conversion can be slow
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

# ── Response limits ────────────────────────────────────────────────────────────
CHARACTER_LIMIT: int = _env_int("CHARACTER_LIMIT", 25000)

# ── Feature toggles ──────────────────────────────────────────────────────────
ENABLE_OCR: bool = _env_bool("ENABLE_OCR", True)
ENABLE_RERANKING: bool = _env_bool("ENABLE_RERANKING", True)

# ── Query Expansion ──────────────────────────────────────────────────────────
ENABLE_QUERY_EXPANSION: bool = _env_bool("ENABLE_QUERY_EXPANSION", False)
ENABLE_HYDE: bool = _env_bool("ENABLE_HYDE", False)
ENABLE_MULTI_QUERY: bool = _env_bool("ENABLE_MULTI_QUERY", False)
ENABLE_QUERY_REWRITE: bool = _env_bool("ENABLE_QUERY_REWRITE", False)

HYDE_MODEL: str = _env("HYDE_MODEL", "")  # empty = use CHAT_MODEL
MULTI_QUERY_COUNT: int = _env_int("MULTI_QUERY_COUNT", 3)
MULTI_QUERY_MODEL: str = _env("MULTI_QUERY_MODEL", "")  # empty = use CHAT_MODEL
QUERY_REWRITE_MODEL: str = _env("QUERY_REWRITE_MODEL", "")  # empty = use CHAT_MODEL

# ── Contextual Retrieval ────────────────────────────────────────────────────
ENABLE_CONTEXTUAL_RETRIEVAL: bool = _env_bool("ENABLE_CONTEXTUAL_RETRIEVAL", False)
CONTEXT_STRATEGY: str = _env("CONTEXT_STRATEGY", "header")  # header | summary | surrounding
CONTEXT_MODEL: str = _env("CONTEXT_MODEL", "")  # empty = use CHAT_MODEL
CONTEXT_PREFIX_MAX_TOKENS: int = _env_int("CONTEXT_PREFIX_MAX_TOKENS", 50)
CONTEXT_BATCH_SIZE: int = _env_int("CONTEXT_BATCH_SIZE", 10)

# ── Embedding batch size ─────────────────────────────────────────────────────
EMBED_BATCH_SIZE: int = _env_int("EMBED_BATCH_SIZE", 32)

# ── Cache Settings ──────────────────────────────────────────────────────────
ENABLE_CACHE: bool = _env_bool("ENABLE_CACHE", False)

CACHE_TTL_EMBEDDING: int = _env_int("CACHE_TTL_EMBEDDING", 86400)  # 24h
CACHE_TTL_SEARCH: int = _env_int("CACHE_TTL_SEARCH", 3600)  # 1h
CACHE_TTL_PARSE: int = _env_int("CACHE_TTL_PARSE", 604800)  # 7d
CACHE_TTL_EXPANSION: int = _env_int("CACHE_TTL_EXPANSION", 21600)  # 6h

CACHE_MAX_MEMORY_MB: int = _env_int("CACHE_MAX_MEMORY_MB", 256)
CACHE_EVICTION_POLICY: str = _env("CACHE_EVICTION_POLICY", "allkeys-lru")

# ── Metadata Enhancement ──────────────────────────────────────────────────────
ENABLE_METADATA_EXTRACTION: bool = _env_bool("ENABLE_METADATA_EXTRACTION", False)
ENABLE_ENTITY_EXTRACTION: bool = _env_bool("ENABLE_ENTITY_EXTRACTION", False)
ENABLE_DOC_CLASSIFICATION: bool = _env_bool("ENABLE_DOC_CLASSIFICATION", False)
ENABLE_TOPIC_TAGGING: bool = _env_bool("ENABLE_TOPIC_TAGGING", False)
ENABLE_LANGUAGE_DETECTION: bool = _env_bool("ENABLE_LANGUAGE_DETECTION", True)

METADATA_MODEL: str = _env("METADATA_MODEL", "")  # empty = use CHAT_MODEL
MAX_ENTITIES_PER_CHUNK: int = _env_int("MAX_ENTITIES_PER_CHUNK", 10)
MAX_TOPICS_PER_CHUNK: int = _env_int("MAX_TOPICS_PER_CHUNK", 5)

# ── Evaluation ──────────────────────────────────────────────────────────────
EVAL_ENABLED: bool = _env_bool("EVAL_ENABLED", False)
EVAL_DATASET_PATH: str = _env("EVAL_DATASET_PATH", "tests/fixtures/evaluation.xml")
EVAL_OUTPUT_DIR: str = _env("EVAL_OUTPUT_DIR", "eval_reports")
EVAL_TOP_K: int = _env_int("EVAL_TOP_K", 10)
EVAL_RUN_RAGAS: bool = _env_bool("EVAL_RUN_RAGAS", False)
EVAL_LOG_TIMING: bool = _env_bool("EVAL_LOG_TIMING", False)
