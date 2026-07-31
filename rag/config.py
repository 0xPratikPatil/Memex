"""Central configuration — YAML-first with env var fallback.

Priority: config.yaml > env var > .env file > hardcoded default.

Loads .env automatically so ``uv run memex`` picks up your settings.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

_log = logging.getLogger(__name__)

# ── Auto-load .env ──────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv

    _env_file = Path(__file__).resolve().parent.parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=False)
except ImportError:
    pass

# ── Load YAML config ────────────────────────────────────────────────────────
from rag.yaml_config import YamlConfig  # noqa: E402

_yaml: YamlConfig | None = None
try:
    _config_path = os.getenv("MEMEX_CONFIG", str(Path(__file__).resolve().parent.parent / "config.yaml"))
    _candidate = YamlConfig(_config_path)
    if _candidate.data:
        _yaml = _candidate
        _log.debug("Loaded YAML config from %s", _config_path)
except Exception:
    _yaml = None

# ── Deprecated env var tracking ─────────────────────────────────────────────
_deprecated_used: list[str] = []
_yaml_exists = _yaml is not None and bool(_yaml.data)


def _check_deprecated(env_key: str) -> None:
    if _yaml_exists and env_key in os.environ:
        _deprecated_used.append(env_key)


def _flush_deprecated() -> None:
    if _deprecated_used:
        _log.warning(
            "Detected %d deprecated env vars: %s — migrate to config.yaml",
            len(_deprecated_used),
            ", ".join(sorted(set(_deprecated_used))),
        )
        _deprecated_used.clear()


# ── Env var helpers (kept as private fallback) ──────────────────────────────
def _env(key: str, default: str) -> str:
    val = os.getenv(key, default)
    if val:
        val = re.sub(r"\s+#.*$", "", val).rstrip()
    return val or default


def _env_int(key: str, default: int) -> int:
    try:
        val = os.getenv(key, str(default))
        if val:
            val = re.sub(r"\s+#.*$", "", val).strip()
        return int(val)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        val = os.getenv(key, str(default))
        if val:
            val = re.sub(r"\s+#.*$", "", val).strip()
        return float(val)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key, str(default)).lower()
    val = re.sub(r"\s+#.*$", "", val).strip()
    return val in ("1", "true", "yes")


def _cfg_str(yaml_path: str, env_key: str, default: str) -> str:
    if _yaml is not None:
        val = _yaml.get_str(yaml_path)
        if val:
            return val
    _check_deprecated(env_key)
    return _env(env_key, default)


def _cfg_int(yaml_path: str, env_key: str, default: int) -> int:
    if _yaml is not None:
        val = _yaml.get(yaml_path)
        if val is not None:
            return _yaml.get_int(yaml_path, default)
    _check_deprecated(env_key)
    return _env_int(env_key, default)


def _cfg_float(yaml_path: str, env_key: str, default: float) -> float:
    if _yaml is not None:
        val = _yaml.get(yaml_path)
        if val is not None:
            return _yaml.get_float(yaml_path, default)
    _check_deprecated(env_key)
    return _env_float(env_key, default)


def _cfg_bool(yaml_path: str, env_key: str, default: bool) -> bool:
    if _yaml is not None:
        val = _yaml.get(yaml_path)
        if val is not None:
            return _yaml.get_bool(yaml_path, default)
    _check_deprecated(env_key)
    return _env_bool(env_key, default)


# ── Service ports ────────────────────────────────────────────────────────────
QDRANT_PORT: int = _cfg_int("vectorstore.port", "QDRANT_PORT", 6333)
OLLAMA_PORT: int = _cfg_int("embedding.port", "OLLAMA_PORT", 11434)
DOCLING_PORT: int = _cfg_int("converter.port", "DOCLING_PORT", 5001)
ML_SERVICES_PORT: int = _cfg_int("sparse.port", "ML_SERVICES_PORT", 5002)
REDIS_PORT: int = _cfg_int("caching.port", "REDIS_PORT", 6379)
MCP_PORT: int = _cfg_int("mcp.port", "MCP_PORT", 8080)

# ── Service URLs ─────────────────────────────────────────────────────────────
DOCLING_URL: str = _cfg_str(
    "converter.docling_url", "DOCLING_URL", f"http://localhost:{DOCLING_PORT}/v1/convert/source"
)
OLLAMA_EMBED_URL: str = _cfg_str("embedding.base_url", "OLLAMA_EMBED_URL", f"http://localhost:{OLLAMA_PORT}/api/embed")
QDRANT_URL: str = _cfg_str("vectorstore.url", "QDRANT_URL", f"http://localhost:{QDRANT_PORT}")
ML_SERVICES_URL: str = _cfg_str("sparse.url", "ML_SERVICES_URL", f"http://localhost:{ML_SERVICES_PORT}")
REDIS_URL: str = _cfg_str("caching.redis_url", "REDIS_URL", f"redis://localhost:{REDIS_PORT}/0")

# ── API Keys ─────────────────────────────────────────────────────────────────
DOCLING_API_KEY: str = _cfg_str("converter.docling_api_key", "DOCLING_API_KEY", "")

# ── Model / collection names ────────────────────────────────────────────────
COLLECTION_NAME: str = _cfg_str("vectorstore.collection", "COLLECTION_NAME", "memex")
EMBED_MODEL: str = _cfg_str("embedding.model", "EMBED_MODEL", "qwen3-embedding:0.6b")
EMBED_MODEL_FALLBACK: str = _cfg_str("embedding.fallback_model", "EMBED_MODEL_FALLBACK", "qwen3-embedding:0.6b")
RERANK_MODEL: str = _cfg_str("reranker.model", "RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
RERANK_MODEL_FALLBACK: str = _cfg_str("reranker.fallback_model", "RERANK_MODEL_FALLBACK", "BAAI/bge-reranker-base")
SPARSE_MODEL: str = _cfg_str("sparse.model", "SPARSE_MODEL", "Qdrant/bm25")
CHAT_MODEL: str = _cfg_str("llm.model", "CHAT_MODEL", "qwen2.5:1.5b")

SPARSE_PROVIDER: str = _cfg_str("sparse.provider", "SPARSE_PROVIDER", "http")
RERANK_PROVIDER: str = _cfg_str("reranker.provider", "RERANK_PROVIDER", "http")
RERANK_TYPE: str = _cfg_str("reranker.type", "RERANK_TYPE", "auto")
DENSE_DIM: int = _cfg_int("embedding.dimensions", "DENSE_DIM", 1024)

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_TOKENIZER: str = _cfg_str("chunking.tokenizer", "CHUNK_TOKENIZER", "Qwen/Qwen3-Embedding-0.6B")
CHUNK_SIZE: int = _cfg_int("chunking.size", "CHUNK_SIZE", 1024)
CHUNK_OVERLAP: int = _cfg_int("chunking.overlap", "CHUNK_OVERLAP", 128)
MIN_CHUNK_LEN: int = _cfg_int("chunking.min_length", "MIN_CHUNK_LEN", 30)
CHUNK_STRATEGY: str = _cfg_str("chunking.strategy", "CHUNK_STRATEGY", "hybrid")
CHUNK_MERGE_PEERS: bool = _cfg_bool("chunking.merge_peers", "CHUNK_MERGE_PEERS", True)

# ── HTTP client settings ────────────────────────────────────────────────────
HTTP_TIMEOUT: float = _cfg_float("http.timeout", "HTTP_TIMEOUT", 60.0)
DOCLING_TIMEOUT: float = _cfg_float("converter.docling_timeout", "DOCLING_TIMEOUT", 300.0)
HTTP_MAX_RETRIES: int = _cfg_int("http.max_retries", "HTTP_MAX_RETRIES", 3)
HTTP_RETRY_BACKOFF: float = _cfg_float("http.retry_backoff", "HTTP_RETRY_BACKOFF", 0.5)

# ── Ingestion pipeline settings ─────────────────────────────────────────────
INGEST_TIMEOUT_PARSE: float = _cfg_float("ingestion.timeout_parse", "INGEST_TIMEOUT_PARSE", 120.0)
INGEST_TIMEOUT_TOTAL: float = _cfg_float("ingestion.timeout_total", "INGEST_TIMEOUT_TOTAL", 300.0)
MAX_CONCURRENT_PARSES: int = _cfg_int("ingestion.max_concurrent_parses", "MAX_CONCURRENT_PARSES", 3)

# ── Qdrant client settings ──────────────────────────────────────────────────
QDRANT_TIMEOUT: float = _cfg_float("qdrant.timeout", "QDRANT_TIMEOUT", 10.0)
QDRANT_MAX_RETRIES: int = _cfg_int("qdrant.max_retries", "QDRANT_MAX_RETRIES", 3)

# ── Search settings ─────────────────────────────────────────────────────────
SEARCH_TOP_K: int = _cfg_int("search.top_k", "SEARCH_TOP_K", 30)
SEARCH_MODE: str = _cfg_str("search.mode", "SEARCH_MODE", "hybrid")
MMR_FETCH_K: int = _cfg_int("search.mmr_fetch_k", "MMR_FETCH_K", 20)
MMR_LAMBDA_MULT: float = _cfg_float("search.mmr_lambda_mult", "MMR_LAMBDA_MULT", 0.5)

# ── MCP server settings ─────────────────────────────────────────────────────
MCP_HOST: str = _cfg_str("mcp.host", "MCP_HOST", "127.0.0.1")

# ── Response limits ─────────────────────────────────────────────────────────
CHARACTER_LIMIT: int = _cfg_int("mcp.character_limit", "CHARACTER_LIMIT", 25000)

# ── Feature toggles ─────────────────────────────────────────────────────────
ENABLE_OCR: bool = _cfg_bool("converter.docling_ocr", "ENABLE_OCR", True)
ENABLE_RERANKING: bool = _cfg_bool("reranker.enabled", "ENABLE_RERANKING", True)
ENABLE_RERANKER: bool = _cfg_bool("reranker.enabled", "ENABLE_RERANKER", True)

# ── Docling enrichment ──────────────────────────────────────────────────────
DOCLING_ENRICH_CODE: bool = _cfg_bool("converter.docling_enrich_code", "DOCLING_ENRICH_CODE", False)
DOCLING_ENRICH_FORMULA: bool = _cfg_bool("converter.docling_enrich_formula", "DOCLING_ENRICH_FORMULA", False)
DOCLING_PICTURE_CLASSIFY: bool = _cfg_bool("converter.docling_picture_classify", "DOCLING_PICTURE_CLASSIFY", True)
DOCLING_CHART_EXTRACT: bool = _cfg_bool("converter.docling_chart_extract", "DOCLING_CHART_EXTRACT", False)
DOCLING_IMAGE_EXPORT: str = _cfg_str("converter.docling_image_export", "DOCLING_IMAGE_EXPORT", "embedded")
DOCLING_PDF_BACKEND: str = _cfg_str("converter.docling_pdf_backend", "DOCLING_PDF_BACKEND", "")

# ── Query Expansion ─────────────────────────────────────────────────────────
ENABLE_QUERY_EXPANSION: bool = _cfg_bool("query_expansion.enabled", "ENABLE_QUERY_EXPANSION", False)
ENABLE_HYDE: bool = _cfg_bool("query_expansion.hyde", "ENABLE_HYDE", False)
ENABLE_MULTI_QUERY: bool = _cfg_bool("query_expansion.multi_query", "ENABLE_MULTI_QUERY", False)
ENABLE_QUERY_REWRITE: bool = _cfg_bool("query_expansion.query_rewrite", "ENABLE_QUERY_REWRITE", False)

HYDE_MODEL: str = _cfg_str("query_expansion.hyde_model", "HYDE_MODEL", "")
MULTI_QUERY_COUNT: int = _cfg_int("query_expansion.multi_query_count", "MULTI_QUERY_COUNT", 3)
MULTI_QUERY_MODEL: str = _cfg_str("query_expansion.multi_query_model", "MULTI_QUERY_MODEL", "")
QUERY_REWRITE_MODEL: str = _cfg_str("query_expansion.query_rewrite_model", "QUERY_REWRITE_MODEL", "")

# ── Contextual Retrieval ───────────────────────────────────────────────────
ENABLE_CONTEXTUAL_RETRIEVAL: bool = _cfg_bool("contextual_retrieval.enabled", "ENABLE_CONTEXTUAL_RETRIEVAL", False)
CONTEXT_STRATEGY: str = _cfg_str("contextual_retrieval.strategy", "CONTEXT_STRATEGY", "summary")
CONTEXT_MODEL: str = _cfg_str("contextual_retrieval.model", "CONTEXT_MODEL", "")
CONTEXT_PREFIX_MAX_TOKENS: int = _cfg_int("contextual_retrieval.max_tokens", "CONTEXT_PREFIX_MAX_TOKENS", 50)
CONTEXT_BATCH_SIZE: int = _cfg_int("contextual_retrieval.batch_size", "CONTEXT_BATCH_SIZE", 10)

# ── Embedding batch size ────────────────────────────────────────────────────
EMBED_BATCH_SIZE: int = max(1, _cfg_int("embedding.batch_size", "EMBED_BATCH_SIZE", 64))

# ── Cache Settings ──────────────────────────────────────────────────────────
ENABLE_CACHE: bool = _cfg_bool("caching.enabled", "ENABLE_CACHE", True)

CACHE_TTL_EMBEDDING: int = _cfg_int("caching.ttl_embedding", "CACHE_TTL_EMBEDDING", 86400)
CACHE_TTL_SEARCH: int = _cfg_int("caching.ttl_search", "CACHE_TTL_SEARCH", 3600)
CACHE_TTL_PARSE: int = _cfg_int("caching.ttl_parse", "CACHE_TTL_PARSE", 604800)
CACHE_TTL_EXPANSION: int = _cfg_int("caching.ttl_expansion", "CACHE_TTL_EXPANSION", 21600)

CACHE_MAX_MEMORY_MB: int = _cfg_int("caching.max_memory_mb", "CACHE_MAX_MEMORY_MB", 256)
CACHE_EVICTION_POLICY: str = _cfg_str("caching.eviction_policy", "CACHE_EVICTION_POLICY", "allkeys-lru")

# ── Metadata Enhancement ────────────────────────────────────────────────────
ENABLE_METADATA_EXTRACTION: bool = _cfg_bool("metadata.extraction_enabled", "ENABLE_METADATA_EXTRACTION", True)
ENABLE_ENTITY_EXTRACTION: bool = _cfg_bool("metadata.entity_extraction", "ENABLE_ENTITY_EXTRACTION", True)
ENABLE_DOC_CLASSIFICATION: bool = _cfg_bool("metadata.doc_classification", "ENABLE_DOC_CLASSIFICATION", True)
ENABLE_TOPIC_TAGGING: bool = _cfg_bool("metadata.topic_tagging", "ENABLE_TOPIC_TAGGING", True)
ENABLE_LANGUAGE_DETECTION: bool = _cfg_bool("metadata.language_detection", "ENABLE_LANGUAGE_DETECTION", True)

METADATA_MODEL: str = _cfg_str("metadata.model", "METADATA_MODEL", "")
MAX_ENTITIES_PER_CHUNK: int = _cfg_int("metadata.max_entities_per_chunk", "MAX_ENTITIES_PER_CHUNK", 10)
MAX_TOPICS_PER_CHUNK: int = _cfg_int("metadata.max_topics_per_chunk", "MAX_TOPICS_PER_CHUNK", 5)

# ── Evaluation ──────────────────────────────────────────────────────────────
EVAL_ENABLED: bool = _cfg_bool("evaluation.enabled", "EVAL_ENABLED", False)
EVAL_DATASET_PATH: str = _cfg_str("evaluation.golden_set_path", "EVAL_DATASET_PATH", "tests/fixtures/evaluation.xml")
EVAL_OUTPUT_DIR: str = _cfg_str("evaluation.output_dir", "EVAL_OUTPUT_DIR", "eval_reports")
EVAL_TOP_K: int = _cfg_int("evaluation.top_k", "EVAL_TOP_K", 10)
EVAL_RUN_RAGAS: bool = _cfg_bool("evaluation.run_ragas", "EVAL_RUN_RAGAS", False)
EVAL_LOG_TIMING: bool = _cfg_bool("evaluation.log_timing", "EVAL_LOG_TIMING", False)

# ── Answer Generation ───────────────────────────────────────────────────────
ENABLE_ANSWER: bool = _cfg_bool("answer.enabled", "ENABLE_ANSWER", False)
ANSWER_MAX_CONTEXT_CHARS: int = _cfg_int("answer.max_context_chars", "ANSWER_MAX_CONTEXT_CHARS", 12000)

# ── Startup checks ──────────────────────────────────────────────────────────


def _run_startup_checks() -> None:
    """Log warnings for misconfigured settings that hurt performance."""
    if EMBED_MODEL == EMBED_MODEL_FALLBACK:
        _log.warning(
            "EMBED_MODEL and EMBED_MODEL_FALLBACK are identical (%s) — "
            "fallback will have no effect. Set embedding.fallback_model in config.yaml.",
            EMBED_MODEL,
        )
    if RERANK_MODEL == RERANK_MODEL_FALLBACK:
        _log.warning(
            "RERANK_MODEL and RERANK_MODEL_FALLBACK are identical (%s) — "
            "fallback will have no effect. Set reranker.fallback_model in config.yaml.",
            RERANK_MODEL,
        )
    if ENABLE_CONTEXTUAL_RETRIEVAL:
        _log.warning(
            "ENABLE_CONTEXTUAL_RETRIEVAL is on — doubles embedding cost. "
            "Consider disabling for small document collections."
        )
    if ENABLE_QUERY_EXPANSION and (ENABLE_HYDE or ENABLE_MULTI_QUERY or ENABLE_QUERY_REWRITE):
        expansion_count = sum([ENABLE_HYDE, ENABLE_MULTI_QUERY, ENABLE_QUERY_REWRITE])
        _log.warning(
            "Query expansion is on with %d sub-techniques — "
            "this fires %d+ LLM calls per search. Consider disabling for small collections.",
            expansion_count,
            expansion_count,
        )
    _flush_deprecated()


_run_startup_checks()
