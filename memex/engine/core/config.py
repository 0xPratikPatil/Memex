"""Central configuration — config.yaml is the single source of truth.

Env vars are used ONLY for ``${VAR}`` substitution inside config.yaml values
(e.g. ``api_key: ${OPENAI_API_KEY}``). No env var fallback for config keys.

``MEMEX_CONFIG`` env var can override the config file path (default: config.yaml).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

_log = logging.getLogger(__name__)

# ── Load YAML config (single source of truth) ─────────────────────────────────
from memex.engine.core.yaml_config import YamlConfig  # noqa: E402

_CONFIG_PATH = os.environ.get("MEMEX_CONFIG", str(Path(__file__).resolve().parent.parent.parent.parent / "config.yaml"))
_yaml: YamlConfig | None = None

try:
    _candidate = YamlConfig(_CONFIG_PATH)
    if _candidate.data:
        _yaml = _candidate
        _log.info("Loaded config from %s", _CONFIG_PATH)
    else:
        _log.warning("Config file %s is empty — using defaults", _CONFIG_PATH)
except FileNotFoundError:
    _log.info("No config.yaml found at %s — using defaults", _CONFIG_PATH)
except Exception as exc:
    _log.warning("Failed to load config.yaml: %s — using defaults", exc)


def _cfg(path: str, default):
    """Read a value from config.yaml. Returns default if yaml unavailable or key missing."""
    if _yaml is not None:
        val = _yaml.get(path)
        if val is not None:
            return val
    return default


def _cfg_str(path: str, default: str) -> str:
    if _yaml is not None:
        return _yaml.get_str(path, default)
    return default


def _cfg_int(path: str, default: int) -> int:
    if _yaml is not None:
        return _yaml.get_int(path, default)
    return default


def _cfg_float(path: str, default: float) -> float:
    if _yaml is not None:
        return _yaml.get_float(path, default)
    return default


def _cfg_bool(path: str, default: bool) -> bool:
    if _yaml is not None:
        return _yaml.get_bool(path, default)
    return default


# ── Service ports ─────────────────────────────────────────────────────────────
QDRANT_PORT: int = _cfg_int("vectorstore.port", 6333)
OLLAMA_PORT: int = _cfg_int("embedding.port", 11434)
DOCLING_PORT: int = _cfg_int("converter.port", 5001)
ML_SERVICES_PORT: int = _cfg_int("sparse.port", 5002)
REDIS_PORT: int = _cfg_int("caching.port", 6379)
MCP_PORT: int = _cfg_int("mcp.port", 8080)

# ── Service URLs ──────────────────────────────────────────────────────────────
DOCLING_URL: str = _cfg_str("converter.docling_url", f"http://localhost:{DOCLING_PORT}/v1/convert/source")
OLLAMA_EMBED_URL: str = _cfg_str("embedding.base_url", f"http://localhost:{OLLAMA_PORT}/api/embed")
ML_SERVICES_URL: str = _cfg_str("sparse.url", f"http://localhost:{ML_SERVICES_PORT}")
QDRANT_URL: str = _cfg_str("vectorstore.url", f"http://localhost:{QDRANT_PORT}")
REDIS_URL: str = _cfg_str("caching.redis_url", f"redis://localhost:{REDIS_PORT}/0")

# ── Converter engine (marker | docling legacy) ─────────────────────────────────
CONVERTER_ENGINE: str = _cfg_str("converter.engine", "marker")
MARKER_URL: str = _cfg_str("converter.marker_url", f"http://localhost:{DOCLING_PORT}")
MARKER_MODE: str = _cfg_str("converter.marker_mode", "balanced")  # balanced | fast
MARKER_FORCE_OCR: bool = _cfg_bool("converter.marker_force_ocr", False)
MARKER_TIMEOUT: float = _cfg_float("converter.marker_timeout", 300.0)
# Cap concurrent in-flight conversions to Marker. Must stay low enough that
# the GPU service is not overwhelmed (2 is safe for a single GPU server).
CONVERTER_MAX_CONCURRENT: int = _cfg_int("converter.max_concurrent", 2)

# ── API Keys ──────────────────────────────────────────────────────────────────
DOCLING_API_KEY: str = _cfg_str("converter.docling_api_key", "")

# ── Provider selection ─────────────────────────────────────────────────────────
LLM_PROVIDER: str = _cfg_str("llm.provider", "ollama")
LLM_BASE_URL: str = _cfg_str("llm.base_url", "http://localhost:11434")
LLM_API_KEY: str = _cfg_str("llm.api_key", "")
EMBED_PROVIDER: str = _cfg_str("embedding.provider", "ollama")
EMBED_API_KEY: str = _cfg_str("embedding.api_key", "")

# ── Model / collection names ──────────────────────────────────────────────────
COLLECTION_NAME: str = _cfg_str("vectorstore.collection", "memex")
EMBED_MODEL: str = _cfg_str("embedding.model", "qwen3-embedding:0.6b")
EMBED_MODEL_FALLBACK: str = _cfg_str("embedding.fallback_model", "qwen3-embedding:0.6b")
RERANK_MODEL: str = _cfg_str("reranker.model", "Qwen/Qwen3-Reranker-0.6B")
RERANK_MODEL_FALLBACK: str = _cfg_str("reranker.fallback_model", "BAAI/bge-reranker-base")
SPARSE_MODEL: str = _cfg_str("sparse.model", "Qdrant/bm25")
CHAT_MODEL: str = _cfg_str("llm.model", "qwen2.5:1.5b")

SPARSE_PROVIDER: str = _cfg_str("sparse.provider", "docker")
RERANK_PROVIDER: str = _cfg_str("reranker.provider", "docker")
RERANK_TYPE: str = _cfg_str("reranker.type", "auto")
DENSE_DIM: int = _cfg_int("embedding.dimensions", 1024)

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_TOKENIZER: str = _cfg_str("chunking.tokenizer", "Qwen/Qwen3-Embedding-0.6B")
CHUNK_SIZE: int = _cfg_int("chunking.size", 1024)
CHUNK_OVERLAP: int = _cfg_int("chunking.overlap", 128)
MIN_CHUNK_LEN: int = _cfg_int("chunking.min_length", 30)
CHUNK_STRATEGY: str = _cfg_str("chunking.strategy", "hybrid")
CHUNK_MERGE_PEERS: bool = _cfg_bool("chunking.merge_peers", True)

# ── HTTP client settings ──────────────────────────────────────────────────────
HTTP_TIMEOUT: float = _cfg_float("http.timeout", 60.0)
DOCLING_TIMEOUT: float = _cfg_float("converter.docling_timeout", 300.0)
HTTP_MAX_RETRIES: int = _cfg_int("http.max_retries", 3)
HTTP_RETRY_BACKOFF: float = _cfg_float("http.retry_backoff", 0.5)
LLM_TIMEOUT: float = _cfg_float("llm.timeout", HTTP_TIMEOUT)

# Connection-level failures (server restart, dropped keep-alive) need a longer
# retry window than HTTP error codes — a container restart takes 30-60s.
HTTP_TRANSPORT_MAX_RETRIES: int = _cfg_int("http.transport_max_retries", 5)
HTTP_TRANSPORT_RETRY_BACKOFF: float = _cfg_float("http.transport_retry_backoff", 2.0)

# ── Docling Serve settings ────────────────────────────────────────────────────
DOCLING_SERVE_MAX_SYNC_WAIT: int = _cfg_int("converter.docling_serve_max_sync_wait", 120)
DOCLING_SKIP_ON_TIMEOUT: bool = _cfg_bool("converter.docling_skip_on_timeout", False)

# ── Docling async settings ────────────────────────────────────────────────────
DOCLING_POLL_INTERVAL: float = _cfg_float("converter.docling_poll_interval", 5.0)
DOCLING_MAX_RETRIES: int = _cfg_int("converter.docling_max_retries", 4)
DOCLING_RETRY_BACKOFF: list[float] = _cfg("converter.docling_retry_backoff", [60.0, 300.0, 1800.0, 7200.0])

# ── Ingestion pipeline settings ───────────────────────────────────────────────
INGEST_TIMEOUT_PARSE: float = _cfg_float("ingestion.timeout_parse", 120.0)
INGEST_TIMEOUT_TOTAL: float = _cfg_float("ingestion.timeout_total", 300.0)
MAX_CONCURRENT_PARSES: int = _cfg_int("ingestion.max_concurrent_parses", 3)
MAX_CONCURRENT_SYNC: int = _cfg_int("ingestion.max_concurrent_sync", 8)

# ── Qdrant client settings ────────────────────────────────────────────────────
QDRANT_TIMEOUT: float = _cfg_float("qdrant.timeout", 30.0)
QDRANT_MAX_RETRIES: int = _cfg_int("qdrant.max_retries", 3)

# ── Search settings ──────────────────────────────────────────────────────────
SEARCH_TOP_K: int = _cfg_int("search.top_k", 30)
SEARCH_MODE: str = _cfg_str("search.mode", "hybrid")
MMR_FETCH_K: int = _cfg_int("search.mmr_fetch_k", 20)
MMR_LAMBDA_MULT: float = _cfg_float("search.mmr_lambda_mult", 0.5)

# ── MCP server settings ───────────────────────────────────────────────────────
MCP_HOST: str = _cfg_str("mcp.host", "127.0.0.1")

# ── Response limits ───────────────────────────────────────────────────────────
CHARACTER_LIMIT: int = _cfg_int("mcp.character_limit", 25000)

# ── Feature toggles ──────────────────────────────────────────────────────────
ENABLE_OCR: bool = _cfg_bool("converter.docling_ocr", True)
ENABLE_RERANKING: bool = _cfg_bool("reranker.enabled", True)

# ── Docling enrichment ───────────────────────────────────────────────────────
DOCLING_ENRICH_CODE: bool = _cfg_bool("converter.docling_enrich_code", False)
DOCLING_ENRICH_FORMULA: bool = _cfg_bool("converter.docling_enrich_formula", False)
DOCLING_PICTURE_CLASSIFY: bool = _cfg_bool("converter.docling_picture_classify", True)
DOCLING_CHART_EXTRACT: bool = _cfg_bool("converter.docling_chart_extract", False)
DOCLING_IMAGE_EXPORT: str = _cfg_str("converter.docling_image_export", "embedded")
DOCLING_PDF_BACKEND: str = _cfg_str("converter.docling_pdf_backend", "")

# ── Docling conversion performance (legacy — only used when engine=docling) ──
DOCLING_TABLE_MODE: str = _cfg_str("converter.docling_table_mode", "accurate")  # accurate | fast
DOCLING_TABLE_STRUCTURE: bool = _cfg_bool("converter.docling_table_structure", True)
DOCLING_NUM_WORKERS: int = _cfg_int("converter.docling_num_workers", 0)  # 0 = server default
DOCLING_MAX_CONCURRENT: int = _cfg_int("converter.docling_max_concurrent", 2)

# ── Query Expansion ──────────────────────────────────────────────────────────
ENABLE_QUERY_EXPANSION: bool = _cfg_bool("query_expansion.enabled", True)
ENABLE_HYDE: bool = _cfg_bool("query_expansion.hyde", True)
ENABLE_MULTI_QUERY: bool = _cfg_bool("query_expansion.multi_query", True)
ENABLE_QUERY_REWRITE: bool = _cfg_bool("query_expansion.query_rewrite", True)

HYDE_MODEL: str = _cfg_str("query_expansion.hyde_model", "")
MULTI_QUERY_COUNT: int = _cfg_int("query_expansion.multi_query_count", 3)
MULTI_QUERY_MODEL: str = _cfg_str("query_expansion.multi_query_model", "")
QUERY_REWRITE_MODEL: str = _cfg_str("query_expansion.query_rewrite_model", "")

# ── Contextual Retrieval ─────────────────────────────────────────────────────
ENABLE_CONTEXTUAL_RETRIEVAL: bool = _cfg_bool("contextual_retrieval.enabled", True)
CONTEXT_STRATEGY: str = _cfg_str("contextual_retrieval.strategy", "summary")
CONTEXT_MODEL: str = _cfg_str("contextual_retrieval.model", "")
CONTEXT_PREFIX_MAX_TOKENS: int = _cfg_int("contextual_retrieval.max_tokens", 50)
CONTEXT_BATCH_SIZE: int = _cfg_int("contextual_retrieval.batch_size", 10)

# ── Embedding batch size ──────────────────────────────────────────────────────
EMBED_BATCH_SIZE: int = max(1, _cfg_int("embedding.batch_size", 64))

# ── Cache Settings ────────────────────────────────────────────────────────────
ENABLE_CACHE: bool = _cfg_bool("caching.enabled", True)

CACHE_TTL_EMBEDDING: int = _cfg_int("caching.ttl_embedding", 86400)
CACHE_TTL_SEARCH: int = _cfg_int("caching.ttl_search", 3600)
CACHE_TTL_PARSE: int = _cfg_int("caching.ttl_parse", 604800)
CACHE_TTL_EXPANSION: int = _cfg_int("caching.ttl_expansion", 21600)

# ── Metadata Enhancement ─────────────────────────────────────────────────────
ENABLE_METADATA_EXTRACTION: bool = _cfg_bool("metadata.extraction_enabled", True)
ENABLE_ENTITY_EXTRACTION: bool = _cfg_bool("metadata.entity_extraction", True)
ENABLE_DOC_CLASSIFICATION: bool = _cfg_bool("metadata.doc_classification", True)
ENABLE_TOPIC_TAGGING: bool = _cfg_bool("metadata.topic_tagging", True)
ENABLE_LANGUAGE_DETECTION: bool = _cfg_bool("metadata.language_detection", True)

METADATA_MODEL: str = _cfg_str("metadata.model", "")
MAX_ENTITIES_PER_CHUNK: int = _cfg_int("metadata.max_entities_per_chunk", 10)
MAX_TOPICS_PER_CHUNK: int = _cfg_int("metadata.max_topics_per_chunk", 5)

# ── Evaluation ───────────────────────────────────────────────────────────────
EVAL_ENABLED: bool = _cfg_bool("evaluation.enabled", False)
EVAL_DATASET_PATH: str = _cfg_str("evaluation.golden_set_path", "tests/fixtures/evaluation.xml")
EVAL_OUTPUT_DIR: str = _cfg_str("evaluation.output_dir", "eval_reports")
EVAL_TOP_K: int = _cfg_int("evaluation.top_k", 10)
EVAL_RUN_RAGAS: bool = _cfg_bool("evaluation.run_ragas", False)
EVAL_LOG_TIMING: bool = _cfg_bool("evaluation.log_timing", False)

# ── Answer Generation ─────────────────────────────────────────────────────────
ENABLE_ANSWER: bool = _cfg_bool("answer.enabled", False)
ANSWER_MAX_CONTEXT_CHARS: int = _cfg_int("answer.max_context_chars", 12000)

# ── Startup checks ────────────────────────────────────────────────────────────


def _run_startup_checks() -> None:
    if EMBED_MODEL == EMBED_MODEL_FALLBACK:
        _log.warning(
            "embedding.model and embedding.fallback_model are identical (%s) — fallback will have no effect.",
            EMBED_MODEL,
        )
    if RERANK_MODEL == RERANK_MODEL_FALLBACK:
        _log.warning(
            "reranker.model and reranker.fallback_model are identical (%s) — fallback will have no effect.",
            RERANK_MODEL,
        )
    if ENABLE_CONTEXTUAL_RETRIEVAL:
        _log.warning("contextual_retrieval is on — doubles embedding cost")
    if ENABLE_QUERY_EXPANSION and (ENABLE_HYDE or ENABLE_MULTI_QUERY or ENABLE_QUERY_REWRITE):
        count = sum([ENABLE_HYDE, ENABLE_MULTI_QUERY, ENABLE_QUERY_REWRITE])
        _log.warning("Query expansion on with %d sub-techniques — %d+ LLM calls per search", count, count)


_run_startup_checks()
