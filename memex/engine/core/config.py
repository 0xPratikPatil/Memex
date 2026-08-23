"""Central configuration — config.yaml is the single source of truth.

Env vars are used ONLY for ``${VAR}`` substitution inside config.yaml values
(e.g. ``api_key: ${OPENAI_API_KEY}``). No env var fallback for config keys.

``MEMEX_CONFIG`` env var can override the config file path (default: config.yaml).
"""

from __future__ import annotations

import logging
import os
import sys
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

# ── Converter engine ──────────────────────────────────────────────────────────
CONVERTER_ENGINE: str = _cfg_str("converter.engine", "markitdown")
CONVERTER_MAX_CONCURRENT: int = 2  # cap parallel conversions

# ── MarkItDown (CPU-only converter) ───────────────────────────────────────────
MARKITDOWN_URL: str = _cfg_str("converter.markitdown_url", "http://localhost:5003")
MARKITDOWN_TIMEOUT: float = _cfg_float("converter.markitdown_timeout", 600.0)

# ── OCR (scanned PDFs that MarkItDown can't extract) ──────────────────────────
OCR_URL: str = _cfg_str("converter.ocr_url", "http://localhost:5004")
OCR_MODEL: str = _cfg_str("converter.ocr_model", "pp-ocrv6-small")
OCR_TIMEOUT: float = _cfg_float("converter.ocr_timeout", 900.0)

# ── Legacy constants (kept for backward compat, unused by new code) ──────────
MARKER_URL: str = ""
MARKER_TIMEOUT: float = 300.0
MARKER_MODE: str = "balanced"
MARKER_FORCE_OCR: bool = False
OCR_FALLBACK: bool = True
OCR_WORKERS: int = 1
OCR_MAX_CONCURRENT: int = 1
OCR_RENDER_SCALE: float = 1.5
OCR_LIMIT_SIDE_LEN: int = 1280
DOCLING_TIMEOUT: float = 300.0
DOCLING_API_KEY: str = ""
DOCLING_SERVE_MAX_SYNC_WAIT: int = 120
DOCLING_SKIP_ON_TIMEOUT: bool = False
DOCLING_ENRICH_CODE: bool = False
DOCLING_ENRICH_FORMULA: bool = False
DOCLING_PICTURE_CLASSIFY: bool = True
DOCLING_CHART_EXTRACT: bool = False
DOCLING_IMAGE_EXPORT: str = "embedded"
DOCLING_PDF_BACKEND: str = "docling_parse"
DOCLING_TABLE_MODE: str = "accurate"
DOCLING_TABLE_STRUCTURE: bool = True
DOCLING_NUM_WORKERS: int = 0
DOCLING_MAX_CONCURRENT: int = 2

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
HTTP_MAX_RETRIES: int = _cfg_int("http.max_retries", 3)
HTTP_RETRY_BACKOFF: float = _cfg_float("http.retry_backoff", 0.5)
LLM_TIMEOUT: float = _cfg_float("llm.timeout", HTTP_TIMEOUT)
LLM_READ_TIMEOUT: float = _cfg_float("llm.read_timeout", 60.0)

# Connection-level failures (server restart, dropped keep-alive) need a longer
# retry window than HTTP error codes — a container restart takes 30-60s.
HTTP_TRANSPORT_MAX_RETRIES: int = _cfg_int("http.transport_max_retries", 5)
HTTP_TRANSPORT_RETRY_BACKOFF: float = _cfg_float("http.transport_retry_backoff", 2.0)

# ── Ingestion pipeline settings ───────────────────────────────────────────────
INGEST_TIMEOUT_PARSE: float = _cfg_float("ingestion.timeout_parse", 120.0)
INGEST_TIMEOUT_TOTAL: float = _cfg_float("ingestion.timeout_total", 300.0)
MAX_CONCURRENT_PARSES: int = _cfg_int("ingestion.max_concurrent_parses", 3)
MAX_CONCURRENT_SYNC: int = _cfg_int("ingestion.max_concurrent_sync", 8)

# ── Automatic retry queue ─────────────────────────────────────────────────────
RETRY_BACKOFF_SECONDS: int = _cfg_int("retry.backoff_seconds", 300)
RETRY_MAX_ATTEMPTS: int = _cfg_int("retry.max_attempts", 5)

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
ENABLE_RERANKING: bool = _cfg_bool("reranker.enabled", True)

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
CONTEXT_BATCH_SIZE: int = _cfg_int("contextual_retrieval.batch_size", 5)
# Cap sequential LLM batches per document — beyond this, remaining batches get
# section-header fallback (prevents pathological docs from dozens of LLM calls).
CONTEXT_MAX_BATCHES: int = _cfg_int("contextual_retrieval.max_batches", 8)

# ── GPU coordination ──────────────────────────────────────────────────────────
# Marker and Ollama share the GPU. When VRAM is tight, GpuLock enforces
# mutual exclusion (evict Ollama before a marker job; Ollama reloads on
# demand). On large GPUs the requester's footprint always fits → no-op.
GPU_ENABLED: bool = _cfg_bool("gpu.enabled", True)
GPU_MAX_WAIT_S: float = _cfg_float("gpu.max_wait_s", 120.0)

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
ANSWER_MAX_CONTEXT_CHARS: int = _cfg_int("answer.max_context_chars", 8000)

# ── Startup checks ────────────────────────────────────────────────────────────


def _stdout_is_tty() -> bool:
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def _print_notes_panel(notes: list[tuple[str, str, str]]) -> None:
    """Render startup notes as a single styled panel (CLI/TTY only).

    On non-TTY stdout (MCP stdio transport, piped output) the notes are
    logged instead — printing to stdout would corrupt the JSON-RPC stream.
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console(stderr=False, highlight=False)
    body = Text()
    for label, msg, kind in notes:
        if kind == "warn":
            body.append(" ⚠ ", style="yellow")
            body.append(label, style="bold")
            body.append(f": {msg}\n")
        else:
            body.append(" i ", style="cyan")
            body.append(label, style="bold")
            body.append(f": {msg}\n")
    border = "yellow" if any(kind == "warn" for _, _, kind in notes) else "cyan"
    panel = Panel(body, title="memex notes", title_align="left", border_style=border, expand=False)
    console.print(panel)


def _run_startup_checks() -> None:
    notes: list[tuple[str, str, str]] = []

    if EMBED_MODEL_FALLBACK and EMBED_MODEL == EMBED_MODEL_FALLBACK:
        notes.append(
            (
                "embedding",
                f"fallback_model is identical to model ({EMBED_MODEL}) — "
                'set a different model or "" to disable',
                "warn",
            )
        )
    if RERANK_MODEL_FALLBACK and RERANK_MODEL == RERANK_MODEL_FALLBACK:
        notes.append(
            (
                "reranker",
                f"fallback_model is identical to model ({RERANK_MODEL}) — "
                'set a different model or "" to disable',
                "warn",
            )
        )
    if ENABLE_CONTEXTUAL_RETRIEVAL:
        notes.append(
            (
                "contextual retrieval",
                "on — doubles embedding cost (each chunk gets an LLM context prefix)",
                "info",
            )
        )
    if ENABLE_QUERY_EXPANSION and (ENABLE_HYDE or ENABLE_MULTI_QUERY or ENABLE_QUERY_REWRITE):
        count = sum([ENABLE_HYDE, ENABLE_MULTI_QUERY, ENABLE_QUERY_REWRITE])
        notes.append(
            (
                "query expansion",
                f"on with {count} sub-techniques — {count}+ LLM calls per search",
                "info",
            )
        )

    if not notes:
        return

    if _stdout_is_tty():
        _print_notes_panel(notes)
    else:
        for label, msg, _kind in notes:
            _log.warning("%s: %s", label, msg)


_run_startup_checks()
