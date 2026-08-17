# AGENTS.md — Memex RAG Project Instructions

OpenCode should proactively use the following features when working with this codebase.

## Package Structure

```
memex/
├── __init__.py                  # v0.5.0
├── cli.py                       # Typer CLI: ingest, sync, eval, serve
├── mcp/
│   ├── server.py                # 13 MCP tools via FastMCP
│   └── schemas.py               # Pydantic request/response models
└── engine/
    ├── core/
    │   ├── config.py            # All config keys as module-level constants
    │   ├── pipeline.py          # RAGEngine: embedding, hybrid search, MMR, rerank, ingest
    │   ├── progress.py          # FileProgress dataclass + PipelineStage enum + ProgressCallback
    │   ├── errors.py            # Typed exception hierarchy (MemexError subclasses)
    │   ├── logging_setup.py     # Colored/structured logging (Rich handler, JSON mode)
    │   └── yaml_config.py       # YamlConfig with dot-notation + ${VAR} substitution
    ├── ingestion/
    │   ├── loader.py            # parse_file / parse_url → Marker or MarkItDown conversion
    │   ├── context.py            # Context prefix generation (contextual retrieval)
    │   ├── embedding.py          # EmbeddingService: batch embed via Ollama/OpenAI/HF
    │   ├── hashing.py            # SHA256 content-hash dedup
    │   ├── ingestion.py          # IngestionOrchestrator for batch ingest
    │   ├── marker_client.py      # Marker HTTP client (submit→poll→fetch with GpuLock)
    │   ├── markitdown_client.py  # MarkItDown HTTP client (CPU-only, multi-format)
    │   ├── splitter.py           # Recursive/fixed chunking (fallback when HybridChunker unavailable)
    │   └── status.py             # FileStatusStore: Qdrant-backed per-file status state machine
    ├── retrieval/
    │   ├── cosine.py             # Cosine similarity utilities
    │   ├── expansion.py          # QueryExpander: HyDE, Multi-Query, Query Rewrite
    │   ├── filter.py             # get_filter_context, extract_filters (metadata filter)
    │   ├── mmr.py                # Maximal Marginal Relevance (numpy or pure Python)
    │   └── reranker.py           # FastAPI ML services: BM25 sparse + Cross-Encoder/Causal-LM rerank
    ├── generation/
    │   └── answers.py            # generate_answer with citations + refusal detection
    ├── evaluation/
    │   ├── golden.py             # GoldenSet: YAML/JSON golden-set loader
    │   ├── metrics.py            # keyword_coverage, match_source
    │   ├── runner.py             # Evaluation runner
    │   └── sweep.py              # Config sweep (multi-variant comparison)
    ├── metadata/
    │   └── extractor.py          # MetadataExtractor: entities, topics, language, classification
    ├── sources/
    │   ├── local.py              # Local directory source
    │   ├── s3.py                 # S3 bucket source
    │   └── sync.py               # Sync engine: reconcile collection against sources
    ├── llm/
    │   ├── __init__.py           # get_llm(), get_embedder() factories
    │   ├── base.py               # LLMProvider, EmbedProvider abstract classes
    │   ├── ollama.py             # Ollama LLM + embedder
    │   ├── openai.py             # OpenAI LLM + embedder
    │   ├── anthropic.py          # Anthropic LLM
    │   ├── groq.py               # Groq LLM
    │   ├── google.py             # Google LLM
    │   ├── openrouter.py         # OpenRouter LLM
    │   ├── huggingface.py        # HuggingFace embedder
    │   └── fastembed.py          # FastEmbed embedder
    └── utils/
```

## Available Features

All features are controlled via `config.yaml`. The master toggle for each group is listed first.

### Query Expansion (`query_expansion.enabled`)
- **HyDE** (`query_expansion.hyde`): LLM generates a hypothetical answer document, embeds it, and searches alongside the original query. Improves recall for conceptual queries.
- **Query Rewrite** (`query_expansion.query_rewrite`): LLM expands ambiguous or conversational queries into well-formed search queries.
- **Multi-Query** (`query_expansion.multi_query`, `query_expansion.multi_query_count`=3): Generates N paraphrases of the query, searches each independently, fuses via RRF.
- Does not activate in MMR search mode (server.py line 487).

### Search (`search.mode`)
- **Hybrid** (`search.mode=hybrid`): Dense (qwen3-embedding:0.6b, 1024d, fallback bge-m3) + Sparse (BM25 via Docker ml-services or in-process fastembed) + RRF fusion (k=60) + rerank (Docker ml-services or in-process sentence-transformers, Qwen/Qwen3-Reranker-0.6B, fallback BAAI/bge-reranker-base).
- **Similarity** (`search.mode=similarity`): Dense only.
- **MMR** (`search.mode=mmr`): Maximal Marginal Relevance for diverse results. Parameters: `search.mmr.fetch_k`=20, `search.mmr.lambda_mult`=0.5.
- **Search Cache** (`caching.enabled`, `caching.ttl_search`=3600): In-memory LRU cache with Redis persistent layer caches full result sets for repeated queries.

### Ingestion
- **Marker conversion** (`converter.engine=marker`): marker-pdf + Surya OCR in Docker — faster (2.9-7.4 pg/s) and higher quality (76 vs 50 olmocr-bench) than Docling. Recursive chunking is the default.
- **MarkItDown conversion** (`converter.engine=markitdown`): Microsoft's MarkItDown in Docker — CPU-only, no GPU contention. Handles DOCX, PPTX, XLSX, HTML, EPUB, images (via OCR), audio (via Whisper), CSV, JSON, XML, ZIP. Good for mixed-format corpora where GPU is needed for other tasks.
- **Multi-format Embedding**: Table chunks → HTML, code chunks → fenced markdown, images → `[Image: caption]`.
- **Legacy Docling path** (`converter.engine=docling`): kept for rollback only.
- **Content-Hash Dedup** (`ingestion/hashing.py`): SHA256-based dedup prevents re-indexing identical content. Partial ingest recovery on crash.
- **Embedding Cache** (`caching.enabled`, `caching.ttl_embedding`=86400): In-memory LRU caches dense vectors (Redis persistent layer).

### Contextual Retrieval (`contextual_retrieval.enabled`)
- **Strategy** (`contextual_retrieval.strategy=summary`): LLM-generated context prefixes for each chunk, improving embedding quality.
- Parameters: `contextual_retrieval.max_tokens`=50, `contextual_retrieval.batch_size`=10.

### Metadata Extraction (`metadata.extraction_enabled`)
- Entity extraction, doc classification, topic tagging, language detection (each individually togglable).
- Parameters: `metadata.max_entities_per_chunk`=10, `metadata.max_topics_per_chunk`=5.

### Answer Generation (`answer.enabled`)
- Cited Answers with `[N]` citations, refusal detection (`answer.refusal_sentinel="INSUFFICIENT_CONTEXT"`), citation confidence scoring.
- Parameter: `answer.max_context_chars`=12000.

### Document Sources & Sync
- **Pluggable Sources** (`sources` section in config.yaml): Local directories (`type: local`) + S3 buckets (`type: s3`).
- **Sync Engine** (`rag_sync`): Reconciles collection against sources — adds new, replaces changed, removes deleted files. Safety: if any source fails to list, all deletions are suppressed for that run.

## MCP Tools (14 available)

| Tool | Use when |
|------|----------|
| `rag_ingest_file` | Ingest a local document by path |
| `rag_ingest_url` | Ingest a document from a URL |
| `rag_ingest_batch` | Ingest multiple files/URLs at once |
| `rag_query` | Hybrid/MMR search with expansion, reranking, citation answers — use for all search needs |
| `rag_list_documents` | See what documents are indexed |
| `rag_collection_stats` | Check collection size and config |
| `rag_delete_document` | Remove a document and its chunks |
| `rag_service_status` | Check Docker service health |
| `rag_sync` | Sync collection against configured sources |
| `rag_get_filter_context` | Show available metadata fields and values |
| `rag_extract_filters` | Extract metadata filters from natural language |
| `rag_eval` | Run golden-set evaluation |
| `rag_eval_sweep` | Compare multiple retrieval configs side by side |
| `rag_processing_status` | Show file processing status (pending, converting, done, error) |

## Key Config (config.yaml)

All configuration lives in `config.yaml`. Copy `config.example.yaml` to `config.yaml` and customize. Env vars in `.env` are used only for `${VAR}` substitution in config.yaml values — not as fallbacks.

| Path | Default | Notes |
|------|---------|-------|
| `vectorstore.url` | `http://localhost:6333` | Qdrant vector DB |
| `vectorstore.collection` | `memex` | Collection name |
| `embedding.model` | `qwen3-embedding:0.6b` | Ollama embedding model |
| `embedding.fallback_model` | `bge-m3` | Fallback if primary fails |
| `embedding.provider` | `ollama` | ollama / openai / huggingface / fastembed |
| `embedding.dimensions` | `1024` | Vector dimensions |
| `embedding.batch_size` | `64` | Batch size for embedding |
| `llm.provider` | `ollama` | ollama / openai / openrouter / anthropic / groq / google |
| `llm.model` | `qwen2.5:1.5b` | Chat model |
| `chunking.strategy` | `hybrid` | hybrid / recursive / fixed |
| `chunking.size` | `1024` | Target tokens per chunk |
| `chunking.overlap` | `128` | Token overlap |
| `chunking.min_length` | `30` | Minimum chunk length |
| `chunking.tokenizer` | `Qwen/Qwen3-Embedding-0.6B` | HF tokenizer for chunk sizing |
| `converter.engine` | `marker` | marker (recommended) / markitdown (CPU-only, multi-format) / docling (legacy) |
| `converter.marker_url` | `http://localhost:5001` | Marker service |
| `converter.marker_mode` | `fast` | fast (default) / balanced (GPU) |
| `converter.marker_force_ocr` | `false` | Force OCR on all pages (scanned-only corpora) |
| `converter.marker_timeout` | `300.0` | Conversion timeout (seconds) |
| `converter.markitdown_url` | `http://localhost:5003` | MarkItDown service |
| `converter.markitdown_timeout` | `30.0` | Conversion timeout (seconds) |
| `converter.docling_timeout` | `300.0` | Conversion timeout (seconds) |
| `converter.docling_picture_classify` | `true` | Image classification |
| `converter.docling_enrich_code` | `false` | Code extraction (needs serve-side model) |
| `converter.docling_enrich_formula` | `false` | Formula extraction (needs serve-side model) |
| `converter.docling_chart_extract` | `false` | Chart extraction (needs serve-side model) |
| `reranker.enabled` | `true` | Cross-encoder/causal-LM rerank |
| `reranker.model` | `Qwen/Qwen3-Reranker-0.6B` | Primary reranker |
| `reranker.fallback_model` | `BAAI/bge-reranker-base` | Fallback reranker |
| `reranker.type` | `auto` | cross-encoder / causal-lm / auto |
| `sparse.model` | `Qdrant/bm25` | BM25 sparse model (Docker ml-services or in-process fastembed) |
| `query_expansion.enabled` | `true` | Master toggle for HyDE + rewrite + multi-query |
| `query_expansion.hyde` | `true` | Hypothetical document embedding |
| `query_expansion.query_rewrite` | `true` | Query rewriting |
| `query_expansion.multi_query` | `true` | Multi-query paraphrasing |
| `query_expansion.multi_query_count` | `3` | Number of paraphrases |
| `contextual_retrieval.enabled` | `true` | Context prefixes on chunks |
| `contextual_retrieval.strategy` | `summary` | Context generation strategy |
| `contextual_retrieval.max_tokens` | `50` | Max tokens in context prefix |
| `contextual_retrieval.batch_size` | `10` | Batch size for context generation |
| `metadata.extraction_enabled` | `true` | Master toggle for all extractors |
| `metadata.entity_extraction` | `true` | Entity extraction |
| `metadata.doc_classification` | `true` | Document type classification |
| `metadata.topic_tagging` | `true` | Topic tagging |
| `metadata.language_detection` | `true` | Language detection |
| `caching.enabled` | `true` | In-memory LRU cache |
| `caching.ttl_embedding` | `86400` | Embedding cache TTL |
| `caching.ttl_search` | `3600` | Search cache TTL |
| `caching.ttl_parse` | `604800` | Parse cache TTL |
| `caching.ttl_expansion` | `21600` | Query expansion cache TTL |
| `search.mode` | `hybrid` | similarity / hybrid / mmr |
| `search.mmr.fetch_k` | `20` | MMR candidate pool size |
| `search.mmr.lambda_mult` | `0.5` | MMR relevance/diversity balance |
| `answer.enabled` | `false` | Citation-based answer generation |
| `answer.max_context_chars` | `12000` | Max context for answer generation |
| `answer.refusal_sentinel` | `INSUFFICIENT_CONTEXT` | Refusal marker |
| `mcp.character_limit` | `25000` | Max response characters |
| `evaluation.log_timing` | `false` | Per-stage pipeline timing in logs |

## Architecture

```
MCP Server (host process, uv run memex)
  │
  ├── HTTP ──► Docker: Ollama (:11434)
  │               embedding (qwen3-embedding:0.6b) + chat (qwen2.5:1.5b)
  │
  ├── HTTP ──► Docker: Marker (:5001)
  │               GPU document conversion (job-based: thin server +
  │               per-job subprocess, crash-proof, no timeouts)
  │
  ├── HTTP ──► Docker: MarkItDown (:5003)
  │               CPU-only document conversion (DOCX, PPTX, XLSX, HTML, EPUB,
  │               images, audio, CSV, JSON, XML, ZIP — no GPU contention)
  │
  ├── HTTP ──► Docker: Qdrant (:6333)
  │               vector DB (1024d HNSW index)
  │
  ├── HTTP ──► Docker: ML Services (:5002) [Docker mode]
  │               BM25 sparse embeddings + reranker
  │
  ├── HTTP ──► Docker: Redis (:6379)
  │               persistent cache layer
  │
  └── In-process (via [local] extras):
       ├── fastembed — BM25 sparse embeddings
       └── sentence-transformers — CrossEncoder/Causal-LM reranker
```

## Docker Best Practices

- **Host-only binding**: All ports bind to `127.0.0.1` — no external exposure.
- **Named volumes**: `qdrant_data`, `ollama_data` — survive `docker compose down`.
- **Health checks**: Every service has `interval`, `timeout`, `start_period`, and `retries`.
- **Resource limits**: `deploy.resources.limits` and `reservations` on every service.
- **Logging**: `json-file` driver with `max-size: 10m` and `max-file: 3` on every service.
- **Security**: `no-new-privileges:true` on every service, no privileged containers.
- **Restart policy**: `unless-stopped` for all persistent services.
- **stop_grace_period**: 30s (qdrant, ml-services), 60s (ollama/marker).
- **Single network**: All services share the `backend` bridge network (MCP server runs on host).
- **Tmpfs for /tmp**: Every service has tmpfs mount — keeps temp writes off the container filesystem.
- **Container names**: `memex-qdrant`, `memex-ollama`, `memex-marker`, `memex-ml`, `memex-markitdown`.
- **Service labels**: Each service has `com.memex.service` and `com.memex.description` labels.
- **Override file**: `compose.override.yaml` is auto-loaded with tighter dev health checks.
- **Ollama runs in Docker only** — never install Ollama on the host.

### When to Rebuild vs Restart

| Change | Action |
|--------|--------|
| Compose config, env vars, or labels | `docker compose up -d` (restart) |
| Ollama models | `docker compose exec ollama ollama pull <model>` + restart if parallel changed |
| Volume data | `docker compose down -v` (destroys all persisted data) |
| MCP server Python (host) | Just restart `uv run memex` — no Docker rebuild needed |

### Anti-Patterns (NEVER do)

- Never use `:latest` tags — always pin to specific versions
- Never bake secrets/API keys into image layers
- Never expose services on `0.0.0.0` — always bind to `127.0.0.1`
- Never store data in container filesystem — always use named volumes
- Never install Ollama on host — Docker-only

## Development Commands

```bash
./setup.sh                    # Bootstrap Docker + models + deps
uv sync --extra local         # Install Python deps (incl. in-process ML: fastembed, sentence-transformers)
uv sync --extra dev --extra test  # Install dev + test deps
uv run memex                  # Start MCP server (stdio transport)
uv run memex serve            # Explicit serve command
make test                     # Unit tests (no Docker needed): pytest tests/unit/
make test-all                 # Unit + integration tests
make lint                     # Run ruff linter
make fmt                      # Auto-format + fix lint (ruff check --fix + ruff format)
make typecheck                # Run mypy
make e2e                      # End-to-end verification
make clean                    # Remove caches and build artifacts
```

## CLI Commands

```bash
# Start the MCP server
uv run memex
uv run memex serve -c config.yaml

# Ingest files or directories (shows Rich progress bar per file)
memex ingest /path/to/docs --recursive
memex ingest report.pdf --verbose

# Sync collection against configured sources (shows Rich progress bar with file stages)
memex sync --dry-run
memex sync --source-name docs

# Evaluate retrieval quality (shows per-query progress + results table)
memex eval golden.yaml --top-k 5
```

### Progress Tracking

All CLI commands use Rich live progress bars. The `sync` command exposes a `progress_cb` callback for per-file stage reporting:

```python
from memex.engine.core.progress import FileProgress, ProgressCallback


async def my_callback(progress: FileProgress) -> None:
    print(f"[{progress.stage.value}] {progress.file_path} ({progress.file_idx}/{progress.total_files})")
```

Sync stages: `Scanning` → `Reconciling` → `Hashing` → `Parsing` → `Ingesting` → `Done` | `Error` | `Deleting`

## LLM Providers

The LLM and embedding providers are configured by `llm.provider` and `embedding.provider` in config.yaml. Both default to `ollama` if unknown.

### LLM Providers (`llm.provider`)

| Provider | Requires | Notes |
|----------|----------|-------|
| `ollama` | Docker Ollama | Default. Uses `llm.base_url` + `llm.model` |
| `openai` | `llm.api_key` | OpenAI-compatible API |
| `openrouter` | `llm.api_key` | OpenRouter API |
| `anthropic` | `llm.api_key` | Anthropic API |
| `groq` | `llm.api_key` | Groq API |
| `google` | `llm.api_key` | Google AI API |

### Embedding Providers (`embedding.provider`)

| Provider | Requires | Notes |
|----------|----------|-------|
| `ollama` | Docker Ollama | Default. Uses `embedding.base_url` + `embedding.model` |
| `openai` | `embedding.api_key` | OpenAI-compatible API |
| `huggingface` | `embedding.model` | HuggingFace Inference API |
| `fastembed` | `uv sync --extra local` | In-process, no network needed |

## Practical Testing Setup

### First-time setup
```bash
./setup.sh                    # bootstraps Docker, pulls models, verifies everything
```

### Start services
```bash
docker compose up -d          # start all backend services
docker compose ps             # verify all healthy
```

### Pull / manage models
```bash
docker compose exec -T ollama ollama list           # list downloaded models
docker compose exec -T ollama ollama pull qwen3-embedding:0.6b  # pull embedding model
docker compose exec -T ollama ollama pull qwen2.5:1.5b  # pull chat model
docker compose exec -T ollama ollama rm <model>     # remove a model
```

### Verify Ollama is working
```bash
curl http://localhost:11434/api/tags
curl -s -X POST http://localhost:11434/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-embedding:0.6b","prompt":"test"}' | jq .
curl -s -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5:1.5b","messages":[{"role":"user","content":"hi"}],"stream":false}' | jq .
```

### Run the MCP server
```bash
uv run memex                  # starts MCP server, connects to Docker services
```

### Run tests
```bash
make test                     # unit tests only (no Docker needed)
make test-all                 # unit + integration tests
make e2e                      # end-to-end verification (needs Docker)
pytest tests/unit/ -v         # unit tests only
pytest tests/integration/ -v  # integration tests (needs Docker services running)
```

### Useful Docker commands
```bash
docker compose logs -f ollama           # tail Ollama logs
docker compose logs -f marker          # tail Marker logs
docker compose restart ollama           # restart Ollama container
docker compose down && docker compose up -d  # full restart
docker volume ls                        # check volumes exist
```

## Logging

### Colored / structured logging

All logging goes through `memex.engine.core.logging_setup.setup_logging()` (Rich handler
with level colors, timestamps, and structured `key=value` extras on each record). Call
`setup_logging()` once at process start (CLI and MCP server already do).

- `MEMEX_LOG_JSON=1` → JSON-lines output for machine ingestion (Logstash/Datadog/OTel).
- `MEMEX_LOG_PLAIN=1` → plain text (CI, non-TTY, piped output).
- `--verbose` / `EVAL_LOG_TIMING` → DEBUG level.

Structured extras are passed via `logger.info("msg", extra={"source": ..., "stage": ...})`
and are rendered as `source=/tmp/a.pdf stage=Converting` (text) or JSON keys (json mode).
The `hint` extra is reserved for actionable remediation text.

### Typed errors

Never raise bare `RuntimeError` for domain failures. Raise subclasses of `MemexError`
from `memex.engine.core.errors`:

- `ConfigError`, `ServiceUnavailableError`
- `ConversionError`, `ConversionTimeoutError`, `CorruptedDocumentError`, `ChunkingError`
- `EmbeddingError`, `StorageError`, `RetrievalError`, `AnswerError`
- `IngestionError` (source + stage), `SourceError`, `SyncError`, `DuplicateDocumentError`

Each carries `component` and optional `hint`. `error_context(exc)` extracts a structured
dict for observability. `MemexError` extends `RuntimeError` for backward compatibility.

### Pipeline stages

Stage names are typed via the `PipelineStage` enum in `memex.engine.core.progress`
(`Converting`, `Context`, `Metadata`, `Embedding`, `Storing`, `Deleting`, `Done`,
`Skipped`, `Error`, ...). Sync workers emit `FileProgress` through `ProgressCallback`;
per-file lifecycle ends only in a terminal stage (`Done`/`Skipped`/`Error`).

Set `evaluation.log_timing` to `true` in config.yaml to see per-stage pipeline timing in logs (embed_query, dense_search, sparse_search, rerank, total_search).
