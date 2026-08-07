# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Inverted contextual retrieval vectors**: `dense` was embedding enriched content while `contextual_dense` embedded raw content — the complete opposite of intended behavior. Swapped assignment so `dense` = raw, `contextual_dense` = enriched. Existing collections must be re-ingested.
- **Single-batch summary bypass**: Documents with ≤ `batch_size` (10) chunks fell through to header-strategy context generation instead of using the LLM summary strategy. All small documents had empty `context_prefix`.
- **Resilience chain**: Added per-chunk fallback when batch context generation fails or returns gaps. Chain: batch LLM → per-chunk LLM → section header → empty.
- **Event-loop-is-closed crash**: httpx `AsyncClient` was created inside a temporary event loop (via `chat_sync` → `asyncio.run()` in a thread), stayed cached with `is_closed=False`, and poisoned the next `await llm.chat()`. Fixed by tracking the event loop in `OllamaLLM._get_client` and `_OpenAIBase._get_client` — client is recreated when the current loop differs.
- **`rag_extract_filters` always returned "No LLM available"**: Server was calling `extract_filters` without the `llm_call` argument. Now passes `engine._llm.chat` so metadata filters are actually extracted via the LLM.

### Changed
- **Documentation overhaul**: README prerequisites section, 4-container Docker architecture, enhanced quick start, ml-services container references across DOCKER.md/AGENTS.md/config.example.yaml.
- **Search parallelism**: Dense + sparse Qdrant queries now run concurrently in a single `ThreadPoolExecutor`. HyDE and multi-query paraphrase searches also run in the same pool. ~40% latency reduction for the Qdrant fetch phase.
- **Contextual embedding parallel**: Dense, sparse, and contextual embeddings run in a single thread pool during ingestion instead of sequentially.
- **Startup vector check**: On startup, warns if collection is missing `contextual_dense` vector.

### Added
- **Rich progress bars**: Live progress with per-file stage tracking for `memex sync`, `memex ingest`, and `memex eval` CLI commands. Sync shows file-level progress (Scanning → Reconciling → Hashing → Parsing → Ingesting → Done). Ingest shows inline per-file progress. Eval shows per-query progress.
- `FileProgress` dataclass in `memex/engine/core/progress.py` with `ProgressCallback` type alias.
- `sync()` engine accepts `progress_cb` parameter for stage-based progress reporting.
- `rag_sync` MCP tool reports progress via `ctx.report_progress()` during sync operations.
- `memex eval` now runs actual golden-set evaluation with Rich tables (was a stub).
- `_fallback_context` method (resilience chain for context generation)
- `_apply_chunk_context` helper (unified context application)
- Loop-aware httpx client creation in `OllamaLLM` and `_OpenAIBase` (OpenAI/Groq/OpenRouter)
- Regression test `test_chat_after_chat_sync_rebinds_client_to_loop`
- 4 new unit tests: `TestSingleBatchSummary`, `TestFallbackContext`
- `rich>=13,<14` added as required dependency

## [0.5.0] - 2026-07-26

### Added

- **Docling HybridChunker** (CHUNK_STRATEGY=hybrid): tokenizer-aware, structure-preserving chunking on DoclingDocument. Multi-format serialization (table→HTML, code→fenced, image→caption).
- **AGENTS.md**: project instructions for OpenCode with feature catalog and tool hints.
- Docling enrichment flags: picture classification, code/formula/chart extraction (opt-in), image export mode.
- Defensive # comment stripping in config loader.
- CHUNK_MERGE_PEERS, CHUNK_REPEAT_TABLE_HEADER, CHUNK_TYPE_FORMAT config options.
- 3 new test files: test_chunking.py (23), test_config.py (25), test_pipeline_chunking.py (6).

### Changed

- **All advanced features enabled by default**: query expansion (HyDE + rewrite + multi-query), contextual retrieval (summary strategy), Redis caching, metadata extraction (entities, topics, classification, language).
- **Chunk defaults**: CHUNK_SIZE 512→1024, CHUNK_OVERLAP 50→128, CHUNK_STRATEGY recursive→hybrid.
- **Search**: SEARCH_TOP_K 20→30, MULTI_QUERY_COUNT stays at 3.
- **Embedding**: EMBED_BATCH_SIZE 32→64.
- **Docker**: ML services now use uv instead of pip for faster builds. Docling serve image unchanged (pre-built ghcr pull).
- **docling added to core dependencies** (no extra needed for HybridChunker).
- Contextual retrieval defaults to summary strategy (was header-only).
- `.env` inline comments removed — values with trailing `# comment` broke python-dotenv parsing.
- **310 tests** (257 unit + 53 integration) — all pass against live Docker services.

### Fixed

- Dockerfile: curl install moved before USER switch (was failing on non-root).
- `.env` model fallback vars (HYDE_MODEL, CONTEXT_MODEL, etc.) stripped inline comments that broke `or config.CHAT_MODEL` fallback.
- Integration tests updated for new defaults (CONTEXT_STRATEGY, query expansion flags).
- `_parse_response` now correctly populates json_content, html_content, text_content from Docling serve response.

## [0.3.0] - 2026-07-26

### Added
- GPU passthrough for Docling, Ollama, and MCP services
- File deduplication with SHA256 content hashing
- Progress visibility during ingestion
- Contextual retrieval with document context prefixes
- Redis caching layer
- Metadata extraction (entities, topics, language)
- Evaluation framework with RAGAS integration
- Comprehensive test suite (169 tests)

### Changed
- Increased Docling timeout to 300s for GPU warmup
- Increased HTTP timeout to 60s for embedding under load
- Updated Qdrant configuration for multiple vectors

## [0.2.0] - 2026-07-25

### Added
- Hybrid search (dense + sparse) with RRF
- Cross-encoder reranking
- Recursive chunking strategy
- MCP tools for ingestion, search, listing, and deletion
- Docker Compose setup with 5 services
- File server for host file access

### Changed
- Improved chunking to respect semantic boundaries
- Enhanced error handling with actionable messages

## [0.1.0] - 2026-07-24

### Added
- Initial project structure
- Basic RAG pipeline
- Docling document conversion
- Qdrant vector storage
- Ollama embeddings
