# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Chunking moved to Docling Serve API**: `rag/chunking.py` now calls `/v1/chunk/hybrid/source` instead of using local `docling` + `docling-core` packages. All heavy processing runs in Docker.
- **Removed `docling>=2` from core dependencies**: No local docling packages needed — the Docling Serve container handles conversion and chunking.
- Updated `rag/pipeline.py` to use API-based chunking via `source_identifier` parameter instead of `docling_json`.
- Updated `memex/server.py` to remove `docling_json` from `ingest_text` calls.
- **Env-driven configuration**: All settings flow from env vars (env > .env > default)
  - `python-dotenv` auto-loads `.env` at startup
  - Docker ports use `${VAR:-default}` syntax
  - Service URLs constructed from `*_PORT` env vars, overridable via `*_URL`
  - Comprehensive `.env.example` with all 80+ settings documented

- **Architecture: MCP local + Docker backend**
  - MCP runs directly on host (stdio mode, `uv run memex`)
  - Backend (Qdrant, Ollama, Docling, ML Services, Redis) in Docker
  - `rag/` package = RAG engine; `memex/` package = MCP server
  - No file server — direct filesystem access via `pathlib`

- **MCP thin + Docker ML**
  - ML models (BM25 sparse, cross-encoder reranker) moved to Docker
  - MCP only does HTTP orchestration — no `fastembed` or `sentence-transformers`
  - Configurable providers: `SPARSE_PROVIDER`/`RERANK_PROVIDER` (http or local)

- **Docker optimization**
  - Multi-stage Dockerfile (deps → preload → runtime) with layer caching
  - ML models pre-cached at build time (instant startup, no runtime download)
  - Pinned image versions (qdrant:v1.18, ollama:0.32.4, redis:7.4.10-alpine)
  - Named volumes with documented persistence rationale
  - Professional Makefile with 14 targets + `make help`

- **Unified chat model**: All LLM features fall back to `CHAT_MODEL=qwen3.5:0.8b`
  - Context retrieval, query expansion, metadata extraction all use one model
  - No more broken fallback to embedding-only `bge-m3`

- **310 tests** (257 unit + 53 integration) — all pass against live Docker services.

### Added
- **8 MCP tools**: ingest_file, ingest_url, ingest_batch, query, list_documents,
  collection_stats, delete_document, service_status
- Query Expansion (HyDE + Multi-Query + Query Rewriting)
- Contextual Retrieval (document context prefixes for chunks)
- Caching Layer (Redis cache-aside for embeddings, search, parsing)
- Metadata Enhancement (entity extraction, classification, topics)
- Evaluation Framework (RAGAS integration, custom metrics, A/B testing)
- **Integration tests**: 53 tests (Redis cache, contextual, query expansion, metadata)
- setup.sh bootstrap script (one-command setup with model downloads)
- scripts/test_e2e.py (9/9 E2E checks)
- scripts/verify_features.py (55-check feature verification)
- Provider system (`SPARSE_PROVIDER`, `RERANK_PROVIDER`)
- Configurable ports (`QDRANT_PORT`, `OLLAMA_PORT`, etc.)
- `.env.example` comprehensive reference
- MIT License, GitHub Actions CI, Dependabot, pre-commit hooks

### Fixed
- **PDF backend case sensitivity**: `rag/docling_client.py` and `rag/chunking.py` now lowercase `DOCLING_PDF_BACKEND` before passing to Docling API (expects `dlparse_v4`, not `DLPARSE_V4`)
- **Contextual retrieval batch parsing**: `rag/services/contextual_retrieval.py` now uses numbered-line format (`1. prefix\n2. prefix`) instead of `|||` separator, which small LLMs don't follow
- **Metadata extraction JSON parsing**: `rag/services/metadata_extractor.py` now strips markdown code fences before `json.loads` — small LLMs wrap output in `\`\`\`json...\`\`\``
- **E2E test mock Context**: `scripts/test_e2e.py` now provides mock `Context` for ingest tool calls
- **Test assertions updated**: `tests/unit/test_server.py` and `test_docling_client.py` aligned with new API
- Import paths after restructuring
- Docker configuration for new file locations
- Environment variable documentation

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
