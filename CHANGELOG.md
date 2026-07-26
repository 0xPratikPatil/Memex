# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Env-driven configuration**: All settings flow from env vars (env > .env > default)
  - `python-dotenv` auto-loads `.env` at startup
  - Docker ports use `${VAR:-default}` syntax
  - Service URLs constructed from `*_PORT` env vars, overridable via `*_URL`
  - Comprehensive `.env.example` with all 60+ settings documented

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
  - Professional Makefile with 12 targets + `make help`

- **Unified chat model**: All LLM features fall back to `CHAT_MODEL=qwen2:5.5b`
  - Context retrieval, query expansion, metadata extraction all use one model
  - No more broken fallback to embedding-only `bge-m3`

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
- Provider system (`SPARSE_PROVIDER`, `RERANK_PROVIDER`)
- Configurable ports (`QDRANT_PORT`, `OLLAMA_PORT`, etc.)
- `.env.example` comprehensive reference
- MIT License, GitHub Actions CI, Dependabot, pre-commit hooks

### Fixed
- Import paths after restructuring
- Docker configuration for new file locations
- Environment variable documentation

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
