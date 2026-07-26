# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Architecture: MCP runs locally, services in Docker**
  - MCP server now runs directly on host machine (not in Docker)
  - Backend services (Qdrant, Ollama, Docling, Redis) remain in Docker
  - Eliminates Docker permission issues for file access
  - Faster development cycle (no Docker rebuild for MCP changes)
  - Added `memex_cli` package for local MCP execution

- **Simplified file handling**: Removed file server dependency, MCP now reads files directly from filesystem
  - Removed `fileserver` service from Docker Compose
  - Removed `FILE_SERVER_URL` configuration
  - Updated `rag_ingest_file` tool to read files directly
  - Improved performance by eliminating HTTP overhead for file reads

- **MCP now lightweight**: Moved ML models (BM25 sparse, cross-encoder reranker) into Docker ML services container
  - MCP only does HTTP orchestration — no model loading at startup
  - ML services run on GPU inside Docker with HTTP API endpoints
  - Added `setup.sh` bootstrap script for one-command setup
  - MCP install drops fastembed and sentence-transformers dependencies (~2GB savings)

### Added
- Query Expansion (HyDE + Multi-Query + Query Rewriting)
- Contextual Retrieval (document context prefixes for chunks)
- Caching Layer (Redis cache-aside for embeddings, search, parsing)
- Metadata Enhancement (entity extraction, classification, topics)
- Evaluation Framework (RAGAS integration, custom metrics, A/B testing)
- GitHub Actions CI/CD pipelines
- Dependabot for automated dependency updates
- Pre-commit hooks with ruff
- Makefile for common development tasks
- MIT License
- Design specifications for all new features

### Changed
- Moved config.py into src/ package
- Restructured repository for production readiness
- Updated entry point to python -m src.cli
- Improved error handling and logging
- Enhanced documentation

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
