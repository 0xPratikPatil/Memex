# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
