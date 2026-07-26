"""Memex — Personal RAG MCP Server.

Thin MCP server for Retrieval-Augmented Generation. Backend models run
in Docker (Ollama, Docling, ML Services, Qdrant, Redis). MCP only does HTTP orchestration.

Features:
- Docling HybridChunker with multi-format serialization
- Contextual retrieval (summary strategy)
- Query expansion: HyDE + rewrite + multi-query
- Metadata extraction: entities, topics, language, classification
- Redis caching: embeddings, search, parse

Usage:
    uv run memex          # start the MCP server
    ./setup.sh            # bootstrap all Docker services
"""

__version__ = "0.5.0"
__author__ = "Pratik"
