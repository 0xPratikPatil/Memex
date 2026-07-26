"""Memex — Personal RAG MCP Server.

Thin MCP server for Retrieval-Augmented Generation. Backend models run
in Docker (Ollama, Docling, ML Services, Qdrant, Redis). MCP only does HTTP orchestration.

Usage:
    uv run memex          # start the MCP server
    ./setup.sh            # bootstrap all Docker services
"""

__version__ = "0.4.0"
__author__ = "Pratik"
