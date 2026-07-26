"""Memex - Personal RAG MCP Server.

A production-ready MCP server for Retrieval-Augmented Generation with
Docling document conversion, Qdrant vector storage, and Ollama embeddings.
"""

from __future__ import annotations

__version__ = "0.3.0"
__author__ = "Pratik"

# Core modules
from . import config
from . import pipeline
from . import server
from . import docling_client

# Services
from .services import (
    cache,
    contextual_retrieval,
    evaluation,
    metadata_extractor,
    query_expansion,
)

__all__ = [
    "config",
    "pipeline",
    "server",
    "docling_client",
    "cache",
    "contextual_retrieval",
    "evaluation",
    "metadata_extractor",
    "query_expansion",
]
