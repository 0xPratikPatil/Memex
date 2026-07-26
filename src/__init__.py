"""Memex - Personal RAG MCP Server.

A production-ready MCP server for Retrieval-Augmented Generation with
Docling document conversion, Qdrant vector storage, and Ollama embeddings.
"""

from __future__ import annotations

__version__ = "0.3.0"
__author__ = "Pratik"

# Core modules
from . import config, docling_client, pipeline, server

# Services
from .services import (
    cache,
    contextual_retrieval,
    evaluation,
    metadata_extractor,
    query_expansion,
)

__all__ = [
    "cache",
    "config",
    "contextual_retrieval",
    "docling_client",
    "evaluation",
    "metadata_extractor",
    "pipeline",
    "query_expansion",
    "server",
]
