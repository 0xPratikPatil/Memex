"""Memex RAG Engine — backend logic for ingestion, retrieval, and generation.

Usage::

    from memex.engine.core.config import config
    from memex.engine.core.pipeline import RAGEngine

    engine = RAGEngine()
    engine.hybrid_search("example query")
"""

from memex.engine.core.pipeline import RAGEngine
from memex.engine.ingestion.context import ContextGenerator
from memex.engine.ingestion.hashing import dedup_chunks
from memex.engine.ingestion.ingestion import IngestionOrchestrator
from memex.engine.ingestion.loader import ConversionResult, parse_file
from memex.engine.retrieval.expansion import ExpandedQuery, QueryExpander

__all__ = [
    "ContextGenerator",
    "ConversionResult",
    "ExpandedQuery",
    "IngestionOrchestrator",
    "QueryExpander",
    "RAGEngine",
    "dedup_chunks",
    "parse_file",
]
