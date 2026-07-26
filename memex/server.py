"""MCP server — tool definitions for Personal RAG Engine.

Uses lazy initialisation so the server can start even if Qdrant / Ollama
are temporarily unavailable (they are contacted only on first tool call).

All tools are async for proper streamable HTTP transport support.
"""

from __future__ import annotations

import atexit
import contextlib
import enum
import json
import logging
import threading
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from rag import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-16s  %(levelname)-5s  %(message)s",
)
logger = logging.getLogger("mcp-server")

CHARACTER_LIMIT = config.CHARACTER_LIMIT

mcp = FastMCP("memex_rag")

_engine = None


def _get_engine():
    """Return the RAGEngine singleton, creating it on first call."""
    global _engine
    if _engine is None:
        from rag.pipeline import RAGEngine

        _engine = RAGEngine()
    return _engine


def _prewarm_models():
    """Load sparse model and reranker in background thread to avoid first-search cold start."""

    def _load():
        try:
            engine = _get_engine()
            engine._get_sparse_model()
            logger.info("Sparse model loaded")
            engine._get_reranker()
            logger.info("Reranker loaded")
            logger.info("Model pre-warming complete")
        except Exception as exc:
            logger.warning("Model pre-warming failed: %s", exc)

    threading.Thread(target=_load, daemon=True, name="model-prewarm").start()


def _shutdown():
    """Best-effort cleanup on process exit."""
    global _engine
    with contextlib.suppress(Exception):
        from rag import docling_client

        if _engine is not None:
            _engine.close()
            _engine = None
        docling_client.close()


atexit.register(_shutdown)


# ── Input schemas ──────────────────────────────────────────────────────────────


class ResponseFormat(enum.Enum):
    MARKDOWN = "markdown"
    JSON = "json"


# ── Helper ─────────────────────────────────────────────────────────────────────


def _truncate(text: str) -> str:
    if len(text) <= CHARACTER_LIMIT:
        return text
    truncated = text[:CHARACTER_LIMIT]
    return (
        f"{truncated}\n\n---\nResponse truncated ({len(text)} chars > {CHARACTER_LIMIT} limit). "
        "Use pagination or add filters to retrieve smaller result sets."
    )


def _format_error(exc: Exception, context: str) -> str:
    """Produce actionable error messages."""
    exc_type = type(exc).__name__
    msg = str(exc)

    if "Cannot reach Docling server" in msg or "ConnectError" in exc_type:
        return (
            f"Error: {context} failed — Docling server is unreachable. "
            f"Ensure Docling is running at {config.DOCLING_URL}. "
            "In Docker: check 'docker compose ps docling'. Locally: check the server process."
        )
    if "Cannot reach Qdrant" in msg or "Qdrant" in msg:
        return (
            f"Error: {context} failed — Qdrant is unreachable. "
            f"Ensure Qdrant is running at {config.QDRANT_URL}. "
            "In Docker: check 'docker compose ps qdrant'."
        )
    if "Cannot reach" in msg or "ConnectionRefused" in exc_type:
        return (
            f"Error: {context} failed — service unreachable. Check that all services are running (docker compose ps)."
        )

    return f"Error: {context} failed: {exc_type}: {msg}"


# ── Tools ──────────────────────────────────────────────────────────────────────


@mcp.tool(
    name="rag_ingest_file",
    title="Ingest Document",
    description="""Parse and index a document into the RAG vector database.

Accepts a file path or URL. For local files, the server reads them directly
from disk. For URLs, fetches directly.

Supports PDF, Word (docx), Markdown, HTML, and images (via OCR).
Documents are chunked recursively (respecting paragraphs, sentences, headers),
embedded (dense + sparse), and stored in Qdrant.

Args:
  - file_path_or_url (string): Local file path or URL to the document.

Returns:
  - string: Confirmation message with chunk count, or error details.

Examples:
  - Use when: "Index this PDF" -> file_path_or_url="/mnt/docs/report.pdf"
  - Use when: "Add this web page" -> file_path_or_url="https://example.com/article"

Error Handling:
  - Returns error if file not found, conversion fails, or document is empty.""",
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def rag_ingest_file(file_path_or_url: str) -> str:
    try:
        from rag.docling_client import parse_file

        engine = _get_engine()

        def _progress(msg: str, pct: int) -> None:
            logger.info("ingest [%d%%] %s", pct, msg)

        _progress("Reading file from disk...", 5)
        result = parse_file(file_path_or_url)

        if not result.ok:
            return f"Error: Docling conversion returned status '{result.status}' with errors: {result.errors}"

        _progress("Checking if already ingested...", 10)
        content_hash = engine.compute_file_hash(result.markdown.encode())
        already, chunk_count = engine.is_already_ingested(file_path_or_url, content_hash)
        if already:
            return (
                f"Already ingested '{file_path_or_url}' "
                f"({chunk_count} chunks, hash: {content_hash[:12]}...). "
                f"File unchanged — skipping."
            )

        _progress("Converting with Docling...", 15)
        count = engine.ingest_text(
            result.markdown,
            source_identifier=file_path_or_url,
            metadata={
                "content_type": file_path_or_url.rsplit(".", 1)[-1] if "." in file_path_or_url else "",
                "content_hash": content_hash,
            },
            content_hash=content_hash,
            progress_cb=_progress,
        )
        return (
            f"Successfully ingested '{file_path_or_url}'. "
            f"Created {count} chunks. "
            f"(Docling: {result.processing_time:.1f}s, "
            f"{len(result.markdown)} chars, hash: {content_hash[:12]}...)"
        )
    except Exception as exc:
        logger.exception("rag_ingest_file failed")
        return _format_error(exc, f"ingestion of '{file_path_or_url}'")


@mcp.tool(
    name="rag_ingest_url",
    title="Ingest URL",
    description="""Parse and index a document from a URL into the RAG vector database.

Fetches the document from the URL and converts it using Docling.
Supports PDF, Word (docx), Markdown, HTML, and images.

Args:
  - url (string): URL of the document to ingest.

Returns:
  - string: Confirmation message with chunk count, or error details.

Examples:
  - Use when: "Index this web page" -> url="https://example.com/article.html"
  - Use when: "Add this PDF from the web" -> url="https://example.com/report.pdf"

Error Handling:
  - Returns error if URL unreachable, conversion fails, or document is empty.""",
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def rag_ingest_url(url: str) -> str:
    try:
        from rag.docling_client import parse_url

        engine = _get_engine()
        result = parse_url(url)

        if not result.ok:
            return f"Error: Docling conversion returned status '{result.status}' with errors: {result.errors}"

        content_hash = engine.compute_file_hash(result.markdown.encode())
        already, chunk_count = engine.is_already_ingested(url, content_hash)
        if already:
            return (
                f"Already ingested '{url}' "
                f"({chunk_count} chunks, hash: {content_hash[:12]}...). "
                f"File unchanged — skipping."
            )

        count = engine.ingest_text(
            result.markdown,
            source_identifier=url,
            metadata={},
            content_hash=content_hash,
        )
        return (
            f"Successfully ingested '{url}'. "
            f"Created {count} chunks. "
            f"(Docling: {result.processing_time:.1f}s, "
            f"{len(result.markdown)} chars, hash: {content_hash[:12]}...)"
        )
    except Exception as exc:
        logger.exception("rag_ingest_url failed")
        return _format_error(exc, f"ingestion of '{url}'")


@mcp.tool(
    name="rag_ingest_batch",
    title="Batch Ingest Documents",
    description="""Batch ingest multiple files or URLs into the RAG vector database.

Processes each document sequentially. Failures on individual items do not
stop the batch — partial results are reported.

Args:
  - items (list): List of strings (file paths or URLs) to ingest.

Returns:
  - object: Map of source identifier -> status message for each item.

Error Handling:
  - Individual item failures are reported per-item without halting the batch.""",
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def rag_ingest_batch(items: list[str]) -> dict[str, str]:
    from src.docling_client import parse_file

    engine = _get_engine()
    summary: dict[str, str] = {}

    for item in items:
        try:
            result = parse_file(item)

            if not result.ok:
                summary[item] = f"Failed: Docling status '{result.status}', errors: {result.errors}"
                continue

            content_hash = engine.compute_file_hash(result.markdown.encode())
            already, chunk_count = engine.is_already_ingested(item, content_hash)
            if already:
                summary[item] = f"Skipped ({chunk_count} chunks, unchanged)"
                continue

            count = engine.ingest_text(
                result.markdown,
                source_identifier=item,
                metadata={"content_type": item.rsplit(".", 1)[-1] if "." in item else ""},
                content_hash=content_hash,
            )
            summary[item] = f"Success ({count} chunks, {result.processing_time:.1f}s conversion)"
        except Exception as exc:
            logger.exception("rag_ingest_batch item failed")
            summary[item] = f"Failed: {exc}"

    return summary


@mcp.tool(
    name="rag_query",
    title="Search RAG Knowledge Base",
    description="""Search personal documents using hybrid vector search (Dense + BM25 Sparse)
with Reciprocal Rank Fusion and optional cross-encoder reranking.

Optionally applies query expansion (HyDE, Multi-Query, Query Rewriting) to
improve recall for complex or ambiguous queries. Expansion techniques are
controlled by server-side feature flags (ENABLE_QUERY_EXPANSION, ENABLE_HYDE,
ENABLE_MULTI_QUERY, ENABLE_QUERY_REWRITE).

When contextual retrieval is enabled, the search can use context-enriched
embeddings for better retrieval of ambiguous chunks.

Supports metadata filtering when metadata extraction is enabled. Filter by
document type, topics, language, keywords, entities, or dates stored in the
Qdrant payload.

Returns relevant document chunks ranked by relevance.

Args:
  - query (string): Natural language search query (2-500 chars).
  - top_k (number): Max results to return, 1-50 (default: 5).
  - use_reranking (boolean): Apply cross-encoder reranking (default: true).
  - source_filter (string, optional): Filter results to a specific document source.
  - use_query_expansion (boolean, optional): Override server default for expansion (default: null).
  - use_contextual_search (boolean, optional): Use contextual embeddings for search
    (default: null, uses server setting).
  - response_format ('markdown' | 'json'): Output format (default: 'markdown').
  - metadata_filter (object, optional): Filter by metadata fields. Keys are
    payload field names (e.g. "doc_type", "topics", "language", "keywords",
    "entities.people", "dates"). Values are strings or lists of strings.
    Example: {"doc_type": "report", "topics": ["finance", "revenue"]}.

Returns:
  For JSON format:
  {
    "total": number,
    "count": number,
    "results": [
      {
        "id": string,
        "rrf_score": number,
        "rerank_score": number | null,
        "source": string,
        "content": string,
        "section_header": string,
        "context_prefix": string,
        "doc_type": string,
        "topics": [string],
        "language": string,
        "keywords": [string]
      }
    ]
  }
  For Markdown format: Formatted list of results with source and content.

Examples:
  - Use when: "Find revenue data" -> query="quarterly revenue figures"
  - Use when: "What did the contract say about termination?" -> query="contract termination clauses"
  - Use when: "Search only in report.pdf" -> query="revenue", source_filter="/docs/report.pdf"
  - Use when: "Find reports about finance" -> query="revenue", metadata_filter={"doc_type": "report"}

Error Handling:
  - Returns error message if search fails or Qdrant is unavailable.""",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def rag_query(
    query: str,
    top_k: int = 5,
    use_reranking: bool = True,
    source_filter: str | None = None,
    use_query_expansion: bool | None = None,
    use_contextual_search: bool | None = None,
    response_format: ResponseFormat = ResponseFormat.MARKDOWN,
    metadata_filter: dict[str, str | list[str]] | None = None,
) -> Any:
    try:
        engine = _get_engine()

        # Determine whether to use query expansion
        expansion_enabled = use_query_expansion if use_query_expansion is not None else config.ENABLE_QUERY_EXPANSION

        expanded = None
        if expansion_enabled:
            from rag.services.query_expansion import QueryExpander

            expander = QueryExpander(engine._get_ollama())
            expanded = expander.expand(query)

        results = engine.hybrid_search(
            query=query,
            top_k=top_k,
            rerank=use_reranking,
            source_filter=source_filter,
            metadata_filter=metadata_filter,
            expanded_query=expanded,
            use_contextual_search=use_contextual_search,
        )

        if not results:
            if response_format == ResponseFormat.JSON:
                return json.dumps({"total": 0, "count": 0, "results": []})
            return f"No results found for '{query}'."

        if response_format == ResponseFormat.JSON:
            output = {
                "total": len(results),
                "count": len(results),
                "results": [
                    {
                        "id": r["id"],
                        "rrf_score": r.get("rrf_score"),
                        "rerank_score": r.get("rerank_score"),
                        "source": r["source"],
                        "content": r["content"],
                        "section_header": r.get("section_header", ""),
                        "context_prefix": r.get("context_prefix", ""),
                        "doc_type": r.get("doc_type", ""),
                        "topics": r.get("topics", []),
                        "language": r.get("language", ""),
                        "keywords": r.get("keywords", []),
                    }
                    for r in results
                ],
            }
            return _truncate(json.dumps(output, indent=2))

        lines = [f"# Search Results for '{query}'", ""]
        for i, r in enumerate(results, 1):
            score_parts = []
            if "rrf_score" in r:
                score_parts.append(f"RRF: {r['rrf_score']:.4f}")
            if "rerank_score" in r:
                score_parts.append(f"rerank: {r['rerank_score']:.4f}")
            score_str = " | ".join(score_parts) if score_parts else "N/A"

            header = r.get("section_header", "")
            source_label = f"{r['source']}"
            if header:
                source_label += f" — {header}"

            lines.append(f"## {i}. {source_label}")
            lines.append(f"**Score**: {score_str}")

            # Show metadata when available
            meta_parts = []
            if r.get("doc_type"):
                meta_parts.append(f"Type: {r['doc_type']}")
            if r.get("topics"):
                meta_parts.append(f"Topics: {', '.join(r['topics'])}")
            if r.get("language"):
                meta_parts.append(f"Lang: {r['language']}")
            if r.get("keywords"):
                meta_parts.append(f"Keywords: {', '.join(r['keywords'][:5])}")
            if meta_parts:
                lines.append(f"**Metadata**: {' | '.join(meta_parts)}")

            lines.append("")
            lines.append(r["content"])
            lines.append("")
        return _truncate("\n".join(lines))
    except Exception as exc:
        logger.exception("rag_query failed")
        return _format_error(exc, "search")


@mcp.tool(
    name="rag_delete_document",
    title="Delete Document",
    description="""Remove a document and all its chunks from the vector database.

Deletes all chunks whose source matches the given identifier.

Args:
  - source_identifier (string): File path or URL of the document to delete.

Returns:
  - string: Confirmation message or error details.

Examples:
  - Use when: "Remove the old report" -> source_identifier="/docs/old_report.pdf"
  - Use when: "Delete that web page" -> source_identifier="https://example.com/old-article"

Error Handling:
  - Returns error if document not found or deletion fails.""",
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def rag_delete_document(source_identifier: str) -> str:
    try:
        engine = _get_engine()
        engine.delete_by_source(source_identifier)
        return f"Successfully deleted all chunks for '{source_identifier}'."
    except Exception as exc:
        logger.exception("rag_delete_document failed")
        return _format_error(exc, f"deletion of '{source_identifier}'")


@mcp.tool(
    name="rag_list_documents",
    title="List Ingested Documents",
    description="""List all documents currently indexed in the RAG knowledge base.

Returns document source paths, chunk counts, ingestion timestamps, and
metadata (doc type, topics, language, keywords) when metadata extraction
is enabled.

Args:
  - (none)

Returns:
  For JSON format:
  {
    "total": number,
    "documents": [
      {
        "source": string,
        "chunk_count": number,
        "ingested_at": string,
        "sections": [string],
        "doc_type": string,
        "topics": [string],
        "language": string,
        "keywords": [string]
      }
    ]
  }
  For Markdown format: Formatted table of documents.

Error Handling:
  - Returns error if Qdrant is unavailable.""",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def rag_list_documents(
    response_format: ResponseFormat = ResponseFormat.MARKDOWN,
) -> str:
    try:
        engine = _get_engine()
        docs = engine.list_documents()

        if not docs:
            if response_format == ResponseFormat.JSON:
                return json.dumps({"total": 0, "documents": []})
            return "No documents ingested yet."

        if response_format == ResponseFormat.JSON:
            return json.dumps({"total": len(docs), "documents": docs}, indent=2)

        lines = ["# Ingested Documents", ""]
        for doc in docs:
            lines.append(f"## {doc['source']}")
            lines.append(f"- **Chunks**: {doc['chunk_count']}")
            if doc.get("ingested_at"):
                lines.append(f"- **Ingested**: {doc['ingested_at']}")
            if doc.get("doc_type"):
                lines.append(f"- **Type**: {doc['doc_type']}")
            if doc.get("topics"):
                lines.append(f"- **Topics**: {', '.join(doc['topics'][:10])}")
            if doc.get("language"):
                lines.append(f"- **Language**: {doc['language']}")
            if doc.get("keywords"):
                lines.append(f"- **Keywords**: {', '.join(doc['keywords'][:10])}")
            if doc.get("sections"):
                lines.append(f"- **Sections**: {', '.join(doc['sections'][:10])}")
            lines.append("")
        return "\n".join(lines)
    except Exception as exc:
        logger.exception("rag_list_documents failed")
        return _format_error(exc, "listing documents")


@mcp.tool(
    name="rag_collection_stats",
    title="Collection Statistics",
    description="""Get statistics about the RAG vector collection.

Returns total point count, vector count, and configuration details.

Args:
  - (none)

Returns:
  JSON object with collection statistics.

Error Handling:
  - Returns error if Qdrant is unavailable.""",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def rag_collection_stats() -> str:
    try:
        engine = _get_engine()
        stats = engine.get_collection_stats()
        return json.dumps(stats, indent=2)
    except Exception as exc:
        logger.exception("rag_collection_stats failed")
        return _format_error(exc, "fetching collection stats")


@mcp.tool(
    name="rag_service_status",
    title="Service Status",
    description="""Check the status of all backend services.

Returns health status and latency for each service:
- Qdrant (vector database)
- Ollama (embedding server)
- Docling (document conversion)
- Redis (caching layer)

Args:
  - (none)

Returns:
  JSON object with service status information.

Error Handling:
  - Returns status for each service individually.""",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def rag_service_status() -> str:
    from urllib.parse import urlparse

    import httpx

    services = {
        "qdrant": config.QDRANT_URL,
        "ollama": config.OLLAMA_EMBED_URL,
        "docling": config.DOCLING_URL,
        "redis": config.REDIS_URL,
    }

    statuses = {}
    for name, url in services.items():
        try:
            parsed = urlparse(url)
            base = f"{parsed.scheme}://{parsed.netloc}"

            async with httpx.AsyncClient(timeout=5.0) as client:
                if name == "ollama":
                    health_url = f"{base}/api/tags"
                elif name == "qdrant":
                    health_url = f"{base}/"
                elif name == "redis":
                    statuses[name] = {"healthy": True, "url": url, "note": "Redis HTTP health check not available"}
                    continue
                else:
                    health_url = f"{base}/health"

                response = await client.get(health_url)
                latency_ms = response.elapsed.total_seconds() * 1000
                statuses[name] = {
                    "healthy": response.status_code == 200,
                    "url": url,
                    "latency_ms": round(latency_ms, 2),
                }
        except httpx.RequestError as e:
            statuses[name] = {
                "healthy": False,
                "url": url,
                "error": str(e),
            }
        except Exception as e:
            statuses[name] = {
                "healthy": False,
                "url": url,
                "error": str(e),
            }

    return json.dumps(statuses, indent=2)


# ── Pre-warm models on import (background thread) ─────────────────────────────
_prewarm_models()
