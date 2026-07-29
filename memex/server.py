"""MCP server — tool definitions for Personal RAG Engine.

Uses lazy initialisation so the server can start even if Qdrant / Ollama
are temporarily unavailable (they are contacted only on first tool call).

All tools are async for proper streamable HTTP transport support.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import threading
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations

from memex.schemas import (
    CollectionStatsOutput,
    DeleteDocumentInput,
    DocumentInfo,
    IngestBatchInput,
    IngestFileInput,
    IngestUrlInput,
    ListDocumentsInput,
    ListDocumentsOutput,
    QueryInput,
    QueryOutput,
    ResponseFormat,
    SearchResult,
)
from rag import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-16s  %(levelname)-5s  %(message)s",
)
logger = logging.getLogger("mcp-server")

CHARACTER_LIMIT = config.CHARACTER_LIMIT

mcp = FastMCP("memex-rag")

_engine = None


def _get_engine():
    """Return the RAGEngine singleton, creating it on first call.

    Also forces Qdrant connection and collection creation so the first
    tool call (ingest, query, stats, etc.) never fails with "collection
    not found".  If Qdrant is unreachable the call is retried on next
    engine access thanks to the reset logic in _get_qdrant().
    """
    global _engine
    if _engine is None:
        from rag.pipeline import RAGEngine

        _engine = RAGEngine()
        _engine._get_qdrant()  # ensures collection exists before any tool runs
    return _engine


def _prewarm_models():
    """Load Ollama embedding and chat models in background thread.

    Sends a trivial request to force Ollama to load models from disk
    into GPU memory. Without this the first user query incurs a cold-start
    penalty while model weights transfer.
    """
    import httpx

    def _load():
        try:
            engine = _get_engine()

            # Prewarm local reranker (CrossEncoder, GPU)
            if config.ENABLE_RERANKER:
                try:
                    engine._rerank("warmup", ["warmup"], top_k=1)
                    logger.info("Reranker loaded")
                except Exception as exc:
                    logger.warning("Reranker prewarm failed: %s", exc)

            # Prewarm Ollama embedding model (cold-start: ~25-30s)
            try:
                with httpx.Client(timeout=120) as client:
                    client.post(
                        config.OLLAMA_EMBED_URL,
                        json={"model": config.EMBED_MODEL, "input": "prewarm"},
                    )
                logger.info("Embedding model pre-warmed: %s", config.EMBED_MODEL)
            except Exception as exc:
                logger.warning("Embedding prewarm failed: %s", exc)

            # Prewarm Ollama chat model (cold-start: ~50-60s)
            try:
                with httpx.Client(timeout=120) as client:
                    chat_url = config.OLLAMA_EMBED_URL.replace("/api/embed", "/api/chat")
                    client.post(
                        chat_url,
                        json={
                            "model": config.CHAT_MODEL,
                            "messages": [{"role": "user", "content": "ok"}],
                            "stream": False,
                            "options": {"num_predict": 1, "temperature": 0},
                        },
                    )
                logger.info("Chat model pre-warmed: %s", config.CHAT_MODEL)
            except Exception as exc:
                logger.warning("Chat prewarm failed: %s", exc)

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


# ── Helper ─────────────────────────────────────────────────────────────────────


def _truncate(text: str) -> str:
    if len(text) <= CHARACTER_LIMIT:
        return text
    truncated = text[:CHARACTER_LIMIT]
    return (
        f"{truncated}\n\n---\nResponse truncated ({len(text)} chars > {CHARACTER_LIMIT} limit). "
        "Use pagination or add filters to retrieve smaller result sets."
    )


def _friendly_error(exc: Exception) -> str:
    """Translate common backend exceptions into user-friendly messages."""
    msg = str(exc).lower()

    if "cannot reach docling" in msg:
        return "Docling service is unreachable (port 5001). Run: docker compose up -d docling"
    if "cannot reach ollama" in msg:
        return "Ollama service is unreachable (port 11434). Run: docker compose up -d ollama"
    if "cannot reach" in msg:
        service = msg.split("cannot reach")[-1].strip().rstrip(".")
        return f"Service unreachable: {service}. Check docker compose ps"
    if "connection refused" in msg:
        return "Connection refused. Is the backend service running? Run: docker compose ps"
    if "collection" in msg and "doesn't exist" in msg:
        return "Qdrant collection not found. The server will auto-create it on next connection."
    if "qdrant" in msg:
        return "Qdrant is unreachable. Check: docker compose ps qdrant"
    if "ollama" in msg and "failed" in msg:
        return "Ollama request failed. Is Ollama running? Check: docker compose ps ollama"

    return f"Error: {exc}"


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
async def rag_ingest_file(input: IngestFileInput, ctx: Context) -> str:
    file_path_or_url = input.file_path_or_url
    try:
        from rag.docling_client import parse_file

        engine = _get_engine()

        def _progress(msg: str, pct: int) -> None:
            logger.info("ingest [%d%%] %s", pct, msg)

        # Phase 1+2: Pre-check — skip if file unchanged
        await ctx.report_progress(progress=2, total=100, message="Checking if already ingested...")
        can_skip, chunk_count = engine.check_unmodified_local(file_path_or_url)
        if can_skip:
            return (
                f"Already ingested '{file_path_or_url}' "
                f"({chunk_count} chunks). File unchanged — skipping."
            )

        await ctx.report_progress(progress=5, total=100, message="Reading file from disk...")
        result = parse_file(file_path_or_url)

        if not result.ok:
            return f"Error: Docling conversion returned status '{result.status}' with errors: {result.errors}"

        await ctx.report_progress(progress=10, total=100, message="Checking content hash...")
        content_hash = engine.compute_file_hash(result.markdown.encode())
        already, existing_chunks = engine.is_already_ingested(file_path_or_url, content_hash)
        if already:
            return (
                f"Already ingested '{file_path_or_url}' "
                f"({existing_chunks} chunks, hash: {content_hash[:12]}...). "
                f"File unchanged — skipping."
            )

        await ctx.report_progress(progress=15, total=100, message="Converting with Docling...")
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
        chunker_status = engine.get_chunker_status()
        chunker_name = chunker_status["active_chunker"]
        return (
            f"Successfully ingested '{file_path_or_url}'. "
            f"Created {count} chunks using {chunker_name}. "
            f"(Docling: {result.processing_time:.1f}s, "
            f"{len(result.markdown)} chars, hash: {content_hash[:12]}...)"
        )
    except Exception as exc:
        logger.exception("rag_ingest_file failed")
        return _friendly_error(exc)


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
async def rag_ingest_url(input: IngestUrlInput, ctx: Context) -> str:
    url = input.url
    try:
        from rag.docling_client import parse_url

        engine = _get_engine()

        def _progress(msg: str, pct: int) -> None:
            logger.info("ingest [%d%%] %s", pct, msg)

        await ctx.report_progress(progress=5, total=100, message="Fetching URL...")
        result = parse_url(url)

        if not result.ok:
            return f"Error: Docling conversion returned status '{result.status}' with errors: {result.errors}"

        await ctx.report_progress(progress=10, total=100, message="Checking if already ingested...")
        content_hash = engine.compute_file_hash(result.markdown.encode())
        already, chunk_count = engine.is_already_ingested(url, content_hash)
        if already:
            return (
                f"Already ingested '{url}' "
                f"({chunk_count} chunks, hash: {content_hash[:12]}...). "
                f"File unchanged — skipping."
            )

        await ctx.report_progress(progress=15, total=100, message="Converting with Docling...")
        count = engine.ingest_text(
            result.markdown,
            source_identifier=url,
            metadata={},
            content_hash=content_hash,
            progress_cb=_progress,
        )
        chunker_status = engine.get_chunker_status()
        chunker_name = chunker_status["active_chunker"]
        return (
            f"Successfully ingested '{url}'. "
            f"Created {count} chunks using {chunker_name}. "
            f"(Docling: {result.processing_time:.1f}s, "
            f"{len(result.markdown)} chars, hash: {content_hash[:12]}...)"
        )
    except Exception as exc:
        logger.exception("rag_ingest_url failed")
        return _friendly_error(exc)


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
async def rag_ingest_batch(input: IngestBatchInput, ctx: Context) -> dict[str, str]:
    from rag.ingestion import IngestionOrchestrator

    engine = _get_engine()
    orchestrator = IngestionOrchestrator(engine)

    total = len(input.items)
    await ctx.report_progress(progress=0, total=total, message="Starting batch ingestion...")
    summary = await orchestrator.ingest_batch(input.items)
    await ctx.report_progress(progress=total, total=total, message="Batch complete")
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
  - top_k (number): Max results to fetch from backend, 1-50 (default: 5).
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
  - offset (number): Pagination offset, skip first N results (default: 0).
  - limit (number): Max results per page, 1-50 (default: 10).

Returns:
  For JSON format: structured QueryOutput object.
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
async def rag_query(input: QueryInput) -> str | QueryOutput:
    try:
        engine = _get_engine()

        expansion_enabled = (
            input.use_query_expansion if input.use_query_expansion is not None else config.ENABLE_QUERY_EXPANSION
        )

        expanded = None
        if expansion_enabled:
            from rag.services.query_expansion import QueryExpander

            expander = QueryExpander(engine._get_ollama())
            expanded = expander.expand(input.query)

        results = engine.hybrid_search(
            query=input.query,
            top_k=input.top_k,
            rerank=input.use_reranking,
            source_filter=input.source_filter,
            metadata_filter=input.metadata_filter,
            expanded_query=expanded,
            use_contextual_search=input.use_contextual_search,
        )
        if not results:
            if input.response_format == ResponseFormat.JSON:
                return QueryOutput(total=0, count=0, results=[])
            return f"No results found for '{input.query}'."

        total = len(results)
        paged = results[input.offset : input.offset + input.limit]

        if not paged:
            if input.response_format == ResponseFormat.JSON:
                return QueryOutput(total=total, count=0, results=[])
            return f"No results found for page (offset={input.offset}, limit={input.limit})."

        if input.response_format == ResponseFormat.JSON:
            search_results = [
                SearchResult(
                    id=r["id"],
                    rrf_score=r.get("rrf_score"),
                    rerank_score=r.get("rerank_score"),
                    source=r["source"],
                    content=r["content"],
                    section_header=r.get("section_header", ""),
                    context_prefix=r.get("context_prefix", ""),
                    doc_type=r.get("doc_type", ""),
                    topics=r.get("topics", []),
                    language=r.get("language", ""),
                    keywords=r.get("keywords", []),
                )
                for r in paged
            ]
            return QueryOutput(total=total, count=len(paged), results=search_results)

        lines = [f"# Search Results for '{input.query}'", ""]
        for i, r in enumerate(paged, 1):
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

            meta_parts = []
            if r.get("doc_type") and r["doc_type"].strip():
                meta_parts.append(f"Type: {r['doc_type']}")
            if r.get("topics") and len(r["topics"]) > 0:
                meta_parts.append(f"Topics: {', '.join(r['topics'])}")
            if r.get("language") and r["language"].strip():
                meta_parts.append(f"Lang: {r['language']}")
            if r.get("keywords") and len(r["keywords"]) > 0:
                meta_parts.append(f"Keywords: {', '.join(r['keywords'][:5])}")
            if meta_parts:
                lines.append(f"**Metadata**: {' | '.join(meta_parts)}")

            lines.append("")
            lines.append(r["content"])
            lines.append("")
        return _truncate("\n".join(lines))
    except Exception as exc:
        logger.exception("rag_query failed")
        return _friendly_error(exc)


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
async def rag_delete_document(input: DeleteDocumentInput) -> str:
    source_identifier = input.source_identifier
    try:
        engine = _get_engine()
        engine.delete_by_source(source_identifier)
        return f"Successfully deleted all chunks for '{source_identifier}'."
    except Exception as exc:
        logger.exception("rag_delete_document failed")
        return _friendly_error(exc)


@mcp.tool(
    name="rag_list_documents",
    title="List Ingested Documents",
    description="""List all documents currently indexed in the RAG knowledge base.

Returns document source paths, chunk counts, ingestion timestamps, and
metadata (doc type, topics, language, keywords) when metadata extraction
is enabled.

Args:
  - response_format ('markdown' | 'json'): Output format (default: 'markdown').
  - offset (number): Pagination offset, skip first N documents (default: 0).
  - limit (number): Max documents per page, 1-100 (default: 20).

Returns:
  For JSON format: structured ListDocumentsOutput object.
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
async def rag_list_documents(input: ListDocumentsInput) -> str | ListDocumentsOutput:
    try:
        engine = _get_engine()
        docs = engine.list_documents()

        if not docs:
            if input.response_format == ResponseFormat.JSON:
                return ListDocumentsOutput(total=0, documents=[])
            return "No documents ingested yet."

        total = len(docs)
        paged = docs[input.offset : input.offset + input.limit]

        if not paged:
            if input.response_format == ResponseFormat.JSON:
                return ListDocumentsOutput(total=total, documents=[])
            return f"No documents found for page (offset={input.offset}, limit={input.limit})."

        if input.response_format == ResponseFormat.JSON:
            doc_infos = [
                DocumentInfo(
                    source=d["source"],
                    chunk_count=d["chunk_count"],
                    ingested_at=d.get("ingested_at", ""),
                    sections=d.get("sections", []),
                    doc_type=d.get("doc_type", ""),
                    topics=d.get("topics", []),
                    language=d.get("language", ""),
                    keywords=d.get("keywords", []),
                )
                for d in paged
            ]
            return ListDocumentsOutput(total=total, documents=doc_infos)

        lines = ["# Ingested Documents", ""]
        for doc in paged:
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
        return _friendly_error(exc)


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
        output = CollectionStatsOutput(
            collection_name=stats.get("collection_name", ""),
            total_points=stats.get("total_points", 0),
            total_vectors=stats.get("total_vectors", 0),
            status=stats.get("status", ""),
            optimizer_status=stats.get("optimizer_status", ""),
            config=stats.get("config", {}),
        )
        return output.model_dump_json(indent=2)
    except Exception as exc:
        logger.exception("rag_collection_stats failed")
        return _friendly_error(exc)


@mcp.tool(
    name="rag_service_status",
    title="Service Status",
    description="""Check the status of all backend services.

Returns health status and latency for each service:
- Qdrant (vector database)
- Ollama (embedding server)
- Docling (document conversion)
- ML Services (sparse BM25 + reranker)
- Redis (caching layer)

Also includes chunker configuration (strategy, size, availability).

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
        "ml-services": config.ML_SERVICES_URL,
        "redis": config.REDIS_URL,
    }

    statuses: dict[str, dict[str, Any]] = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in services.items():
            try:
                parsed = urlparse(url)
                base = f"{parsed.scheme}://{parsed.netloc}"

                if name == "redis":
                    try:
                        import redis

                        r = redis.Redis.from_url(url)
                        r.ping()
                        statuses[name] = {
                            "healthy": True,
                            "url": url,
                            "latency_ms": 0.0,
                        }
                    except Exception as e:
                        statuses[name] = {
                            "healthy": False,
                            "url": url,
                            "error": str(e),
                        }
                    continue

                if name == "ollama":
                    health_url = f"{base}/api/tags"
                elif name == "qdrant":
                    health_url = f"{base}/"
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

    try:
        engine = _get_engine()
        chunker_info = engine.get_chunker_status()
    except Exception:
        chunker_info = {"error": "unavailable"}
    statuses["chunker"] = chunker_info

    try:
        return json.dumps(statuses, indent=2)
    except Exception as exc:
        logger.exception("rag_service_status failed")
        return _friendly_error(exc)


# ── Pre-warm models on import (background thread) ─────────────────────────────
_prewarm_models()
_features = []
if config.ENABLE_QUERY_EXPANSION:
    _features.append("query-expansion")
if config.ENABLE_HYDE:
    _features.append("hyde")
if config.ENABLE_MULTI_QUERY:
    _features.append("multi-query")
if config.ENABLE_QUERY_REWRITE:
    _features.append("query-rewrite")
if config.ENABLE_CONTEXTUAL_RETRIEVAL:
    _features.append("contextual-retrieval")
if config.ENABLE_METADATA_EXTRACTION:
    _features.append("metadata-extraction")
if config.ENABLE_CACHE:
    _features.append("cache")
_features.append(f"chunk-strategy={config.CHUNK_STRATEGY}")
logger.info("startup complete — %d features enabled: %s", len(_features), ", ".join(_features))
