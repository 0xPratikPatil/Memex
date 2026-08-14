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

from memex.engine.core import config
from memex.engine.core.logging_setup import setup_logging
from memex.mcp.schemas import (
    AnswerOutput,
    CitationInfo,
    CollectionStatsOutput,
    DeleteDocumentInput,
    DocumentInfo,
    EvalInput,
    EvalOutput,
    EvalQueryResult,
    EvalSweepInput,
    EvalSweepOutput,
    ExtractedFiltersOutput,
    ExtractFiltersInput,
    FieldInfoOutput,
    FilterContextInput,
    FilterContextOutput,
    IngestBatchInput,
    IngestFileInput,
    IngestUrlInput,
    ListDocumentsInput,
    ListDocumentsOutput,
    QueryInput,
    QueryOutput,
    ResponseFormat,
    SearchResult,
    SyncInput,
    SyncStatsOutput,
)

setup_logging(verbose=config.EVAL_LOG_TIMING)
logger = logging.getLogger("mcp-server")

CHARACTER_LIMIT = config.CHARACTER_LIMIT

mcp = FastMCP("memex-rag")

_engine = None
_engine_lock = threading.Lock()


def _get_engine():
    """Return the RAGEngine singleton, creating it on first call (thread-safe).

    Also forces Qdrant connection and collection creation so the first
    tool call (ingest, query, stats, etc.) never fails with "collection
    not found".  If Qdrant is unreachable the call is retried on next
    engine access thanks to the reset logic in _get_qdrant().
    """
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from memex.engine.core.pipeline import RAGEngine

                _engine = RAGEngine()
                _engine._get_qdrant()  # ensures collection exists before any tool runs
    return _engine


def _prewarm_models() -> None:
    """Load Ollama embedding and chat models in background thread.

    Sends a trivial request to force Ollama to load models from disk
    into GPU memory. Without this the first user query incurs a cold-start
    penalty while model weights transfer.
    """
    import httpx

    def _load() -> None:
        try:
            engine = _get_engine()

            # Prewarm local reranker (CrossEncoder, GPU)
            if config.ENABLE_RERANKING:
                try:
                    engine._rerank("warmup", ["warmup"], top_k=1)
                    logger.info("Reranker loaded")
                except Exception as exc:
                    logger.warning("Reranker prewarm failed: %s", exc)

            # Prewarm Ollama embedding + chat models in parallel
            import concurrent.futures

            def _prewarm_embedding() -> None:
                client = httpx.Client(timeout=120)
                try:
                    client.post(
                        config.OLLAMA_EMBED_URL,
                        json={"model": config.EMBED_MODEL, "input": ["prewarm"]},
                    )
                    logger.info("Embedding model pre-warmed: %s", config.EMBED_MODEL)
                finally:
                    client.close()

            def _prewarm_chat() -> None:
                client = httpx.Client(timeout=120)
                try:
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
                finally:
                    client.close()

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(_prewarm_embedding), pool.submit(_prewarm_chat)]
                concurrent.futures.wait(futures)
                for f in futures:
                    if f.exception():
                        logger.warning("Prewarm failed: %s", f.exception())

            logger.info("Model pre-warming complete")
        except Exception as exc:
            logger.warning("Model pre-warming failed: %s", exc)

    threading.Thread(target=_load, daemon=True, name="model-prewarm").start()


def _shutdown() -> None:
    """Best-effort cleanup on process exit."""
    global _engine
    with contextlib.suppress(Exception):
        from memex.engine.ingestion import loader as docling_client
        from memex.engine.ingestion.marker_client import close as close_marker

        if _engine is not None:
            _engine.close()
            _engine = None
        docling_client.close()
        close_marker()


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
    """Translate exceptions into user-friendly messages.

    Dispatches on typed ``MemexError`` subclasses first (returning their
    actionable ``hint``), then falls back to a best-effort string map, then
    to the raw error.
    """
    from memex.engine.core.errors import (
        ConfigError,
        ConversionTimeoutError,
        CorruptedDocumentError,
        ServiceUnavailableError,
    )

    if isinstance(exc, ConversionTimeoutError):
        return (
            "Docling timed out on this document (it may be too large or the "
            "server is overloaded). "
            + (exc.hint or "Reduce converter.docling_max_concurrent or retry.")
        )
    if isinstance(exc, ServiceUnavailableError):
        return (
            f"{exc.service} is unreachable. "
            + (exc.hint or "Check: docker compose ps")
        )
    if isinstance(exc, CorruptedDocumentError):
        return f"The document could not be parsed into usable content: {exc}"
    if isinstance(exc, ConfigError):
        return f"Configuration error: {exc}"

    msg = str(exc).lower()

    if "cannot reach docling" in msg:
        if "timeout" in msg:
            return "Docling service timed out (port 5001). It may be overloaded or handling a large document. Retry."
        if "disconnected" in msg or "read" in msg:
            return "Docling server disconnected (port 5001). It may be overloaded or restarting. Retry in a moment."
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
        from memex.engine.ingestion.loader import parse_file

        engine = _get_engine()

        def _progress(msg: str, pct: int) -> None:
            logger.info("ingest [%d%%] %s", pct, msg)

        # Phase 1+2: Pre-check — skip if file unchanged
        await ctx.report_progress(progress=2, total=100, message="Checking if already ingested...")
        can_skip, chunk_count = engine.check_unmodified_local(file_path_or_url)
        if can_skip:
            return f"Already ingested '{file_path_or_url}' ({chunk_count} chunks). File unchanged — skipping."

        await ctx.report_progress(progress=5, total=100, message="Reading file from disk...")

        # Validate local file path (reject relative paths and non-existent files)
        import os

        if not file_path_or_url.startswith(("http://", "https://")):
            if not file_path_or_url.startswith("/"):
                return f"Error: Relative paths not allowed. Use absolute path: {os.path.abspath(file_path_or_url)}"
            abs_path = os.path.realpath(file_path_or_url)
            if not os.path.isfile(abs_path):
                return f"Error: File not found: {file_path_or_url}"

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
        from memex.engine.ingestion.loader import parse_url

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
    from memex.engine.ingestion.ingestion import IngestionOrchestrator

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
document type, topics, language, keywords, or entities stored in the
Qdrant payload.

Supports search modes:
- 'hybrid': Dense + BM25 with RRF fusion (default)
- 'similarity': Dense only
- 'mmr': Dense similarity + Maximal Marginal Relevance for diversity

When answer generation is enabled, returns a structured answer with citations.

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
    "entities.people", "entities.dates"). Values are strings or lists of strings.
    Example: {"doc_type": "report", "topics": ["finance", "revenue"]}.
  - offset (number): Pagination offset, skip first N results (default: 0).
  - limit (number): Max results per page, 1-50 (default: 10).
  - search_mode ('similarity' | 'hybrid' | 'mmr', optional): Override search mode.
  - generate_answer (boolean, optional): Override answer generation setting.

Returns:
  For JSON format: structured AnswerOutput (when answer enabled) or QueryOutput.
  For Markdown format: Formatted answer with citations or search results.

Examples:
  - Use when: "Find revenue data" -> query="quarterly revenue figures"
  - Use when: "What did the contract say about termination?" -> query="contract termination clauses"
  - Use when: "Search only in report.pdf" -> query="revenue", source_filter="/docs/report.pdf"
  - Use when: "Find reports about finance" -> query="revenue", metadata_filter={"doc_type": "report"}
  - Use when: "Diverse results for broad topic" -> query="AI trends", search_mode="mmr"

Error Handling:
  - Returns error message if search fails or Qdrant is unavailable.
  - If answer generation fails, falls back to raw search results.""",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def rag_query(input: QueryInput) -> str | QueryOutput | AnswerOutput:
    try:
        engine = _get_engine()

        # Determine search mode
        search_mode = input.search_mode or config.SEARCH_MODE

        expansion_enabled = (
            input.use_query_expansion if input.use_query_expansion is not None else config.ENABLE_QUERY_EXPANSION
        )

        expanded = None
        if expansion_enabled and search_mode != "mmr":
            from memex.engine.retrieval.expansion import QueryExpander

            expander = QueryExpander(engine._llm, engine._embedder)
            expanded = expander.expand(input.query)

        # Execute search based on mode
        if search_mode == "mmr":
            results = engine.mmr_search(
                query=input.query,
                top_k=input.top_k,
                fetch_k=config.MMR_FETCH_K,
                lambda_mult=config.MMR_LAMBDA_MULT,
                rerank=input.use_reranking,
                source_filter=input.source_filter,
                metadata_filter=input.metadata_filter,
                use_contextual_search=input.use_contextual_search,
            )
        else:
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
                if _answer_enabled(input.generate_answer):
                    return AnswerOutput(
                        text="No results found.",
                        refused=True,
                        confidence=0.0,
                        citations=[],
                        sources=[],
                        search_mode=search_mode,
                        results=[],
                    )
                return QueryOutput(total=0, count=0, results=[])
            return f"No results found for '{input.query}'."

        total = len(results)
        paged = results[input.offset : input.offset + input.limit]

        if not paged:
            if input.response_format == ResponseFormat.JSON:
                if _answer_enabled(input.generate_answer):
                    return AnswerOutput(
                        text="No results found for this page.",
                        refused=True,
                        confidence=0.0,
                        citations=[],
                        sources=[],
                        search_mode=search_mode,
                        results=[],
                    )
                return QueryOutput(total=total, count=0, results=[])
            return f"No results found for page (offset={input.offset}, limit={input.limit})."

        # Check if answer generation is requested
        if _answer_enabled(input.generate_answer):
            return await _generate_and_return_answer(
                engine=engine,
                query=input.query,
                results=results,
                search_mode=search_mode,
                response_format=input.response_format,
            )

        # Standard search result format
        if input.response_format == ResponseFormat.JSON:
            search_results = _build_search_results(paged)
            return QueryOutput(total=total, count=len(paged), results=search_results)

        return _format_markdown_results(input.query, paged, total)
    except Exception as exc:
        logger.exception("rag_query failed")
        return _friendly_error(exc)


def _answer_enabled(override: bool | None) -> bool:
    """Check if answer generation is enabled (via override or config)."""
    if override is not None:
        return override
    return config.ENABLE_ANSWER


async def _generate_and_return_answer(
    engine: Any,
    query: str,
    results: list[dict[str, Any]],
    search_mode: str,
    response_format: ResponseFormat,
) -> str | AnswerOutput:
    """Generate a cited answer and return structured output."""
    from memex.engine.generation.answers import generate_answer

    async def _chat_wrapper(prompt: str) -> str:
        return await engine._llm.chat(prompt)

    answer = await generate_answer(
        query=query,
        chunks=results,
        ollama_chat_fn=_chat_wrapper,
        max_context_chars=config.ANSWER_MAX_CONTEXT_CHARS,
    )

    if response_format == ResponseFormat.JSON:
        citation_infos = [
            CitationInfo(
                index=c.index,
                source=c.source,
                snippet=c.chunk_text[:200] + ("..." if len(c.chunk_text) > 200 else ""),
                score=c.rerank_score,
            )
            for c in answer.citations
        ]
        search_results = _build_search_results(results[: len(results)])
        return AnswerOutput(
            text=answer.text,
            refused=answer.refused,
            confidence=answer.confidence,
            citations=citation_infos,
            sources=answer.sources,
            search_mode=search_mode,
            results=search_results,
        )

    # Markdown format
    lines = [answer.text, ""]
    if answer.citations:
        lines.append("Sources:")
        seen: list[str] = []
        for c in answer.citations:
            if c.source not in seen:
                seen.append(c.source)
                lines.append(f"  [{c.index}] {c.source}")
    return _truncate("\n".join(lines))


def _build_search_results(paged: list[dict]) -> list[SearchResult]:
    """Convert raw result dicts to SearchResult models."""
    return [
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


def _format_markdown_results(query: str, paged: list[dict], total: int) -> str:
    """Format search results as markdown."""
    lines = [f"# Search Results for '{query}'", ""]
    for i, r in enumerate(paged, 1):
        score_parts = []
        if "rrf_score" in r:
            score_parts.append(f"RRF: {r['rrf_score']:.4f}")
        if "rerank_score" in r:
            score_parts.append(f"rerank: {r['rerank_score']:.4f}")
        if "dense_score" in r and "rrf_score" not in r:
            score_parts.append(f"dense: {r['dense_score']:.4f}")
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

    converter_name = "marker" if config.CONVERTER_ENGINE == "marker" else "docling"
    services = {
        "qdrant": config.QDRANT_URL,
        "ollama": config.OLLAMA_EMBED_URL,
        converter_name: config.DOCLING_URL if config.CONVERTER_ENGINE == "docling" else f"{config.MARKER_URL}/health",
    }

    statuses: dict[str, dict[str, Any]] = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in services.items():
            try:
                parsed = urlparse(url)
                base = f"{parsed.scheme}://{parsed.netloc}"

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
                    "url": health_url,
                    "latency_ms": round(latency_ms, 2),
                }
            except httpx.RequestError as e:
                statuses[name] = {
                    "healthy": False,
                    "url": health_url,
                    "error": str(e),
                }
            except Exception as e:
                statuses[name] = {
                    "healthy": False,
                    "url": health_url,
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


@mcp.tool(
    name="rag_sync",
    title="Sync Document Sources",
    description="""Sync the vector collection against configured document sources.

Reconciles the collection with sources defined in config.yaml:
- New files are ingested
- Changed files (different content hash) replace old chunks
- Deleted files (not in any source) have their chunks removed

Safety: if any source fails to list, all deletions are suppressed for that run.

Args:
  - source_name (string, optional): Sync a specific source. Null = sync all.
  - dry_run (boolean): Report what would change without writing (default: false).

Returns:
  JSON object with sync statistics (added, changed, deleted, unchanged, errors).

Error Handling:
  - Returns partial results if some sources fail.""",
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def rag_sync(input: SyncInput, ctx: Context) -> str:
    try:
        from memex.engine.core.progress import FileProgress
        from memex.engine.sources.sync import sync

        def _report(progress: FileProgress) -> None:
            msg = f"[{progress.stage}] {progress.path}"

            async def _send() -> None:
                await ctx.report_progress(
                    progress=progress.current,
                    total=progress.total,
                    message=msg,
                )

            import asyncio

            try:
                loop = asyncio.get_running_loop()
                _task = loop.create_task(_send())  # noqa: RUF006
            except RuntimeError:
                pass

        result = await sync(
            config_module=config,
            source_name=input.source_name,
            dry_run=input.dry_run,
            progress_cb=_report,
        )
        output = SyncStatsOutput(
            added=result.added,
            changed=result.changed,
            deleted=result.deleted,
            unchanged=result.unchanged,
            errors=result.errors,
            dry_run=input.dry_run,
        )
        return output.model_dump_json(indent=2)
    except Exception as exc:
        logger.exception("rag_sync failed")
        return _friendly_error(exc)


@mcp.tool(
    name="rag_get_filter_context",
    title="Get Filter Context",
    description="""Show available metadata fields, their stored values, and suggest filters.

Discovers all metadata fields in the collection and their unique values.
Optionally suggests filters for a given query using the LLM.

Args:
  - query (string, optional): Query to get filter suggestions for.

Returns:
  JSON object with available fields, their types/values, and suggested filters.

Error Handling:
  - Returns empty fields list if collection is empty.""",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def rag_get_filter_context(input: FilterContextInput) -> str:
    try:
        from memex.engine.retrieval.filter import get_filter_context

        engine = _get_engine()
        ctx = await get_filter_context(
            config=config,
            query=input.query,
            qdrant_client=engine._get_qdrant(),
            collection=config.COLLECTION_NAME,
        )
        output = FilterContextOutput(
            fields=[
                FieldInfoOutput(
                    name=f.name,
                    type=f.type,
                    values=f.values,
                    count=f.count,
                )
                for f in ctx.fields
            ],
            suggested_filters=ctx.suggested_filters,
            sample_query=ctx.sample_query,
        )
        return output.model_dump_json(indent=2)
    except Exception as exc:
        logger.exception("rag_get_filter_context failed")
        return _friendly_error(exc)


@mcp.tool(
    name="rag_extract_filters",
    title="Extract Metadata Filters",
    description="""Extract metadata filters from a natural language query.

Parses a query into structured metadata filters without executing search.
Useful for agents that want to inspect/modify filters before searching.

Args:
  - query (string): Natural language query to extract filters from.

Returns:
  JSON object with extracted filters, explanation, and confidence score.

Error Handling:
  - Returns empty filters if extraction fails.""",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def rag_extract_filters(input: ExtractFiltersInput) -> str:
    try:
        from memex.engine.retrieval.filter import extract_filters, get_filter_context

        engine = _get_engine()
        ctx = await get_filter_context(
            config=config,
            qdrant_client=engine._get_qdrant(),
            collection=config.COLLECTION_NAME,
        )
        result = await extract_filters(
            query=input.query,
            available_fields=ctx.fields,
            llm_call=engine._llm.chat,
        )
        output = ExtractedFiltersOutput(
            filters=result.filters,
            explanation=result.explanation,
            confidence=result.confidence,
        )
        return output.model_dump_json(indent=2)
    except Exception as exc:
        logger.exception("rag_extract_filters failed")
        return _friendly_error(exc)


@mcp.tool(
    name="rag_eval",
    title="Evaluate Retrieval Quality",
    description="""Run golden-set evaluation against the RAG system.

Loads a golden set of queries with expected sources, runs retrieval,
and computes metrics (recall, precision, hit_rate, MRR, keyword_coverage).

Args:
  - golden_set_path (string): Path to golden set YAML/JSON file.
  - top_k (number): Results per query (default: 5).
  - compare_rerank (boolean): Compare with/without reranking (default: false).
  - source_match_mode (string): Source matching mode (default: "basename").

Returns:
  JSON object with aggregate and per-query evaluation metrics.
""",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def rag_eval(input: EvalInput) -> str:
    try:
        from memex.engine.evaluation import GoldenSet, keyword_coverage, match_source

        # Load golden set
        golden_path = input.golden_set_path
        if golden_path.endswith(".json"):
            golden_set = GoldenSet.from_json(golden_path)
        else:
            golden_set = GoldenSet.from_yaml(golden_path)

        if not golden_set.queries:
            return "Error: Golden set is empty — no queries to evaluate."

        engine = _get_engine()
        query_results: list[EvalQueryResult] = []
        mode = input.source_match_mode

        for gq in golden_set.queries:
            try:
                results = engine.hybrid_search(
                    query=gq.query,
                    top_k=input.top_k,
                    rerank=True,
                    metadata_filter=gq.filters,
                )
            except Exception as search_exc:
                logger.warning("Eval query failed: %s (%s)", gq.query[:60], search_exc)
                results = []

            # Extract sources and content from results
            retrieved_sources = [r["source"] for r in results]
            retrieved_content = " ".join(r.get("content", "") for r in results)

            # Compute metrics
            k = input.top_k
            recall = 0.0
            precision = 0.0
            hit_rate = 0.0
            mrr = 0.0

            if gq.expected_sources:
                expected = gq.expected_sources
                expected_set = set(expected)

                # Recall: fraction of expected found in top k
                window = set(retrieved_sources[:k])
                recall = len(expected_set & window) / len(expected_set) if expected_set else 0.0

                # Precision: fraction of top k that are correct
                hits = sum(1 for s in retrieved_sources[:k] if s in expected_set)
                precision = hits / k if k > 0 else 0.0

                # Hit rate: 1.0 if any expected found
                hit_rate = 1.0 if window & expected_set else 0.0

                # MRR: 1/rank of first correct result
                for position, s in enumerate(retrieved_sources[:k], start=1):
                    if s in expected_set:
                        mrr = 1.0 / position
                        break

            # Keyword coverage
            kw_coverage = keyword_coverage(retrieved_content, gq.expected_keywords)

            # Build matched expected sources for display
            matched_expected = []
            for es in gq.expected_sources:
                for rs in retrieved_sources:
                    if match_source(es, rs, mode=mode):
                        if es not in matched_expected:
                            matched_expected.append(es)
                        break

            query_results.append(
                EvalQueryResult(
                    query=gq.query,
                    recall=recall,
                    precision=precision,
                    hit_rate=hit_rate,
                    mrr=mrr,
                    keyword_coverage=kw_coverage,
                    expected_sources=gq.expected_sources,
                    retrieved_sources=retrieved_sources[:k],
                )
            )

        # Aggregate
        n = len(query_results)
        avg = lambda metric, _qr=query_results, _n=n: sum(getattr(q, metric) for q in _qr) / _n if _n else 0.0  # noqa: E731

        output = EvalOutput(
            total_queries=n,
            avg_recall=avg("recall"),
            avg_precision=avg("precision"),
            avg_hit_rate=avg("hit_rate"),
            avg_mrr=avg("mrr"),
            avg_keyword_coverage=avg("keyword_coverage"),
            queries=query_results,
        )
        return output.model_dump_json(indent=2)
    except FileNotFoundError as exc:
        return f"Error: Golden set not found — {exc}"
    except Exception as exc:
        logger.exception("rag_eval failed")
        return _friendly_error(exc)


@mcp.tool(
    name="rag_eval_sweep",
    title="Sweep Evaluation Configs",
    description="""Compare multiple retrieval configurations side by side.

Runs evaluation with different configs and shows delta comparison.

Args:
  - golden_set_path (string): Path to golden set file.
  - variants (list): List of variant configs to compare.
  - top_k (number): Results per query (default: 5).

Returns:
  Formatted comparison table with metrics and deltas.
""",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def rag_eval_sweep(input: EvalSweepInput) -> str:
    try:
        from memex.engine.evaluation import GoldenSet, keyword_coverage

        # Load golden set
        golden_path = input.golden_set_path
        if golden_path.endswith(".json"):
            golden_set = GoldenSet.from_json(golden_path)
        else:
            golden_set = GoldenSet.from_yaml(golden_path)

        if not golden_set.queries:
            return "Error: Golden set is empty — no queries to evaluate."

        engine = _get_engine()
        all_variant_results: list[EvalOutput] = []

        for variant in input.variants:
            variant_name = variant.get("name", "unnamed")
            rerank = variant.get("rerank", True)
            top_k = variant.get("top_k", input.top_k)

            query_results: list[EvalQueryResult] = []
            for gq in golden_set.queries:
                try:
                    results = engine.hybrid_search(
                        query=gq.query,
                        top_k=top_k,
                        rerank=rerank,
                        metadata_filter=gq.filters,
                    )
                except Exception as search_exc:
                    logger.warning("Eval query failed: %s (%s)", gq.query[:60], search_exc)
                    results = []

                retrieved_sources = [r["source"] for r in results]
                retrieved_content = " ".join(r.get("content", "") for r in results)

                k = top_k
                recall = precision = hit_rate = mrr = 0.0

                if gq.expected_sources:
                    expected_set = set(gq.expected_sources)
                    window = set(retrieved_sources[:k])
                    recall = len(expected_set & window) / len(expected_set) if expected_set else 0.0
                    hits = sum(1 for s in retrieved_sources[:k] if s in expected_set)
                    precision = hits / k if k > 0 else 0.0
                    hit_rate = 1.0 if window & expected_set else 0.0
                    for position, s in enumerate(retrieved_sources[:k], start=1):
                        if s in expected_set:
                            mrr = 1.0 / position
                            break

                kw_coverage = keyword_coverage(retrieved_content, gq.expected_keywords)

                query_results.append(
                    EvalQueryResult(
                        query=gq.query,
                        recall=recall,
                        precision=precision,
                        hit_rate=hit_rate,
                        mrr=mrr,
                        keyword_coverage=kw_coverage,
                        expected_sources=gq.expected_sources,
                        retrieved_sources=retrieved_sources[:k],
                    )
                )

            n = len(query_results)
            avg = lambda metric, _qr=query_results, _n=n: sum(getattr(q, metric) for q in _qr) / _n if _n else 0.0  # noqa: E731

            all_variant_results.append(
                EvalOutput(
                    total_queries=n,
                    avg_recall=avg("recall"),
                    avg_precision=avg("precision"),
                    avg_hit_rate=avg("hit_rate"),
                    avg_mrr=avg("mrr"),
                    avg_keyword_coverage=avg("keyword_coverage"),
                    queries=query_results,
                )
            )

        # Build delta table
        delta_lines: list[str] = []
        metric_keys = ["avg_recall", "avg_precision", "avg_hit_rate", "avg_mrr", "avg_keyword_coverage"]
        col_headers = ["recall", "precision", "hit_rate", "mrr", "kw_cov"]

        name_width = max(len(variant.get("name", "unnamed")) for variant in input.variants)
        name_width = max(name_width, len("variant"))
        cell = 11

        header = f"{'variant':<{name_width}}  " + "  ".join(f"{h:>{cell}}" for h in col_headers)
        delta_lines.append(header)
        delta_lines.append("-" * len(header))

        if all_variant_results:
            baseline = all_variant_results[0]
            baseline_metrics = {k: getattr(baseline, k, 0.0) for k in metric_keys}

            for i, (variant, result) in enumerate(zip(input.variants, all_variant_results, strict=True)):
                variant_name = variant.get("name", "unnamed")
                cells = []
                for mk in metric_keys:
                    value = getattr(result, mk, 0.0)
                    if i == 0:
                        cells.append(f"{value:.3f}".rjust(cell))
                    else:
                        delta = value - baseline_metrics.get(mk, 0.0)
                        cells.append(f"{value:.3f}{delta:+.2f}".rjust(cell))
                delta_lines.append(f"{variant_name:<{name_width}}  " + "  ".join(cells))

            if len(all_variant_results) > 1:
                best_variant, best_result = max(
                    zip(input.variants, all_variant_results, strict=True),
                    key=lambda x: x[1].avg_recall,
                )
                delta_lines.append("")
                delta_lines.append(f"Best recall: {best_variant.get('name', 'unnamed')} ({best_result.avg_recall:.3f})")

        sweep_output = EvalSweepOutput(
            variants=all_variant_results,
            delta_table="\n".join(delta_lines),
        )
        return sweep_output.model_dump_json(indent=2)
    except FileNotFoundError as exc:
        return f"Error: Golden set not found — {exc}"
    except Exception as exc:
        logger.exception("rag_eval_sweep failed")
        return _friendly_error(exc)


# ── rag_processing_status ──────────────────────────────────────────────────────


@mcp.tool(
    name="rag_processing_status",
    title="Processing Status",
    description="""Show the current processing status of files in the RAG system.

Displays per-file status including pending, processing, done, skipped, failed,
and retry states, with the current pipeline stage and error hint.

Args:
  - status (string, optional): Filter by status (pending/processing/done/skipped/failed/retry).
  - limit (number, optional): Max per-file records to return (default: 100).

Returns:
  - JSON object with aggregate summary and per-file detail.

Use when: "What's happening with my files?", "Show processing status",
"Which files are being processed?", "What failed?".""",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def rag_processing_status(
    status: str | None = None,
    limit: int = 100,
) -> str:
    try:
        from memex.engine.ingestion.status import FileStatusStore

        engine = _get_engine()
        qdrant = engine._get_qdrant()
        store = FileStatusStore(qdrant)
        summary = store.get_summary()
        records = store.list_records(status_filter=status, limit=limit)

        return json.dumps(
            {
                "summary": summary,
                "total": sum(summary.values()),
                "pending": summary.get("pending", 0),
                "processing": summary.get("processing", 0),
                "done": summary.get("done", 0),
                "skipped": summary.get("skipped", 0),
                "failed": summary.get("failed", 0),
                "retry": summary.get("retry", 0),
                "files": records,
            },
            indent=2,
        )
    except Exception as exc:
        logger.exception("rag_processing_status failed")
        return _friendly_error(exc)


# ── rag_retry_failed ───────────────────────────────────────────────────────────


@mcp.tool(
    name="rag_retry_failed",
    title="Retry Failed Files",
    description="""Retry files that previously failed ingestion.

Resets FAILED files to processing immediately (bypassing backoff).

Args:
  - source_filter (string, optional): Only retry files whose source contains this substring.

Returns:
  - JSON object with count of files reset.

Use when: "Retry the failed files", "Re-ingest what failed".""",
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
async def rag_retry_failed(source_filter: str | None = None) -> str:
    try:
        from memex.engine.ingestion.status import FileStatusStore
        from memex.engine.sources.retry_queue import RetryQueue

        engine = _get_engine()
        store = FileStatusStore(engine._get_qdrant())
        queue = RetryQueue(status_store=store)
        count = queue.reset_failed(status_filter=source_filter)
        return json.dumps({"reset": count}, indent=2)
    except Exception as exc:
        logger.exception("rag_retry_failed failed")
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
