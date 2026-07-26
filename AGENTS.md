# AGENTS.md — Memex RAG Project Instructions

OpenCode should proactively use the following features when working with this codebase.

## Available RAG Features (all enabled by default)

### Query Expansion
- **HyDE** (ENABLE_HYDE=true): LLM generates a hypothetical answer document, embeds it, and searches alongside the original query. Improves recall for conceptual queries.
- **Query Rewrite** (ENABLE_QUERY_REWRITE=true): LLM expands ambiguous or conversational queries into well-formed search queries.
- **Multi-Query** (ENABLE_MULTI_QUERY=true, MULTI_QUERY_COUNT=3): Generates 3 paraphrases of the query, searches each independently, fuses via RRF.
- All controlled by ENABLE_QUERY_EXPANSION=true (master toggle).

### Search
- **Hybrid Search**: Dense (bge-m3, 1024d) + Sparse (BM25 via Qdrant/bm25) + RRF fusion (k=60) + cross-encoder rerank (BAAI/bge-reranker-base).
- **Contextual Retrieval** (ENABLE_CONTEXTUAL_RETRIEVAL=true, CONTEXT_STRATEGY=summary): LLM-generated context prefixes for each chunk, improving embedding quality.
- **Search Cache**: Redis caches full result sets for repeated queries (CACHE_TTL_SEARCH=3600).

### Ingestion
- **Docling HybridChunker** (CHUNK_STRATEGY=hybrid): Tokenizer-aware, structure-preserving chunking on DoclingDocument. Repeats table headers across boundaries.
- **Multi-format Embedding**: Table chunks → HTML, code chunks → fenced markdown, images → [Image: caption].
- **Docling Enrichment**: Picture classification (enabled), code/formula/chart extraction (opt-in, needs serve-side models).
- **Metadata Extraction** (ENABLE_METADATA_EXTRACTION=true): Entities, topics, document classification, language detection, dates, keywords.
- **Embedding Cache**: Redis caches dense vectors (CACHE_TTL_EMBEDDING=86400).

## MCP Tools (8 available)

| Tool | Use when |
|------|----------|
| `rag_ingest_file` | Ingest a local document by path |
| `rag_ingest_url` | Ingest a document from a URL |
| `rag_ingest_batch` | Ingest multiple files/URLs at once |
| `rag_query` | Hybrid search with expansion + reranking — use for all search needs |
| `rag_list_documents` | See what documents are indexed |
| `rag_collection_stats` | Check collection size and config |
| `rag_delete_document` | Remove a document and its chunks |
| `rag_service_status` | Check Docker service health |

## Key Config (env vars)

| Variable | Default | Notes |
|----------|---------|-------|
| CHUNK_STRATEGY | hybrid | hybrid / recursive / fixed |
| CHUNK_SIZE | 1024 | Target tokens per chunk |
| ENABLE_QUERY_EXPANSION | true | Master for HyDE + rewrite + multi-query |
| ENABLE_CONTEXTUAL_RETRIEVAL | true | Context prefixes on chunks |
| ENABLE_CACHE | true | Redis caching layer |
| ENABLE_METADATA_EXTRACTION | true | All metadata extractors |
| DOCLING_PICTURE_CLASSIFY | true | Image classification in Docling |
| DOCLING_ENRICH_CODE | false | Opt-in — needs CodeFormula model in serve |
| DOCLING_ENRICH_FORMULA | false | Opt-in — needs CodeFormula model in serve |
| DOCLING_CHART_EXTRACT | false | Opt-in — needs chart model in serve |

## Development Commands

```bash
./setup.sh                    # Bootstrap Docker + models
uv sync                       # Install all deps (includes docling for HybridChunker)
uv run memex                  # Start MCP server
make test                     # 299 tests (246 unit + 53 integration)
make lint                     # Ruff linter
make fmt                      # Auto-format
```

## Architecture

```
MCP Server (local, uv run memex)
  ├── Docling HybridChunker (in-process, docling>=2)
  └── HTTP → Docker Services
       ├── Docling Serve :5001 (GPU doc conversion)
       ├── Ollama :11434 (embeddings + chat LLM)
       ├── Qdrant :6333 (vector DB)
       ├── ML Services :5002 (sparse BM25 + reranker)
       └── Redis :6379 (caching)
```

## Logging

Set `EVAL_LOG_TIMING=true` to see per-stage pipeline timing in logs (embed_query, dense_search, sparse_search, rerank, total_search).
