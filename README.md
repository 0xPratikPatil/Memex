# Memex Server

Production-ready MCP server for Retrieval-Augmented Generation with Docling document conversion, Qdrant vector storage, and Ollama embeddings.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     HOST MACHINE                            │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              MCP Server (Local)                     │   │
│  │                                                     │   │
│  │  - Runs directly on host                           │   │
│  │  - Direct filesystem access                        │   │
│  │  - HybridChunker via Docker Docling Serve            │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           │ HTTP                            │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Docker Services                        │   │
│  │                                                     │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │   │
│  │  │ Qdrant  │ │ Ollama  │ │ Docling │ │  Redis  │  │   │
│  │  │ :6333   │ │ :11434  │ │ :5001   │ │ :6379   │  │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘  │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────┐   │   │
│  │  │ ML Services :5002                           │   │   │
│  │  │  - Sparse BM25 (Qdrant/bm25)                │   │   │
│  │  │  - Causal-LM Reranker (Qwen/Qwen3-Reranker-0.6B)   │   │   │
│  │  └─────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Docling v1 API**: Document conversion (PDF, DOCX, PPTX, XLSX, HTML, images, CSV, Markdown) via Docling Serve
- **Hybrid Chunking**: Docling HybridChunker — tokenizer-aware, structure-preserving (headings, tables, captions, lists)
- **Multi-format Embedding**: Table chunks → HTML, code chunks → fenced, text → Markdown
- **Docling Enrichment**: Picture classification, image export, code/formula/chart extraction (opt-in)
- **Contextual Retrieval**: LLM-generated context prefixes for each chunk (Anthropic's contextual retrieval)
- **Hybrid Search**: Dense (qwen3-embedding:0.6b, 1024d) + Sparse (BM25) + RRF fusion + causal-LM rerank (Qwen/Qwen3-Reranker-0.6B)
- **Query Expansion**: HyDE (hypothetical document), query rewrite, multi-query paraphrasing
- **Metadata Extraction**: Entities, topics, document classification, language detection, dates, keywords — LLM-powered via qwen2.5:1.5b
- **Redis Caching**: Embedding cache, search cache, parse cache — all enabled by default
- **8 MCP Tools**: Ingest, search, list, stats, delete, status, batch, URL ingest

## Quick Start

```bash
./setup.sh        # bootstrap everything (Docker + models + config)
uv sync           # install MCP server deps
uv run memex      # start MCP server
```

Add to your MCP client config:
```json
{
  "mcpServers": {
    "personal_rag": {
      "command": "uv",
      "args": ["run", "memex"],
      "cwd": "/path/to/memex"
    }
  }
}
```

## Development

```bash
uv sync --extra dev --extra test
make test
make lint
make fmt
make help
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `rag_ingest_file` | Ingest a document from a local file path |
| `rag_ingest_url` | Ingest a document from a URL |
| `rag_ingest_batch` | Batch ingest multiple files/URLs |
| `rag_query` | Hybrid search with HyDE + multi-query expansion + reranking |
| `rag_list_documents` | List all ingested documents |
| `rag_collection_stats` | Get collection statistics |
| `rag_delete_document` | Remove a document and its chunks |
| `rag_service_status` | Check backend service health |

## Configuration

All settings via environment variables. Copy and edit `.env.example`:

```bash
cp .env.example .env
```

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `qwen3-embedding:0.6b` | Embedding model (Ollama) |
| `CHAT_MODEL` | `qwen2.5:1.5b` | Chat/LLM model for context, metadata, query expansion |
| `CHUNK_SIZE` | `1024` | Target chunk size (tokens) |
| `CHUNK_STRATEGY` | `hybrid` | `hybrid`, `recursive`, or `fixed` |
| `ENABLE_CACHE` | `true` | Redis caching (embeddings, search, parse) |
| `ENABLE_QUERY_EXPANSION` | `true` | HyDE + rewrite + multi-query |
| `ENABLE_HYDE` | `true` | Hypothetical document search |
| `ENABLE_MULTI_QUERY` | `true` | Paraphrase-based recall boost |
| `ENABLE_QUERY_REWRITE` | `true` | LLM query refinement |
| `ENABLE_CONTEXTUAL_RETRIEVAL` | `true` | Context prefixes for chunks |
| `ENABLE_METADATA_EXTRACTION` | `true` | Entities, topics, language, classification |
| `DOCLING_PICTURE_CLASSIFY` | `true` | Image type classification in Docling |
| `DOCLING_IMAGE_EXPORT` | `embedded` | Base64 inline images in Markdown |

**Full reference**: See `.env.example` for all 80+ options.

## Docker Services

All backend services run in Docker. The MCP server runs locally (not in Docker) for direct filesystem access.

| Service | Port | Image | Purpose |
|---------|------|-------|---------|
| Qdrant | 6333 | `qdrant/qdrant:v1.18` | Vector database |
| Ollama | 11434 | `ollama/ollama:0.32.4` | Embeddings + chat LLM (GPU) |
| Docling | 5001 | `ghcr.io/docling-project/docling-serve-cu130:v1.27.0` | Document conversion (GPU) |
| ML Services | 5002 | Built from `Dockerfile` | Sparse BM25 + reranker (GPU) |
| Redis | 6379 | `redis:7.4.10-alpine` | Caching layer |

> **Important**: Ollama runs exclusively in Docker — do NOT install Ollama on the host. All `localhost:11434` access goes through Docker port mapping. Models are persisted in the `ollama_data` Docker volume.

## Project Structure

```
memex/
├── memex/                  # MCP server (thin — HTTP orchestration only)
│   ├── __init__.py
│   ├── cli.py              # Entry point (uv run memex)
│   ├── server.py           # MCP tool definitions (8 tools)
│   └── status.py           # Service health checker
├── rag/                    # RAG engine (backend logic)
│   ├── __init__.py
│   ├── config.py           # Env-driven central configuration (80+ options)
│   ├── chunking.py         # Docling HybridChunker via Serve API
│   ├── pipeline.py         # RAGEngine: embeddings, Qdrant, search
│   ├── docling_client.py   # Docling document conversion client
│   ├── ml_server.py        # ML services (runs in Docker)
│   ├── services/           # Business logic
│   │   ├── cache.py        # Redis caching layer
│   │   ├── contextual_retrieval.py  # Context prefixes for chunks
│   │   ├── evaluation.py   # RAGAS evaluation framework
│   │   ├── metadata_extractor.py  # Entity, topic, language extraction
│   │   └── query_expansion.py  # HyDE + Multi-Query + Rewrite
│   └── utils/              # Shared utilities
├── Dockerfile              # ML services container (uv-based, multi-stage)
├── docker-compose.yml      # Backend services (5 containers)
├── setup.sh                # One-command bootstrap
├── Makefile                # Development commands
├── tests/
│   ├── unit/               # 257 unit tests
│   ├── integration/        # 53 integration tests
│   └── fixtures/           # Test data
├── scripts/
│   ├── evaluate.py         # Evaluation CLI tool
│   ├── test_e2e.py         # End-to-end verification
│   └── verify_features.py  # 55-check feature verification
├── docs/
│   └── superpowers/specs/  # Design specifications
└── pyproject.toml          # uv-managed deps with chunking/extra
```

## Chunking Strategies

**Hybrid** (default): Docling Serve HybridChunker via `/v1/chunk/hybrid/source` API. Tokenizer-aware (aligned to qwen3-embedding), two-pass (split oversized + merge undersized peers). Preserves headings, captions, table structure. Repeats table headers across chunk boundaries. All heavy processing runs in Docker — no local `docling` packages needed.

**Recursive**: Legacy fallback. Regex-splits markdown by headers → paragraphs → sentences → words.

**Fixed**: Simple word-count splitting. Fastest but lowest quality.

## Search Pipeline

```
User query
  → Query Rewrite (LLM expands ambiguous queries)
  → HyDE (LLM generates hypothetical answer, embeds it)
  → Multi-Query (3 paraphrases, each embedded + searched)
  → Dense search (qwen3-embedding:0.6b, semantic) + Sparse search (BM25, lexical)
  → RRF fusion (k=60, merges all rankings)
  → Cross-encoder / causal-LM rerank (Qwen/Qwen3-Reranker-0.6B, fallback bge-reranker-base)
  → Top-k results
```

Each stage fails gracefully — a HyDE failure doesn't block multi-query, a rerank failure returns RRF results as-is.

## Testing

```bash
make test                # 310 tests (257 unit + 53 integration)
make e2e                 # 9/9 end-to-end checks
uv run python scripts/verify_features.py  # 55-check feature verification
pytest tests/ -v         # verbose output
pytest tests/ --cov=rag  # coverage report
```

## License

MIT
