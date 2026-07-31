# Memex — Personal RAG

Thin MCP server for personal RAG. Models run in Docker; MCP only does HTTP orchestration.

## Quick Start

```bash
./setup.sh            # bootstrap Docker + models + deps
uv run memex serve    # start MCP server (stdio transport)
```

Override models via env vars:

```bash
EMBED_MODEL=qwen3-embedding:0.6b CHAT_MODEL=qwen2.5:1.5b ./setup.sh
```

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│  HOST MACHINE                                                     │
│  ┌─────────────────────┐                                          │
│  │  memex MCP Server    │───HTTP──▶ Docker (127.0.0.1)            │
│  │  (uv run memex)      │           ┌──────────────────────────┐  │
│  │  • Python (stdio)    │           │ Qdrant   :6333  vector DB │  │
│  └─────────────────────┘           │ Ollama   :11434 LLM/embed│  │
│                                     │ Docling  :5001  converter │  │
│  In-process ML (host)              └──────────────────────────┘  │
│  ┌─────────────────────┐                                          │
│  │ fastembed (BM25)     │                                          │
│  │ sentence-transformers│──reranker (Qwen3-Reranker-0.6B)        │
│  └─────────────────────┘                                          │
└───────────────────────────────────────────────────────────────────┘
```

## Features

- **Multi-provider LLM** — Ollama, OpenAI, Anthropic, Groq, Google, OpenRouter
- **Multi-provider embedding** — Ollama, OpenAI, HuggingFace, FastEmbed
- **Hybrid search** — dense (qwen3-embedding:0.6b, 1024d) + sparse BM25 (fastembed in-process) + RRF fusion (k=60)
- **MMR search** — Maximal Marginal Relevance for diverse results, configurable λ
- **Cross-encoder reranking** — Qwen/Qwen3-Reranker-0.6B (in-process via sentence-transformers), fallback to BAAI/bge-reranker-base
- **Query expansion** — HyDE, Multi-Query (3 paraphrases + RRF), Query Rewrite (master toggle per-method)
- **Contextual Retrieval** — LLM-generated context prefixes on every chunk for better embedding quality
- **Cited answers** — structured answers with `[N]` citations, refusal detection (`INSUFFICIENT_CONTEXT`), citation confidence
- **Agent filter tools** — discover metadata fields/values; extract filters from natural language
- **Docling HybridChunker** — tokenizer-aware, structure-preserving chunking on DoclingDocument; repeats table headers
- **Multi-format embedding** — table chunks → HTML, code chunks → fenced markdown, images → `[Image: caption]`
- **Metadata extraction** — entities, topics, document classification, language detection, dates, keywords
- **Content-hash dedup** — SHA256-based; skips re-indexing unchanged content; partial ingest recovery
- **Search cache** — in-memory LRU (Redis opt-in); caches full result sets
- **Embedding cache** — in-memory LRU; caches dense vectors
- **Pluggable sources** — local directories + S3 buckets; defined in `config.yaml`
- **Sync engine** — reconciles collection against sources (add, replace, delete); safety rails suppress deletions on source failure
- **Golden-set evaluation** — recall@K, precision@K, hit_rate@K, MRR, keyword_coverage
- **Eval sweep** — compare multiple retrieval configs side-by-side with delta comparison

## MCP Tools

| Tool | Description |
|------|-------------|
| `rag_ingest_file` | Ingest a local document by path |
| `rag_ingest_url` | Ingest a document from a URL |
| `rag_ingest_batch` | Ingest multiple files/URLs at once |
| `rag_query` | Hybrid/MMR search with expansion, reranking, and cited answers |
| `rag_list_documents` | List all indexed documents with metadata |
| `rag_collection_stats` | Collection size, vector count, and config |
| `rag_delete_document` | Remove a document and all its chunks |
| `rag_service_status` | Health check for all Docker services |
| `rag_sync` | Sync collection against configured sources |
| `rag_get_filter_context` | Show available metadata fields and values |
| `rag_extract_filters` | Extract metadata filters from natural language |
| `rag_eval` | Run golden-set evaluation |
| `rag_eval_sweep` | Compare multiple retrieval configs side by side |

## CLI Commands

```
memex serve          start MCP server (stdio)
memex ingest PATH    ingest file or directory (--recursive for subdirectories)
memex sync           sync collection against sources (--dry-run, --source-name)
memex eval GOLDEN    evaluate retrieval against golden set (--top-k, --compare-rerank)
```

## Configuration

`config.yaml` is the single source of truth. Copy from the template:

```bash
cp config.example.yaml config.yaml
```

Key sections:

| Path | Default | Notes |
|------|---------|-------|
| `embedding.model` | `qwen3-embedding:0.6b` | 1024d, Ollama |
| `embedding.fallback_model` | `bge-m3` | fallback if primary unavailable |
| `llm.model` | `qwen2.5:1.5b` | Ollama chat model |
| `chunking.strategy` | `hybrid` | hybrid / recursive / fixed |
| `chunking.size` | `1024` | target tokens per chunk |
| `reranker.model` | `Qwen/Qwen3-Reranker-0.6B` | in-process via sentence-transformers |
| `reranker.fallback_model` | `BAAI/bge-reranker-base` | fallback reranker |
| `search.mode` | `hybrid` | similarity / hybrid / mmr |
| `search.mmr.lambda_mult` | `0.5` | MMR diversity weight |
| `query_expansion.enabled` | `true` | master toggle for HyDE + rewrite + multi-query |
| `query_expansion.multi_query_count` | `3` | number of paraphrases |
| `contextual_retrieval.enabled` | `true` | context prefixes on chunks |
| `contextual_retrieval.strategy` | `summary` | context generation strategy |
| `metadata.extraction_enabled` | `true` | entities, topics, classification, language |
| `caching.enabled` | `true` | embedding + search cache (Redis opt-in) |
| `answer.enabled` | `true` | citation-based answer generation |
| `sources` | local `/mnt/documents` | pluggable local + S3 sources |

## Providers

### LLM Providers

| Provider | Location | Notes |
|----------|----------|-------|
| `ollama` | local (Docker) | default; qwen2.5:1.5b |
| `openai` | remote | needs `api_key` + model |
| `anthropic` | remote | needs `api_key` + model |
| `groq` | remote | needs `api_key` + model |
| `google` | remote | needs `api_key` + model |
| `openrouter` | remote | needs `api_key` + model |

### Embedding Providers

| Provider | Location | Notes |
|----------|----------|-------|
| `ollama` | local (Docker) | default; qwen3-embedding:0.6b |
| `openai` | remote | needs `api_key` + model |
| `huggingface` | local (in-process) | loaded via transformers |
| `fastembed` | local (in-process) | loaded via fastembed |

## Docker

Three required containers + one optional:

| Service | Image | Port |
|---------|-------|------|
| Qdrant | `qdrant/qdrant:v1.18` | `127.0.0.1:6333` |
| Ollama | `ollama/ollama:0.32.4` | `127.0.0.1:11434` |
| Docling | `ghcr.io/docling-project/docling-serve-cu130:v1.27.0` | `127.0.0.1:5001` |
| Redis (opt-in) | `redis:7.4.10-alpine` | `127.0.0.1:6379` |

All ports bind to `127.0.0.1` only. GPU support via NVIDIA runtime on Ollama and Docling.

```bash
docker compose up -d          # start all services
docker compose ps             # verify all healthy
docker compose logs -f ollama # tail logs
docker compose down           # stop everything
docker compose down -v        # stop + remove persisted data
```

## Development

```bash
# Install deps
uv sync --extra dev --extra test --extra local

# Run tests
make test       # unit tests only (no Docker needed)
make test-all   # unit + integration tests (needs Docker)
make e2e        # end-to-end verification

# Code quality
make lint       # ruff check
make fmt        # ruff format + fix
make typecheck  # mypy
```

Requirements: Python >= 3.12, Docker.
