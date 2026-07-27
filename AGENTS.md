# AGENTS.md — Memex RAG Project Instructions

OpenCode should proactively use the following features when working with this codebase.

## Available RAG Features (all enabled by default)

### Query Expansion
- **HyDE** (ENABLE_HYDE=true): LLM generates a hypothetical answer document, embeds it, and searches alongside the original query. Improves recall for conceptual queries.
- **Query Rewrite** (ENABLE_QUERY_REWRITE=true): LLM expands ambiguous or conversational queries into well-formed search queries.
- **Multi-Query** (ENABLE_MULTI_QUERY=true, MULTI_QUERY_COUNT=3): Generates 3 paraphrases of the query, searches each independently, fuses via RRF.
- All controlled by ENABLE_QUERY_EXPANSION=true (master toggle).

### Search
- **Hybrid Search**: Dense (qwen3-embedding:0.6b, 1024d, fallback bge-m3) + Sparse (BM25 via Qdrant/bm25) + RRF fusion (k=60) + cross-encoder/causal-LM rerank (Qwen/Qwen3-Reranker-0.6B, fallback BAAI/bge-reranker-base).
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
uv sync                       # Install all deps (no local docling needed)
uv run memex                  # Start MCP server
make test                     # 310 tests (257 unit + 53 integration)
make lint                     # Ruff linter
make fmt                      # Auto-format
```

## Architecture

```
MCP Server (local, uv run memex)
  └── HTTP → Docker Services
       ├── Docling Serve :5001 (GPU doc conversion + HybridChunker)
       ├── Ollama :11434 (embeddings + chat LLM)
       ├── Qdrant :6333 (vector DB)
       ├── ML Services :5002 (sparse BM25 + reranker)
       └── Redis :6379 (caching)
```

## Docker Ollama (required — no host Ollama)

Ollama runs **exclusively in Docker**. Never install Ollama on the host.

- All `localhost:11434` requests go through Docker port mapping to the `ollama/ollama:0.32.4` container
- Models are persisted in the `ollama_data` Docker volume (survives `docker compose down`)
- Model pulls happen inside the container: `docker compose exec -T ollama ollama pull <model>`
- Health check: `curl http://localhost:11434/api/tags` (hits Docker container)

## Practical Testing Setup

### First-time setup
```bash
./setup.sh                    # bootstraps Docker, pulls models, verifies everything
```

### Start services
```bash
docker compose up -d          # start all backend services
docker compose ps             # verify all healthy
```

### Pull / manage models
```bash
docker compose exec -T ollama ollama list           # list downloaded models
docker compose exec -T ollama ollama pull qwen3-embedding:0.6b  # pull embedding model
docker compose exec -T ollama ollama pull qwen3.5:0.8b  # pull chat model
docker compose exec -T ollama ollama rm <model>     # remove a model
```

### Verify Ollama is working
```bash
# Health check
curl http://localhost:11434/api/tags

# Test embedding
curl -s -X POST http://localhost:11434/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-embedding:0.6b","prompt":"test"}' | jq .

# Test chat
curl -s -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5:0.8b","messages":[{"role":"user","content":"hi"}],"stream":false}' | jq .
```

### Run the MCP server
```bash
uv run memex                  # starts MCP server, connects to Docker services
```

### Run tests
```bash
make test                     # unit tests only (no Docker needed)
make test-all                 # unit + integration tests
make e2e                      # end-to-end verification (needs Docker)
uv run python scripts/verify_features.py  # 55-check feature verification
pytest tests/unit/ -v         # unit tests only
pytest tests/integration/ -v  # integration tests (needs Docker services running)
```

### Useful Docker commands
```bash
docker compose logs -f ollama           # tail Ollama logs
docker compose logs -f ml-services      # tail ML services logs
docker compose restart ollama           # restart Ollama container
docker compose down && docker compose up -d  # full restart
docker volume ls                        # check ollama_data volume exists
```

## Logging

Set `EVAL_LOG_TIMING=true` to see per-stage pipeline timing in logs (embed_query, dense_search, sparse_search, rerank, total_search).
