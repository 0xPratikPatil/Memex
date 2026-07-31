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
- **MMR Search**: Maximal Marginal Relevance mode for diverse results (exploratory queries). Balances relevance (λ) with diversity. Config: `search.mode=mmr`, `search.mmr.lambda_mult=0.5`.
- **Contextual Retrieval** (ENABLE_CONTEXTUAL_RETRIEVAL=true, CONTEXT_STRATEGY=summary): LLM-generated context prefixes for each chunk, improving embedding quality.
- **Search Cache**: Redis caches full result sets for repeated queries (CACHE_TTL_SEARCH=3600).

### Ingestion
- **Docling HybridChunker** (CHUNK_STRATEGY=hybrid): Tokenizer-aware, structure-preserving chunking on DoclingDocument. Repeats table headers across boundaries.
- **MarkItDown Converter**: Lightweight alternative converter — user-selectable via `converter.engine` config (docling | markitdown). No GPU, faster for simple docs.
- **Multi-format Embedding**: Table chunks → HTML, code chunks → fenced markdown, images → [Image: caption].
- **Docling Enrichment**: Picture classification (enabled), code/formula/chart extraction (opt-in, needs serve-side models).
- **Metadata Extraction** (ENABLE_METADATA_EXTRACTION=true): Entities, topics, document classification, language detection, dates, keywords.
- **Content-Hash Dedup**: SHA256-based dedup prevents re-indexing identical content. Partial ingest recovery on crash.
- **Embedding Cache**: Redis caches dense vectors (CACHE_TTL_EMBEDDING=86400).

### Document Sources & Sync
- **Pluggable Sources**: Local directories + S3 buckets. Define in `config.yaml` → `sources` section.
- **Sync Engine**: `rag_sync` reconciles collection against sources — adds new, replaces changed, removes deleted files. Safety rails suppress deletions if any source fails.

### Answer Generation
- **Cited Answers**: `rag_query` returns structured Answer objects with `[N]` citations, refusal detection (INSUFFICIENT_CONTEXT sentinel), and citation confidence scoring.
- **Agent Filter Tools**: `rag_get_filter_context` shows available metadata fields/values; `rag_extract_filters` parses natural language into metadata filters.

### Evaluation
- **Golden-Set Evaluation**: YAML-based golden sets with recall@K, precision@K, hit_rate@K, MRR, keyword_coverage.
- **Eval Sweep**: Compare multiple retrieval configs side by side with delta comparison. MCP tools + CLI.

## MCP Tools (13 available)

| Tool | Use when |
|------|----------|
| `rag_ingest_file` | Ingest a local document by path |
| `rag_ingest_url` | Ingest a document from a URL |
| `rag_ingest_batch` | Ingest multiple files/URLs at once |
| `rag_query` | Hybrid/MMR search with expansion, reranking, citation answers — use for all search needs |
| `rag_list_documents` | See what documents are indexed |
| `rag_collection_stats` | Check collection size and config |
| `rag_delete_document` | Remove a document and its chunks |
| `rag_service_status` | Check Docker service health |
| `rag_sync` | Sync collection against configured sources |
| `rag_get_filter_context` | Show available metadata fields and values |
| `rag_extract_filters` | Extract metadata filters from natural language |
| `rag_eval` | Run golden-set evaluation |
| `rag_eval_sweep` | Compare multiple retrieval configs side by side |

## Key Config (config.yaml)

All configuration lives in `config.yaml`. Copy `config.example.yaml` to `config.yaml` and customize.

| Path | Default | Notes |
|------|---------|-------|
| `chunking.strategy` | hybrid | hybrid / recursive / fixed |
| `chunking.size` | 1024 | Target tokens per chunk |
| `query_expansion.enabled` | true | Master for HyDE + rewrite + multi-query |
| `contextual_retrieval.enabled` | true | Context prefixes on chunks |
| `caching.enabled` | true | Redis caching layer |
| `metadata.enabled` | true | All metadata extractors |
| `converter.engine` | docling | docling / markitdown |
| `search.mode` | hybrid | similarity / hybrid / mmr |
| `answer.enabled` | true | Citation-based answer generation |
| `docling_picture_classify` | true | Image classification in Docling |
| `docling_enrich_code` | false | Opt-in — needs CodeFormula model |
| `docling_enrich_formula` | false | Opt-in — needs CodeFormula model |
| `docling_chart_extract` | false | Opt-in — needs chart model |

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
       ├── MarkItDown :5003 (lightweight doc conversion)
       ├── Ollama :11434 (embeddings + chat LLM)
       ├── Qdrant :6333 (vector DB)
       ├── ML Services :5002 (sparse BM25 + reranker)
       ├── Redis :6379 (caching)
       └── S3 Service :5004 (S3 source connector)
```

## CLI Commands

```bash
memex ingest /path/to/docs --recursive   # Ingest files or directories
memex sync --dry-run                     # Sync collection against sources
memex eval golden.yaml --top-k 5         # Evaluate retrieval quality
```

## Docker Ollama (required — no host Ollama)

Ollama runs **exclusively in Docker**. Never install Ollama on the host.

- All `localhost:11434` requests go through Docker port mapping to the `ollama/ollama:0.32.4` container
- Models are persisted in the `ollama_data` Docker volume (survives `docker compose down`)
- Model pulls happen inside the container: `docker compose exec -T ollama ollama pull <model>`
- Health check: `curl http://localhost:11434/api/tags` (hits Docker container)

## Docker Best Practices (apply when modifying Dockerfile or compose)

### Multi-Stage Dockerfile Rules

- **Four stages**: `uv` → `deps` → `preload` → `ml` (runtime), in order of most stable to most volatile
- **Pinned versions**: Every base image tag is pinned (`ollama/ollama:0.32.4`, not `:latest`)
- **Non-root user**: Runtime stage MUST use USER with explicit UID:GID (1001:1001)
- **COPY --link**: Use `--link` on all `COPY --from` in final stage to avoid preserving intermediate layers
- **Pre-cache models**: All ML models are downloaded at build time in the `preload` stage — containers start instantly
- **BuildKit cache mounts**: Use `RUN --mount=type=cache` for uv pip to speed up rebuilds
- **Layer ordering**: System packages first, then Python deps, then models, then app code (most volatile last)
- **No secrets in layers**: All credentials come from env vars at runtime, never baked into images
- **Add gcc/g++**: Required for Triton kernel compilation during HuggingFace model loading — always include in `apt-get install`

### Docker Compose Rules

- **Host-only binding**: All ports bind to `127.0.0.1` explicitly — no external exposure
- **Named volumes**: Use named volumes (not anonymous) for all persistent data: `qdrant_data`, `ollama_data`, `redis_data`
- **Health checks**: Every service has a healthcheck with `interval`, `timeout`, `start_period`, and `retries`
- **Resource limits**: Set `deploy.resources.limits` and `reservations` on every service (especially GPU memory)
- **Logging**: `json-file` driver with `max-size: 10m` and `max-file: 3` on every service
- **Security**: `no-new-privileges:true` on every service, no privileged containers
- **Restart policy**: `unless-stopped` for all persistent services
- **stop_grace_period**: Set judiciously (30s for fast shutdown, 60s for model-heavy services like ollama/docling)
- **Environment variables**: Use `${VAR:-default}` syntax for all configurable values, referencing `.env`
- **Single network**: All services share the `backend` bridge network (MCP server runs on host)
- **Tmpfs for /tmp**: Use `tmpfs` mounts for `/tmp` on every service — keeps temp writes off the container filesystem
- **Container names**: Use explicit `container_name: memex-<service>` for readable `docker compose ps` output
- **Service labels**: Each service has `com.memex.service` and `com.memex.description` labels
- **Override file**: `compose.override.yaml` is auto-loaded for development settings (tighter health checks)

### When to Rebuild vs Restart

| Change | Action |
|--------|--------|
| `rag/ml_server.py` | `docker compose build ml-services && docker compose up -d` |
| `docker/markitdown/server.py` | `docker compose build markitdown && docker compose up -d` |
| `docker/s3-service/server.py` | `docker compose build s3-service && docker compose up -d` |
| Python packages in Dockerfile | `docker compose build --no-cache ml-services && docker compose up -d` |
| System deps (apt-get) | `docker compose build --no-cache ml-services && docker compose up -d` |
| Compose config, env vars, or labels | `docker compose up -d` (restart) |
| Ollama models | `docker compose exec ollama ollama pull <model>` + restart if parallel changed |
| Volume data | `docker compose down -v` (destroys all persisted data) |
| MCP server Python (host) | Just restart `uv run memex` — no Docker rebuild needed |

### Anti-Patterns (NEVER do)

- Never use `:latest` tags — always pin to specific versions
- Never run containers as root — always use non-root user in Dockerfile
- Never bake secrets/API keys into image layers
- Never expose services on `0.0.0.0` — always bind to `127.0.0.1`
- Never store data in container filesystem — always use named volumes
- Never install Ollama on host — Docker-only
- Never use `--no-cache` unless absolutely necessary (model downloads are expensive)
- Never leave gcc/g++ out of apt-get (breaks Triton kernel compilation)

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
docker compose exec -T ollama ollama pull qwen2.5:1.5b  # pull chat model
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
  -d '{"model":"qwen2.5:1.5b","messages":[{"role":"user","content":"hi"}],"stream":false}' | jq .
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
