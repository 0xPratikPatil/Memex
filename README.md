# Memex — Personal RAG

Thin MCP server for personal RAG. Models run in Docker; MCP only does HTTP orchestration.

## Prerequisites

- **Python 3.12+** — check with `python3 --version`
- **Docker Engine + Compose** — check with `docker compose version`
- **NVIDIA GPU + drivers** (recommended) — for Ollama and Docling acceleration
- **uv** — Python package manager: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Quick Start

```bash
# 1. Bootstrap everything (Docker services, models, Python deps)
./setup.sh

# 2. Start the MCP server
uv run memex serve
```

`setup.sh` handles the full setup: installs system prerequisites, runs server hardening (kernel swap accounting, Docker daemon config, GPU passthrough verification), creates `.env` and `config.yaml` from templates if missing, starts all Docker services (including Redis), pulls Ollama models, and verifies health checks.

> **First run on a fresh server:** hardening may enable kernel swap accounting and require a reboot. If so, setup stops with a clear message — reboot and re-run `./setup.sh` to continue. Skip hardening with `./setup.sh --no-hardening`.

### Override models via env vars

```bash
EMBED_MODEL=qwen3-embedding:0.6b CHAT_MODEL=qwen2.5:1.5b ./setup.sh
```

### Manual setup (without setup.sh)

```bash
# 1. Create config files
cp .env.example .env              # add any API keys needed
cp config.example.yaml config.yaml  # edit to taste

# 2. Install Python deps
uv sync --extra local             # includes fastembed + sentence-transformers

# 3. Start Docker services
docker compose up -d
docker compose ps                 # verify all healthy

# 4. Pull Ollama models
docker compose exec -T ollama ollama pull qwen3-embedding:0.6b
docker compose exec -T ollama ollama pull qwen2.5:1.5b

# 5. Start MCP server
uv run memex serve
```

## Architecture

```
Host machine
├── MCP Server (uv run memex)                  # Python process on host
│   ├── HTTP → qdrant      :6333              # Vector DB (HNSW, 1024d)
│   ├── HTTP → ollama      :11434             # LLM inference (Docker-only)
│   ├── HTTP → docling     :5001              # Document parsing + chunking
│   ├── HTTP → ml-services :5002              # BM25 sparse + reranker (Docker)
│   ├── HTTP → redis       :6379              # Caching (persistent layer)
│   └── In-process ML: fastembed + sentence-transformers  # local fallback mode
│
└── Docker Compose (5 containers, all on 127.0.0.1)
    ├── memex-qdrant     qdrant/qdrant:v1.18            :6333, :6334
    ├── memex-ollama     ollama/ollama:0.32.4            :11434
    ├── memex-docling    docling-serve-cu130:v1.30.0     :5001
    ├── memex-ml         ml-services (built from Dockerfile) :5002
    └── memex-redis      redis:7.4.10-alpine             :6379
         [network: backend — internal: false for host access]
```

## Features

- **Multi-provider LLM** — Ollama, OpenAI, Anthropic, Groq, Google, OpenRouter
- **Multi-provider embedding** — Ollama, OpenAI, HuggingFace, FastEmbed
- **Hybrid search** — dense (qwen3-embedding:0.6b, 1024d) + sparse BM25 + RRF fusion (k=60)
- **MMR search** — Maximal Marginal Relevance for diverse results, configurable lambda
- **Cross-encoder reranking** — Qwen3-Reranker-0.6B (Docker or in-process), fallback to bge-reranker-base
- **Query expansion** — HyDE, Multi-Query (N paraphrases + RRF), Query Rewrite (master toggle per-method)
- **Contextual Retrieval** — LLM-generated context prefixes on every chunk for better embedding quality
- **Cited answers** — structured answers with `[N]` citations, refusal detection (`INSUFFICIENT_CONTEXT`), citation confidence
- **Agent filter tools** — discover metadata fields/values; extract filters from natural language
- **Docling HybridChunker** — tokenizer-aware, structure-preserving chunking on DoclingDocument; repeats table headers
- **Multi-format embedding** — table chunks → HTML, code chunks → fenced markdown, images → `[Image: caption]`
- **Metadata extraction** — entities, topics, document classification, language detection, dates, keywords
- **Content-hash dedup** — SHA256-based; skips re-indexing unchanged content; partial ingest recovery
- **Search cache** — in-memory LRU with Redis persistent layer; caches full result sets
- **Embedding cache** — in-memory LRU; caches dense vectors
- **Pluggable sources** — local directories + S3 buckets; defined in `config.yaml`
- **Sync engine** — reconciles collection against sources (add, replace, delete); safety rails suppress deletions on source failure
- **Rich progress bars** — live per-file stage tracking for `sync`, `ingest`, and `eval` CLI commands
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

All CLI commands display Rich progress bars with per-file stage tracking:

- **`memex ingest`** — inline progress per file (Converting → Ingesting → Done/Error)
- **`memex sync`** — file-level progress through reconciliation stages (Scanning → Reconciling → Hashing → Parsing → Ingesting → Done)
- **`memex eval`** — per-query progress with results table and aggregate metrics

## Configuration

`config.yaml` is the single source of truth. Copy from the template:

```bash
cp config.example.yaml config.yaml
```

Secrets go in `.env` (gitignored) and are referenced as `${VAR}` in `config.yaml`:

```bash
cp .env.example .env
# Edit .env — only needed if using cloud providers or Qdrant Cloud
```

Key config sections:

| Path | Default | Notes |
|------|---------|-------|
| `embedding.provider` | `ollama` | ollama / openai / huggingface / fastembed |
| `embedding.model` | `qwen3-embedding:0.6b` | 1024d, Ollama |
| `embedding.fallback_model` | `bge-m3` | fallback if primary unavailable |
| `llm.provider` | `ollama` | ollama / openai / openrouter / anthropic / groq / google |
| `llm.model` | `qwen2.5:1.5b` | Ollama chat model |
| `chunking.strategy` | `hybrid` | hybrid / recursive / fixed |
| `chunking.size` | `1024` | target tokens per chunk |
| `sparse.provider` | `docker` | docker (ml-services container) / local (fastembed) |
| `reranker.provider` | `docker` | docker (ml-services container) / local (sentence-transformers) |
| `reranker.model` | `Qwen/Qwen3-Reranker-0.6B` | primary reranker |
| `search.mode` | `hybrid` | similarity / hybrid / mmr |
| `search.mmr.lambda_mult` | `0.5` | MMR diversity weight |
| `query_expansion.enabled` | `true` | master toggle for HyDE + rewrite + multi-query |
| `query_expansion.multi_query_count` | `3` | number of paraphrases |
| `contextual_retrieval.enabled` | `true` | context prefixes on chunks |
| `metadata.extraction_enabled` | `true` | entities, topics, classification, language |
| `caching.enabled` | `true` | embedding + search cache (Redis persistent layer) |
| `answer.enabled` | `true` | citation-based answer generation |
| `mcp.character_limit` | `25000` | max chars in MCP tool response |

## Docker

Five containers — all ports bound to `127.0.0.1`, GPU support via NVIDIA runtime:

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| Qdrant | `qdrant/qdrant:v1.18` | `6333` | Vector DB (HNSW, 1024d) |
| Ollama | `ollama/ollama:0.32.4` | `11434` | LLM inference (embeddings + chat) |
| Docling | `ghcr.io/docling-project/docling-serve-cu130:v1.30.0` | `5001` | Document parsing + HybridChunker |
| ML Services | Built from `Dockerfile` | `5002` | BM25 sparse embeddings + reranker |
| Redis | `redis:7.4.10-alpine` | `6379` | Caching (persistent layer) |

```bash
docker compose up -d          # start all services
docker compose ps             # verify all healthy
docker compose logs -f ollama # tail logs
docker compose down           # stop everything
docker compose down -v        # stop + remove persisted data
```

## AI Coding Tool Integration

Memex runs as an MCP server over stdio. Connect it to any MCP-compatible coding tool:

### OpenCode

Add to `~/.config/opencode/opencode.jsonc`:

```json
{
  "mcp": {
    "memex": {
      "type": "local",
      "command": ["uv", "run", "memex", "serve"],
      "cwd": "/path/to/memex",
      "enabled": true,
      "timeout": 60000
    }
  }
}
```

### Claude Code

```bash
claude mcp add memex -- uv run memex serve
```

Or add to `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "memex": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "memex", "serve"]
    }
  }
}
```

### Cursor / Windsurf

Create `.cursor/mcp.json` in your project:

```json
{
  "mcpServers": {
    "memex": {
      "command": "uv",
      "args": ["run", "memex", "serve"],
      "cwd": "/path/to/memex"
    }
  }
}
```

### VS Code (GitHub Copilot)

Create `.vscode/mcp.json` in your project:

```json
{
  "servers": {
    "memex": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "memex", "serve"],
      "cwd": "${workspaceFolder}"
    }
  }
}
```

After connecting, the agent can use all 13 MCP tools (`rag_query`, `rag_ingest_file`, `rag_sync`, etc.) to search, ingest, and manage your personal knowledge base.

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
| `huggingface` | remote (HF API) | needs `api_key` + model |
| `fastembed` | local (in-process) | no network, `uv sync --extra local` |

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

See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow and code style.

## Documentation

- [DOCKER.md](DOCKER.md) — Docker service reference, deployment modes, debugging
- [CONTRIBUTING.md](CONTRIBUTING.md) — Development workflow, testing, code style
- [CHANGELOG.md](CHANGELOG.md) — Version history
- [AGENTS.md](AGENTS.md) — OpenCode agent instructions (features, config, architecture)
