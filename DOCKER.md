# Docker — Memex RAG

Reference for the Memex Docker backend. The MCP server runs on the host (`uv run memex`), not in Docker. All three backend services run in containers on a shared bridge network, bound to `127.0.0.1`.

## Architecture Overview

```
Host machine
├── MCP Server (uv run memex)                 # Python process on host
│   ├── HTTP → qdrant   :6333                # Vector DB (HNSW, 1024d)
│   ├── HTTP → ollama   :11434               # LLM inference (Docker-only)
│   ├── HTTP → docling  :5001                # Document parsing + chunking
│   └── In-process ML: BM25 sparse + reranker # fastembed + sentence-transformers
│
└── Docker Compose (memex project)
    ├── memex-qdrant   qdrant/qdrant:v1.18            :6333, :6334
    ├── memex-ollama   ollama/ollama:0.32.4            :11434
    └── memex-docling  docling-serve-cu130:v1.27.0     :5001
         [network: memex_backend — internal: false for host access]
```

All ports bind to `127.0.0.1` only. Sparse BM25 embeddings and cross-encoder reranking run in-process on the host (installed via `uv sync --extra local`). Redis caching is opt-in (commented out in compose).

## Quick Start

```bash
./setup.sh                       # one-command bootstrap (Docker + models + deps)
docker compose up -d             # start all 3 backend services
docker compose ps                # verify all healthy
uv run memex                     # start MCP server
```

Alternatively, start services individually without `setup.sh`:

```bash
docker compose up -d qdrant      # vector DB only
docker compose up -d ollama      # vector DB + LLM
docker compose up -d docling     # full stack (all 3)
```

## Service Details

| Service | Image | Port | GPU | Health Check | Notes |
|---------|-------|------|-----|-------------|-------|
| `qdrant` | `qdrant/qdrant:v1.18` | `6333` (REST), `6334` (gRPC) | No | TCP port check, interval=15s, start_period=15s, retries=5 | HNSW index, 1024d vectors. `init: true`. memlock unlimited. |
| `ollama` | `ollama/ollama:0.32.4` | `11434` | Yes (`nvidia`, count: all) | TCP port check, interval=15s, start_period=30s, retries=5 | `OLLAMA_KEEP_ALIVE: 24h`, `OLLAMA_NUM_PARALLEL: 4`, `OLLAMA_MAX_LOADED_MODELS: 2`. Models persist in `ollama_data` volume. |
| `docling` | `ghcr.io/docling-project/docling-serve-cu130:v1.27.0` | `5001` | Yes (`nvidia`, count: all) | HTTP `/health`, interval=30s, start_period=30s, retries=5 | Document parsing (PDF, DOCX, HTML, images) + HybridChunker. CUDA 13.0 image. |

### Resource Limits

| Service | CPU limit | Memory limit | CPU reservation | Memory reservation |
|---------|-----------|-------------|-----------------|--------------------|
| `qdrant` | 1.0 | 1G | 0.25 | 256M |
| `ollama` | 4.0 | 6G | 1.0 | 2G |
| `docling` | 4.0 | 6G | 0.5 | 2G |

### Dev Overrides (`compose.override.yaml`)

Automatically loaded by `docker compose up`. Tightens health check intervals and start periods:

| Service | Dev interval | Dev start_period |
|---------|-------------|-----------------|
| `qdrant` | 10s | 10s |
| `ollama` | 10s | 20s |
| `docling` | 15s | 20s |

## Docker Compose Rules

Every service in `docker-compose.yml` follows these rules. All values are the actual config, not aspirational.

### Port Binding
All ports bind to `127.0.0.1` explicitly:
```yaml
ports:
  - "127.0.0.1:${QDRANT_PORT:-6333}:6333"
  - "127.0.0.1:${OLLAMA_PORT:-11434}:11434"
  - "127.0.0.1:${DOCLING_PORT:-5001}:5001"
```
Port numbers are configurable via environment variables (`.env`) with defaults as shown. No `0.0.0.0` binding anywhere.

### Health Checks
Every service has `healthcheck` with `interval`, `timeout`, `start_period`, and `retries`. See Service Details table above for exact values. The `compose.override.yaml` tightens intervals for development.

### Resource Limits
Every service has `deploy.resources.limits` and `reservations` (see Resource Limits table). GPU services (ollama, docling) include `reservations.devices` for NVIDIA GPU allocation.

### Logging
Every service uses `json-file` driver with rotation:
```yaml
logging:
  driver: json-file
  options:
    max-size: "10m"
    max-file: "3"
```

### Security
```yaml
security_opt:
  - no-new-privileges:true
```
Applied to every service. No privileged containers. No secrets in image layers or compose files — secrets go in `.env` (gitignored).

### Restart Policy
```yaml
restart: unless-stopped
```
All services restart automatically unless explicitly stopped.

### Stop Grace Period
| Service | stop_grace_period |
|---------|-------------------|
| `qdrant` | 30s |
| `ollama` | 60s |
| `docling` | 60s |

Longer periods for model-heavy services (ollama, docling) to allow graceful shutdown.

### Tmpfs
Every service mounts `/tmp` as tmpfs:
| Service | tmpfs size |
|---------|-----------|
| `qdrant` | 64M |
| `ollama` | 256M |
| `docling` | 128M |

Keeps temp writes off the container filesystem.

### Labels
Every service carries:
```yaml
labels:
  - "com.memex.service=<name>"
  - "com.memex.description=<description>"
```

### Init
All three services use `init: true` for proper PID 1 signal handling.

### Network
All services share a single `backend` bridge network (`internal: false` so the host MCP server can reach containers via `127.0.0.1`).

### Ulimits
Only `qdrant` has `ulimits` configured — the `memlock` ulimit is set to `-1:-1` (unlimited) for HNSW index performance.

## Deployment Modes

Choose which services to start based on your needs:

### Quick Start (qdrant only + remote API)
Use Qdrant locally, point Ollama/embeddings at a remote API. Only qdrant runs in Docker.
```bash
docker compose up -d qdrant
uv run memex    # configure embedding/llm provider in config.yaml
```

### Local AI (qdrant + ollama)
Add local LLM inference for embeddings and chat. No document ingestion yet.
```bash
docker compose up -d qdrant ollama
# Pull models after Ollama starts:
docker compose exec -T ollama ollama pull qwen3-embedding:0.6b
docker compose exec -T ollama ollama pull qwen2.5:1.5b
```

### Full Local (all 3)
Everything local: vector DB, LLM inference, and document parsing.
```bash
docker compose up -d           # or: ./setup.sh
```
This is the default `./setup.sh` path — all three services started.

## Ollama Management

Ollama runs exclusively in Docker — never install on host. Models persist in the `ollama_data` named volume.

```bash
# Pull models inside the container
docker compose exec -T ollama ollama pull qwen3-embedding:0.6b
docker compose exec -T ollama ollama pull qwen2.5:1.5b

# List installed models
docker compose exec -T ollama ollama list

# Remove a model
docker compose exec -T ollama ollama rm <model>

# Verify Ollama health
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

### Key Ollama Env Vars
| Variable | Value | Purpose |
|----------|-------|---------|
| `OLLAMA_KEEP_ALIVE` | `24h` | Keep models loaded in GPU memory |
| `OLLAMA_HOST` | `0.0.0.0` | Listen on all interfaces inside container |
| `OLLAMA_NUM_PARALLEL` | `4` | Concurrent inference requests |
| `OLLAMA_MAX_LOADED_MODELS` | `2` | Max models in memory simultaneously |
| `NVIDIA_VISIBLE_DEVICES` | `all` | GPU passthrough |

## Volume Management

Two named volumes, both persisted across container restarts and `docker compose down`:

| Volume | Mounted at | Contents |
|--------|-----------|----------|
| `memex_qdrant_data` | `/qdrant/storage` in qdrant | HNSW vector index, payload data, collection config |
| `memex_ollama_data` | `/root/.ollama` in ollama | Downloaded models (several GB each), model configs |

Inspect volumes:
```bash
docker volume ls | grep memex
docker volume inspect memex_qdrant_data
docker volume inspect memex_ollama_data
```

Redis has a commented-out `memex_redis_data` volume — uncomment if enabling Redis.

## When to Rebuild vs Restart

| Change | Action |
|--------|--------|
| Compose config, env vars, or labels | `docker compose up -d` |
| Ollama models added/removed | `docker compose exec ollama ollama pull <model>` then `docker compose restart ollama` |
| Volume data (wipe everything) | `docker compose down -v` (**destroys** all persisted data) |
| Image update (`compose.yml` image tag) | `docker compose down && docker compose up -d` |
| MCP server Python (host) | Just restart `uv run memex` — no Docker changes needed |

## Debugging

```bash
docker compose ps                          # service status
docker compose logs -f ollama              # tail Ollama logs
docker compose logs -f docling             # tail Docling logs
docker compose restart ollama              # restart one service
docker compose down && docker compose up -d  # full restart
docker compose config --quiet              # validate compose syntax
docker stats                               # resource usage
```

## Anti-Patterns

- **`:latest` tags** — always pin to specific versions. The compose file uses `v1.18`, `0.32.4`, `v1.27.0`.
- **`0.0.0.0` binding** — always bind to `127.0.0.1`. No external exposure.
- **Secrets in compose or images** — all credentials go in `.env` (gitignored).
- **Data in container filesystem** — always use named volumes (`memex_qdrant_data`, `memex_ollama_data`).
- **Ollama on host** — Docker-only. Models live in the `memex_ollama_data` volume. Use `docker compose exec` for all Ollama operations.
- **Privileged containers** — every service uses `no-new-privileges:true`.
- **No resource limits** — CPU and memory limits are set on every service.
- **No health checks** — every service has health checks with explicit intervals, timeouts, and retries.
