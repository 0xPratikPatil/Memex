# Docker Patterns — Memex RAG

Reference for the Memex Docker architecture, guided by the `docker-patterns`, `docker-compose-orchestration`, and `multi-stage-dockerfile` skill families.

## Architecture Overview

```
Host machine
  ├── MCP Server (uv run memex)          # Python, local process
  └── Docker Compose (5 services)        # Backend infrastructure
       ├── qdrant       :6333            # Vector DB
       ├── ollama       :11434           # LLM inference (never on host)
       ├── docling      :5001            # Document conversion (GPU)
       ├── ml-services  :5002            # Sparse BM25 + reranker (GPU)
       └── redis        :6379            # Search + embedding cache
```

All services bind to `127.0.0.1` — no external exposure. The MCP server talks to them over localhost.

## Dockerfile: 4-Stage ML Services Image

Located at `./Dockerfile`. Stages in order of most to least stable:

```
Stage 0: uv-tool      → ghcr.io/astral-sh/uv:0.6.0           (pinned tool)
Stage 1: python-base  → pytorch/pytorch:2.6.0-cuda12.6-rt    (system + pip deps)
Stage 2: model-cache  → FROM python-base                       (pre-cache ML models)
Stage 3: runtime      → FROM python-base                       (minimal production image)
```

### Stage Responsibilities

| Stage | What it does | Rebuilds when |
|-------|-------------|---------------|
| `uv` | Provides pinned `uv` binary | uv version changes |
| `deps` | System packages + Python venv | `apt` or `pip` deps change |
| `preload` | Downloads HuggingFace models at build time | Model names change |
| `ml` | Copies models + code, sets up non-root user | `rag/ml_server.py` changes |

### Key Rules

1. **Always pin versions** — every base image, every pip package uses exact versions/ranges
2. **BuildKit cache mounts** — `RUN --mount=type=cache` for uv pip downloads
3. **`COPY --link`** on all `--from` copies in final stage (avoids preserving intermediate layers)
4. **Non-root user** — runtime stage MUST use `USER 1001` (explicit UID:GID)
5. **gcc/g++ are required** — Triton compiles CUDA kernels at model load time; always include in apt-get
6. **Pre-cache all models** — the `preload` stage downloads models so containers start instantly
7. **No secrets in layers** — all credentials come from env vars at runtime
8. **`SHELL` with `pipefail`** — all `RUN` scripts use `bash -o pipefail`

## Docker Compose: 5 Services on One Network

Located at `./docker-compose.yml`.

### Every Service Has

- **Pinned image tag** (no `:latest`)
- **`127.0.0.1` port binding** (no `0.0.0.0`)
- **Health check** with `interval`, `timeout`, `start_period`, `retries`
- **Resource limits** (`deploy.resources.limits` + `reservations`)
- **`no-new-privileges:true`** security opt
- **`json-file` logging** with `max-size: 10m`, `max-file: 3`
- **`unless-stopped` restart** policy
- **Named volumes** for persistent data

### Service-Specific Notes

| Service | Image/Version | GPU | Health Check | Notes |
|---------|--------------|-----|--------------|-------|
| qdrant | `qdrant/qdrant:v1.18` | No | TCP port check | HNSW index, 1024d vectors |
| ollama | `ollama/ollama:0.32.4` | Yes | TCP port check | `OLLAMA_NUM_PARALLEL: 4`, `MAX_LOADED_MODELS: 2` |
| docling | `ghcr.io/docling-project/docling-serve-cu130:v1.27.0` | Yes | HTTP /health | Document parsing + hybrid chunking |
| ml-services | Built from `./Dockerfile` | Yes | HTTP /health | BM25 sparse + causal-LM reranker |
| redis | `redis:7.4.10-alpine` | No | redis-cli PING | `maxmemory: 256mb`, `allkeys-lru` |

### Ollama in Docker (never on host)

Ollama runs exclusively in Docker. The `ollama_data` named volume persists models across restarts.

```bash
# Pull models inside the container
docker compose exec -T ollama ollama pull qwen3-embedding:0.6b
docker compose exec -T ollama ollama pull qwen2.5:1.5b

# List installed models
docker compose exec -T ollama ollama list

# Verify
curl http://localhost:11434/api/tags
```

## Rebuild Strategy

| What Changed | Command |
|-------------|---------|
| Python packages (`pip install` list) | `docker compose build --no-cache ml-services && docker compose up -d` |
| `rag/ml_server.py` | `docker compose build ml-services && docker compose up -d` |
| System deps (apt-get) | `docker compose build --no-cache ml-services && docker compose up -d` |
| Compose config or env vars | `docker compose up -d` (restart only) |
| Ollama model added/removed | `docker compose restart ollama` |
| Volume data | **`docker compose down -v`** (DESTRUCTIVE — deletes all data) |

### When to Use --no-cache

`--no-cache` forces a full rebuild and re-downloads all HuggingFace models (~2GB). Only use when:
- Changing system packages (`apt-get install`)
- Changing the base image
- Cached layer is producing wrong results

Model downloads are expensive — prefer `docker compose build ml-services` (with cache) for code changes.

## Security Checklist

- [ ] No `:latest` tags in any image reference
- [ ] No `0.0.0.0` port binding — all `127.0.0.1`
- [ ] No secrets in Dockerfile or image layers
- [ ] All services use `no-new-privileges:true`
- [ ] Runtime stage uses non-root user (`USER 1001`)
- [ ] No `--privileged` containers
- [ ] `.env` is gitignored, `.env.example` is committed
- [ ] Log rotation configured (10m/3 files)
- [ ] Resource limits set on every service

## Anti-Patterns (NEVER do)

- **`:latest` tags** — always pin to specific versions
- **Root containers** — always use non-root user in Dockerfile
- **Secrets in images** — all credentials from env vars at runtime
- **`0.0.0.0` binding** — always bind to `127.0.0.1`
- **Data in container filesystem** — always use named volumes
- **Ollama on host** — Docker-only
- **Unnecessary `--no-cache`** — model downloads are expensive (~2GB)
- **Missing gcc/g++** — breaks Triton CUDA kernel compilation for the reranker

## Debugging

```bash
# View service status
docker compose ps

# Follow logs
docker compose logs -f ollama
docker compose logs -f ml-services

# Shell into a container
docker compose exec ml-services bash

# Check network connectivity
docker compose exec ml-services curl -s http://ollama:11434/api/tags

# Force full restart
docker compose down && docker compose up -d

# Validate compose config
docker compose config --quiet

# Resource usage
docker stats
```

## Model Notes

### Triton Kernel Compilation (Reranker)

The `Qwen/Qwen3-Reranker-0.6B` causal-LM reranker uses Triton to compile CUDA kernels at model load time. This requires `gcc` and `g++` in the image. Without them, `accelerate` falls back to eager execution which is 10-100x slower.

The Dockerfile installs `gcc` and `g++` via apt-get for this reason. Do NOT remove them.

### Temperature=0 Compatibility

The `qwen2.5:1.5b` CHAT_MODEL correctly handles `temperature=0` (no `thinking` mode). Do NOT switch to a "thinking" model (e.g. `qwen3.5:0.8b`) without testing structured output parsing — thinking models may return content in `thinking` field instead of `content` field.

## References

- [Docker Compose file reference](https://docs.docker.com/compose/compose-file/)
- [Dockerfile reference](https://docs.docker.com/reference/dockerfile/)
- [Docker security best practices](https://docs.docker.com/develop/security-best-practices/)
- [OCI image spec annotations](https://github.com/opencontainers/image-spec/blob/main/annotations.md)
