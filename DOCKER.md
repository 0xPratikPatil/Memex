# Docker Patterns — Memex RAG

Reference for the Memex Docker architecture, guided by the `docker-patterns`, `docker-compose-orchestration`, and `multi-stage-dockerfile` skill families.

## Architecture Overview

```
Host machine
  ├── MCP Server (uv run memex)          # Python, local process
  └── Docker Compose (3 services)        # Backend infrastructure
       ├── qdrant       :6333            # Vector DB
       ├── ollama       :11434           # LLM inference (never on host)
       └── docling      :5001            # Document conversion (GPU)
```

All services bind to `127.0.0.1` — no external exposure. The MCP server talks to them over localhost.

Sparse BM25 embeddings and cross-encoder reranking now run **in-process** via `pip install sentence-transformers fastembed` (installed by `uv sync --extra local`). Redis caching is replaced by an in-memory LRU cache by default, with Redis available as an opt-in addition.

## Docker Compose: 3 Services on One Network

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

### Redis (Opt-in)

Redis caching is optional. To enable, uncomment the `redis` service block in `docker-compose.yml`, uncomment the `redis_data` volume, and restart. Otherwise, the in-memory LRU cache is used automatically.

## Rebuild Strategy

| What Changed | Command |
|-------------|---------|
| Compose config or env vars | `docker compose up -d` (restart only) |
| Ollama model added/removed | `docker compose restart ollama` |
| Volume data | **`docker compose down -v`** (DESTRUCTIVE — deletes all data) |

## Security Checklist

- [ ] No `:latest` tags in any image reference
- [ ] No `0.0.0.0` port binding — all `127.0.0.1`
- [ ] No secrets in Dockerfile or image layers
- [ ] All services use `no-new-privileges:true`
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

## Debugging

```bash
# View service status
docker compose ps

# Follow logs
docker compose logs -f ollama
docker compose logs -f docling

# Force full restart
docker compose down && docker compose up -d

# Validate compose config
docker compose config --quiet

# Resource usage
docker stats
```

## References

- [Docker Compose file reference](https://docs.docker.com/compose/compose-file/)
- [Dockerfile reference](https://docs.docker.com/reference/dockerfile/)
- [Docker security best practices](https://docs.docker.com/develop/security-best-practices/)
- [OCI image spec annotations](https://github.com/opencontainers/image-spec/blob/main/annotations.md)
