# GPU Passthrough + File Intelligence + Progress Visibility

**Date:** 2026-07-26
**Status:** Approved
**Problem:** No GPU utilization, no file dedup, no progress feedback, frequent timeouts

## Problem Statement

The personal-rag-mcp stack runs entirely on CPU despite having an RTX 3070 (8GB VRAM). The Docling image (`cu130`) has CUDA baked in but no GPU device is passed to any container. This causes:

1. Docling conversion takes 30-120s on CPU vs ~2-5s on GPU
2. Re-ingesting the same file wastes full pipeline time (no dedup check)
3. No progress feedback during long operations
4. Timeout chain breaks: 120s Docling timeout insufficient for CPU-bound work

## Solution Overview

Three parallel workstreams:

1. **GPU Passthrough** — Add NVIDIA device reservation to Docling, Ollama, and MCP services
2. **File Intelligence** — SHA256-based dedup: skip re-ingestion of unchanged files
3. **Progress Visibility** — MCP protocol progress notifications during ingestion

## Architecture

### GPU Passthrough

Add to each GPU-capable service in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
environment:
  NVIDIA_VISIBLE_DEVICES: all
```

**VRAM Budget (8GB RTX 3070):**

| Component | VRAM | When Active |
|-----------|------|-------------|
| Ollama bge-m3 | ~1.3GB | Always (embedding) |
| Docling models | ~3-5GB | Ingestion only |
| Reranker | ~1.1GB | Search only |

Peak during ingestion: ~6.3GB (Ollama + Docling). Peak during search: ~2.4GB (Ollama + Reranker). Docling and reranker never run simultaneously — fits in 8GB.

### File Intelligence

**Storage:** SHA256 hash stored in Qdrant payload as `content_hash` field.

**Flow on ingest:**

1. Read file bytes directly from filesystem (using pathlib)
2. Compute SHA256 of bytes
3. Query Qdrant: does source exist with matching `content_hash`?
4. If match → return "Already ingested (22 chunks, hash: abc123...)"
5. If no match (new or changed) → proceed with full pipeline
6. After successful ingest, store `content_hash` in Qdrant payload

**Hash storage:** Add `content_hash` field to each point's payload alongside existing metadata.

**Cleanup:** Old chunks from changed files are automatically overwritten by deterministic UUID5 upsert (existing behavior).

### Progress Visibility

Use MCP's built-in `notifications/progress` protocol:

```
Client → Server: tools/call (with progress_token)
Server → Client: notifications/progress (percentage + message)
Server → Client: tools/call result (final)
```

**Progress steps for ingestion:**

| Step | Message | Approximate % |
|------|---------|---------------|
| 1 | Fetching file from server... | 0-10% |
| 2 | Checking if already ingested... | 10-15% |
| 3 | Converting with Docling... | 15-70% |
| 4 | Chunking document... | 70-75% |
| 5 | Generating embeddings (N chunks)... | 75-90% |
| 6 | Storing in Qdrant... | 90-95% |
| 7 | Done | 100% |

**Implementation:** Use `mcp.server.fastmcp` progress reporting via context token.

### Timeout Adjustments

| Setting | Old | New | Reason |
|---------|-----|-----|--------|
| `DOCLING_TIMEOUT` | 120s | 300s | GPU warmup on first call, large docs |
| `HTTP_TIMEOUT` | 30s | 60s | Embedding under load |
| `QDRANT_TIMEOUT` | 10s | 10s | No change (fast) |

Also increase MCP client timeout in `opencode.jsonc` to 600000 (already set).

## Files to Modify

| File | Changes |
|------|---------|
| `docker-compose.yml` | GPU device reservation on docling, ollama, mcp services |
| `config.py` | Increase DOCLING_TIMEOUT, HTTP_TIMEOUT |
| `src/pipeline.py` | Add `content_hash` to payload, add hash-check method, add progress reporting |
| `src/docling_client.py` | No changes needed |
| `src/server.py` | Add progress token handling, pass progress callback to engine |
| `.env.example` | Document new GPU env vars |

## Testing

1. **GPU test:** `docker compose exec docling nvidia-smi` should show GPU inside container
2. **Dedup test:** Ingest same file twice — second time should return "Already ingested" instantly
3. **Progress test:** Ingest a large PDF, observe step-by-step progress notifications
4. **Timeout test:** Ingest a large file, verify no timeout errors
5. **VRAM test:** Monitor `nvidia-smi` during ingestion — should see VRAM usage spike then drop

## Risks

- **VRAM OOM:** If Docling loads too many models, may exceed 8GB. Mitigation: Docling has model selection flags.
- **First-call latency:** GPU warmup adds ~5-10s to first inference. Subsequent calls are fast.
- **Hash collision:** SHA256 collision is astronomically unlikely. Not a practical concern.
