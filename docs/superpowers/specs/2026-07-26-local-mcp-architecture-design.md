# Architecture: Local MCP + Docker Services

**Date:** 2026-07-26
**Status:** Proposed

## Current State

All services run in Docker:
- Qdrant (vector DB)
- Ollama (embeddings)
- Docling (document conversion)
- Redis (caching)
- MCP Server (RAG tools)

**Problem:** MCP in Docker needs volume mounts to access host files, causing permission issues.

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     HOST MACHINE                            │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              MCP Server (Local)                     │   │
│  │                                                     │   │
│  │  - Runs directly on host                           │   │
│  │  - Direct filesystem access                        │   │
│  │  - No Docker permission issues                     │   │
│  │  - Connects to Docker services via localhost       │   │
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
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Key Changes

### 1. Remove MCP from Docker

**docker-compose.yml:**
- Remove `mcp` service
- Keep all other services unchanged
- Services now accessible via `localhost` ports

### 2. Create Local MCP Package

**Structure:**
```
memex/
├── src/                    # Existing code (unchanged)
│   ├── config.py
│   ├── server.py
│   ├── pipeline.py
│   └── ...
├── memex_cli/              # NEW: Local MCP package
│   ├── __init__.py
│   ├── __main__.py         # Entry point
│   ├── server.py           # Local server runner
│   └── config.py           # Local config (localhost URLs)
├── docker-compose.yml      # Backend services only
└── run_mcp.py              # Simple script to run MCP locally
```

### 3. Configuration Changes

**Local MCP config (memex_cli/config.py):**
```python
# Services accessible via localhost (Docker ports exposed)
DOCLING_URL = "http://localhost:5001/v1/convert/source"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
QDRANT_URL = "http://localhost:6333"
REDIS_URL = "redis://localhost:6379/0"
```

### 4. Entry Point

**run_mcp.py:**
```python
#!/usr/bin/env python3
"""Run MCP server locally."""

import sys

sys.path.insert(0, "src")
from src.server import mcp

mcp.run()
```

## Benefits

1. **No permission issues** - MCP reads files directly as host user
2. **Simpler debugging** - MCP logs appear in terminal
3. **Faster development** - No Docker rebuild for MCP changes
4. **Clean separation** - Backend in Docker, MCP on host

## Migration Steps

1. **Update docker-compose.yml**
   - Remove MCP service
   - Remove MCP-related volumes
   - Keep all backend services

2. **Create memex_cli package**
   - Add `__init__.py`
   - Add `__main__.py`
   - Add local config

3. **Create run_mcp.py**
   - Simple entry point
   - Sets up path and runs MCP

4. **Update documentation**
   - README with new setup instructions
   - Development guide

## Usage

```bash
# Start backend services
docker compose up -d

# Run MCP locally
python run_mcp.py

# Or via package
python -m memex_cli
```

## Testing

All existing tests should pass since the code is unchanged. Only the deployment architecture changes.

## Rollback Plan

If issues arise, restore MCP service in docker-compose.yml and remove local package.
