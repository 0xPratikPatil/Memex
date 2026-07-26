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
│  │  - No Docker permission issues                     │   │
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

## Features

- **Docling v1 API**: Document conversion (PDF, DOCX, HTML, images) via Docling Serve
- **Recursive Chunking**: Splits by headers, paragraphs, sentences — preserves semantic boundaries
- **Hybrid Search**: Dense (bge-m3) + Sparse (BM25) with Reciprocal Rank Fusion
- **Cross-Encoder Reranking**: Optional reranking via BAAI/bge-reranker
- **Rich Metadata**: Section headers, ingestion timestamps, source tracking
- **8 MCP Tools**: Ingest, search, list, stats, delete, status, batch, URL ingest

## Quick Start

```bash
./setup.sh        # bootstrap everything (Docker + models + config)
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
# Install dev deps
uv sync --extra dev --extra test

# Tests, lint, format
make test
make lint
make fmt
make help     # all targets
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `rag_ingest_file` | Ingest a document from a local file path |
| `rag_ingest_url` | Ingest a document from a URL |
| `rag_ingest_batch` | Batch ingest multiple files/URLs |
| `rag_query` | Hybrid search with optional reranking |
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
| `EMBED_MODEL` | `bge-m3` | Embedding model (Ollama) |
| `CHAT_MODEL` | `qwen2.5:0.5b` | Chat/LLM model for advanced features |
| `CHUNK_SIZE` | `512` | Target chunk size (tokens) |
| `ENABLE_CACHE` | `false` | Enable Redis caching |
| `ENABLE_QUERY_EXPANSION` | `false` | Enable HyDE + Multi-Query |
| `ENABLE_CONTEXTUAL_RETRIEVAL` | `false` | Add context prefixes to chunks |
| `ENABLE_METADATA_EXTRACTION` | `false` | Extract entities, topics, language |

**Full reference**: See `.env.example` for all 60+ options.
| `CHUNK_OVERLAP` | `50` | Overlap between chunks (tokens) |
| `CHUNK_STRATEGY` | `recursive` | Chunking: `recursive` or `fixed` |
| `SEARCH_FUSION` | `rrf` | Fusion: `rrf` (reciprocal rank) or `weighted` |
| `RERANK_ENABLED` | `true` | Enable cross-encoder reranking |
| `MCP_TRANSPORT` | `http` | Transport: `http` or `stdio` |

## Docker Services

| Service | Port | Purpose |
|---------|------|---------|
| Qdrant | 6333 | Vector database |
| Ollama | 11434 | Embeddings + chat LLM |
| Docling | 5001 | Document conversion |
| ML Services | 5002 | Sparse BM25 + reranker |
| Redis | 6379 | Caching layer |

**Note:** MCP server runs locally (not in Docker) for direct file system access.

## Project Structure

```
memex/
├── memex/                  # MCP server (thin — HTTP orchestration only)
│   ├── __init__.py         # v0.4.0
│   ├── cli.py              # Entry point (uv run memex)
│   ├── server.py           # MCP tool definitions (8 tools)
│   └── status.py           # Service health checker
├── rag/                    # RAG engine (backend logic)
│   ├── __init__.py
│   ├── config.py           # Env-driven central configuration
│   ├── pipeline.py         # RAGEngine: embeddings, Qdrant, search
│   ├── docling_client.py   # Docling document conversion client
│   ├── ml_server.py        # ML services (runs in Docker)
│   ├── models/             # Data models
│   ├── services/           # Business logic
│   │   ├── cache.py        # Redis caching layer
│   │   ├── contextual_retrieval.py  # Context prefixes for chunks
│   │   ├── evaluation.py   # RAGAS evaluation framework
│   │   ├── metadata_extractor.py  # Entity extraction
│   │   └── query_expansion.py  # HyDE + Multi-Query
│   └── utils/              # Shared utilities
├── Dockerfile              # ML services container (multi-stage)
├── docker-compose.yml      # Backend services (5 containers)
├── setup.sh                # One-command bootstrap
├── Makefile                # Development commands
├── tests/
│   ├── unit/               # 181 unit tests
│   ├── integration/        # 53 integration tests
│   └── fixtures/           # Test data
├── scripts/
│   ├── evaluate.py         # Evaluation CLI tool
│   └── test_e2e.py         # End-to-end verification
├── docs/
│   └── superpowers/specs/  # Design specifications
├── .github/               # CI/CD workflows
├── Dockerfile             # Multi-stage: MCP server + File server
├── docker-compose.yml     # Full stack orchestration
├── Makefile               # Common development tasks
├── pyproject.toml         # Project metadata and tooling config
├── LICENSE                # MIT license
└── README.md              # Project documentation
```

## Development

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Setup

```bash
# Clone and configure
cd mcp/rag
cp .env.example .env

# Install dependencies (with uv)
uv sync

# Or with pip
pip install -e .

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Running Locally

```bash
# Start infrastructure services only
docker compose up -d qdrant ollama docling

# Pull embedding model into Ollama (first time only)
docker compose exec ollama ollama pull bge-m3

# Run MCP server in stdio mode

# Run MCP server in HTTP mode

# Or use Makefile
make run        # stdio mode
make run-http   # HTTP mode
```

### Common Tasks

```bash
make lint       # Run ruff linter
make fmt        # Auto-format with ruff
make test       # Run pytest
make build      # Build Docker images
make up         # docker compose up -d
make down       # docker compose down
make logs       # docker compose logs -f
make clean      # Remove build artifacts
```

## Testing

```bash
# Run all tests
make test

# Run with verbose output
pytest tests/ -v

# Run a specific test file
pytest tests/unit/test_config.py -v

# Run tests with coverage
pip install pytest-cov
pytest tests/ --cov=src --cov-report=html
```

## Contributing

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes and add tests
3. Run linting and tests: `make lint make test`
4. Commit with a descriptive message
5. Push and open a Pull Request

### Code Style

- Formatter/linter: [Ruff](https://docs.astral.sh/ruff/) (configured in `pyproject.toml`)
- Line length: 120 characters
- Import sorting: `ruff check --select I`
- All code must pass `make lint` before committing

## Chunking Strategies

**Recursive** (default): Splits markdown by headers → paragraphs → sentences → words. Preserves section context.

**Fixed**: Simple word-count splitting. Faster but may break mid-sentence.

## Search

Hybrid search combines:
- **Dense vectors** (bge-m3 via Ollama) for semantic similarity
- **Sparse vectors** (BM25 via fastembed) for keyword matching
- **Reciprocal Rank Fusion** to combine rankings
- **Optional reranking** via cross-encoder for precision

## Running MCP Locally

MCP runs locally for direct file system access. Backend services run in Docker.

```bash
# Start backend services
docker compose up -d

# Run MCP locally (stdio mode)

# Or via package
python -m memex
```

### Benefits of Local MCP

- **No permission issues** - Reads files directly as host user
- **Faster development** - No Docker rebuild for MCP changes
- **Better debugging** - Logs appear in terminal
- **Direct file access** - No volume mounts needed

## License

MIT
