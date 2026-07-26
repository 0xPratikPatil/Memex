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
- **6 MCP Tools**: Ingest, search, list, stats, delete

## Quick Start

```bash
# 1. Clone and configure
cd mcp/rag
cp .env.example .env

# 2. Install dependencies
uv sync

# 3. Start backend services (Docker)
docker compose up -d

# 4. Pull embedding model into Ollama (first time only)
docker compose exec ollama ollama pull bge-m3

# 5. Verify services are healthy
curl http://localhost:6333/dashboard  # Qdrant
curl http://localhost:11434/api/tags  # Ollama
curl http://localhost:5001/health     # Docling

# 6. Run MCP server locally
uv run memex
```

## Development with uv

[uv](https://docs.astral.sh/uv/) is the recommended package manager for development.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Run MCP server
uv run memex

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/ memex/
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

## Configuration

All settings via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCLING_URL` | `http://docling:5001/v1/convert/source` | Docling Serve endpoint |
| `OLLAMA_EMBED_URL` | `http://ollama:11434/api/embeddings` | Ollama embedding endpoint |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant endpoint |
| `EMBED_MODEL` | `bge-m3` | Ollama embedding model |
| `CHUNK_SIZE` | `512` | Target chunk size (tokens) |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks (tokens) |
| `CHUNK_STRATEGY` | `recursive` | Chunking: `recursive` or `fixed` |
| `SEARCH_FUSION` | `rrf` | Fusion: `rrf` (reciprocal rank) or `weighted` |
| `RERANK_ENABLED` | `true` | Enable cross-encoder reranking |
| `MCP_TRANSPORT` | `http` | Transport: `http` or `stdio` |

## MCP Client Configuration

Add to your MCP client config (e.g., `opencode.json`):

```json
{
  "mcpServers": {
    "personal_rag": {
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

For local uv mode (recommended):

```json
{
  "mcpServers": {
    "personal_rag": {
      "command": "uv",
      "args": ["run", "memex"],
      "cwd": "/path/to/mcp/rag"
    }
  }
}
```

For local python mode:

```json
{
  "mcpServers": {
    "personal_rag": {
      "command": "python",
      "args": ["-m", "memex"],
      "cwd": "/path/to/mcp/rag"
    }
  }
}
```

## Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `qdrant` | 6333, 6334 | Vector database |
| `ollama` | 11434 | Embedding server |
| `docling` | 5001 | Document conversion |
| `redis` | 6379 | Caching layer |

**Note:** MCP server runs locally (not in Docker) for direct file system access.

## Project Structure

```
memex/
├── src/                    # Core MCP server code
│   ├── __init__.py        # Package exports
│   ├── config.py          # Central configuration (env vars)
│   ├── server.py          # MCP server tool definitions
│   ├── pipeline.py        # RAG engine: embeddings, Qdrant, search
│   ├── docling_client.py  # Docling document conversion client
│   ├── cli/               # CLI entry point
│   │   └── __init__.py    # Main entry point (stdio/HTTP mode)
│   ├── models/            # Data models (Pydantic/dataclass)
│   ├── services/          # Business logic services
│   │   ├── cache.py       # Redis caching layer
│   │   ├── contextual_retrieval.py  # Context prefixes for chunks
│   │   ├── evaluation.py  # RAGAS evaluation framework
│   │   ├── metadata_extractor.py  # Entity extraction
│   │   └── query_expansion.py  # HyDE + Multi-Query
│   └── utils/             # Shared utilities
├── memex/              # Local MCP package (runs outside Docker)
│   ├── __init__.py        # Package version
│   └── __main__.py        # Entry point for local execution
├── tests/
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── fixtures/          # Test fixtures and sample data
├── scripts/               # Utility scripts
│   └── evaluate.py        # Evaluation CLI tool
├── docs/
│   └── superpowers/specs/ # Design specifications
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
