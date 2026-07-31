# Package Restructure + Multi-Provider + Docker Consolidation

**Date:** 2026-07-31
**Status:** Approved

---

## 1. Package Structure: `memex/engine/`

Move all RAG logic under `memex/engine/`. Everything lives under one package.

```
memex/
├── engine/                          # Core RAG engine (was rag/)
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                # YAML config, single source of truth
│   │   └── pipeline.py              # RAGEngine orchestrator
│   ├── llm/                         # Multi-provider LLM + embedding
│   │   ├── __init__.py              # get_llm(), get_embedder()
│   │   ├── base.py                  # LLMProvider ABC, EmbedProvider ABC
│   │   ├── ollama.py                # Ollama via httpx
│   │   ├── openai.py                # OpenAI via httpx
│   │   ├── openrouter.py            # OpenRouter via httpx (OpenAI-compat)
│   │   ├── anthropic.py             # Anthropic via SDK
│   │   ├── groq.py                  # Groq via httpx (OpenAI-compat)
│   │   └── google.py                # Google via SDK
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py                # Docling + MarkItDown in-process
│   │   ├── splitter.py              # Hybrid/Recursive/Fixed chunking
│   │   ├── embedding.py             # Multi-provider embedding orchestrator
│   │   └── hashing.py               # SHA256 content/chunk/file hashing
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── search.py                # Hybrid + MMR + similarity
│   │   ├── reranker.py              # Cross-encoder (local) + Cohere (remote)
│   │   └── fusion.py                # Reciprocal Rank Fusion
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── answers.py               # Answer, Citation dataclasses
│   │   └── generator.py             # AnswerGenerator with context budget
│   ├── metadata/
│   │   ├── __init__.py
│   │   ├── schema.py                # Pydantic metadata models
│   │   └── extractor.py             # LLM-powered extraction with stored values
│   ├── sources/
│   │   ├── __init__.py              # Source ABC + lazy registry
│   │   ├── local.py                 # LocalSource
│   │   └── s3.py                    # S3Source (boto3 in-process)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── golden.py                # GoldenSet YAML/JSON loader
│   │   ├── metrics.py               # recall, precision, hit_rate, MRR, keyword_coverage
│   │   └── runner.py                # evaluate() + sweep() + delta formatting
│   └── utils/
│       ├── __init__.py
│       ├── retry.py                 # retry_call with exponential backoff
│       ├── cache.py                 # Redis cache (optional: in-memory dict fallback)
│       └── logging.py               # setup_logging, colored output
├── mcp/                             # MCP layer
│   ├── __init__.py
│   ├── server.py                    # FastMCP server + tool registration
│   ├── tools.py                     # Tool implementations (separated from server)
│   └── schemas.py                   # Pydantic input/output schemas
├── cli.py                           # Typer CLI: serve, ingest, sync, eval
└── __init__.py                      # Version + package metadata
```

**Import path changes:**

- `from rag.config import ...` → `from memex.engine.core.config import ...`
- `from rag.pipeline import RAGEngine` → `from memex.engine.core.pipeline import RAGEngine`
- `from rag.embedding import EmbeddingService` → `from memex.engine.ingestion.embedding import ...`

**Merged / eliminated:**

- `rag/services/` → `engine/utils/` + `engine/retrieval/` + `engine/metadata/`
- `rag/search/` → `engine/retrieval/`
- `rag/converters/` → deleted, not needed
- `rag/docling_client.py` → `engine/ingestion/loader.py`
- `memex/server.py` + `memex/schemas.py` → `memex/mcp/`

---

## 2. Multi-Provider LLM + Embedding

Two ABCs, factory functions, ~50 lines per provider.

```python
# engine/llm/base.py
class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, prompt: str, *, model: str | None = None) -> str: ...

class EmbedProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]: ...
```

**LLM providers:**

| Provider   | Transport                             | Config keys                        |
| ---------- | ------------------------------------- | ---------------------------------- |
| ollama     | httpx → `POST /api/chat`              | `base_url`, `model`, `temperature` |
| openai     | httpx → `POST /v1/chat/completions`   | `api_key`, `model`                 |
| openrouter | httpx → same endpoint, different host | `api_key`, `model`, `base_url`     |
| anthropic  | anthropic SDK                         | `api_key`, `model`                 |
| groq       | httpx → OpenAI-compat                 | `api_key`, `model`                 |
| google     | google-generativeai SDK               | `api_key`, `model`                 |

**Embedding providers:**

| Provider    | Transport                     | Default dims |
| ----------- | ----------------------------- | ------------ |
| ollama      | httpx → `POST /api/embed`     | 1024         |
| openai      | httpx → `POST /v1/embeddings` | 1536         |
| openrouter  | httpx → OpenAI-compat         | varies       |
| huggingface | sentence-transformers (local) | 384          |
| fastembed   | fastembed library (local)     | 384          |

**Factory:**

```python
# engine/llm/__init__.py
def get_llm(config) -> LLMProvider:    # reads llm.provider from config
def get_embedder(config) -> EmbedProvider:  # reads embedding.provider from config
```

**Config:**

```yaml
llm:
  provider: ollama # ollama | openai | openrouter | anthropic | groq | google
  model: qwen2.5:1.5b
  base_url: "http://localhost:11434"
  temperature: 0
  api_key: ${OPENAI_API_KEY}

embedding:
  provider: ollama # ollama | openai | openrouter | huggingface | fastembed
  model: qwen3-embedding:0.6b
  base_url: "http://localhost:11434"
  fallback_model: bge-m3
  dimensions: 1024
  api_key: ${OPENAI_API_KEY}
```

**Switching providers:** Only change `config.yaml`. Zero code changes.

---

## 3. Docker — 3 Containers (Tiered)

ML Services eliminated (runs in-process via pip). Redis optional (in-memory as default).

```
docker compose up    # 3 services
  ├── qdrant      :6333    # Vector DB
  ├── ollama      :11434   # Local LLM (skip with remote API)
  └── docling     :5001    # Doc conversion
```

**Eliminated:**

- `ml-services` (Custom Dockerfile, 4 stages) → `pip install fastembed sentence-transformers`
- `redis` → `functools.lru_cache` + in-memory dict (Docker Redis opt-in for persistence)
- `markitdown` → already removed
- `s3-service` → already removed

**Deployment modes:**

| Mode        | Containers                | Setup                                      |
| ----------- | ------------------------- | ------------------------------------------ |
| Quick start | qdrant                    | Use OpenRouter/OpenAI for LLM + embeddings |
| Local AI    | qdrant + ollama           | Ollama for both LLM + embeddings           |
| Full local  | qdrant + ollama + docling | GPU doc conversion                         |

**Dockerfile eliminated:** The 4-stage ML Services Dockerfile (~200 lines) is removed entirely. `setup.sh` simplified to `uv sync --extra local && docker compose up -d`.

---

## 4. Bug Fixes & Missing Features

From the RAGWire comparison audit:

| #   | Issue                                                             | Fix                                                                             |
| --- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 1   | `check_partial_ingest` checks 1 point, should scroll all          | Scroll all points, sum total_chunks                                             |
| 2   | `dedup_chunks` defined but never called in pipeline               | Call after chunking in `ingest_text()`                                          |
| 3   | Field type inference missing — all keyword                        | Infer integer/float from payload values                                         |
| 4   | Source extensions not normalized (`.PDF` ≠ `.pdf`)                | Normalize to lowercase with leading dot                                         |
| 5   | Filter values not lowercased (Qdrant case-sensitive)              | `_normalize_filters()` before Qdrant query                                      |
| 6   | Path form normalization missing (relative vs absolute, backslash) | `_path_forms()` for platform-agnostic matching                                  |
| 7   | MCP filter parsing doesn't accept JSON strings                    | `_parse_filters()` accepts dict or string                                       |
| 8   | Answer `__bool__` returns True even when refused                  | `__bool__` = `not self.refused`                                                 |
| 9   | Embedding dimension not validated at startup                      | Check `EMBED_MODEL` dims match `DENSE_DIM`                                      |
| 10  | Rollback on write failure missing                                 | Delete by `content_hash` in exception handler                                   |
| 11  | No `auto_filter` for LLM-powered filter extraction                | Add `auto_filter` to retriever config                                           |
| 12  | Lazy registry missing for plugin sources                          | `_LazyRegistry` with `register()` for custom sources                            |
| 13  | Context budget truncation (MIN_CHUNK_CHARS=200)                   | Add to `_pack_context()`                                                        |
| 14  | Refusal sentinel length guard                                     | `_is_refusal` checks `INSUFFICIENT_CONTEXT` containment + length < sentinel+120 |
| 15  | Citation parsing: invalid refs not stripped from text             | Strip invalid `[N]` markers, clean whitespace                                   |

---

## 5. Config Consolidation

Match the RAGWire config structure — top-level sections that map to components:

```yaml
version: "1.0"

# Document loading
loader:
  extensions: [".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md"]

# Chunking
splitter:
  strategy: hybrid
  size: 1024
  overlap: 128
  min_length: 30

# Ingestion
ingestion:
  workers: 3
  batch_size: 64
  retries: 2
  dedup_chunks: true
  replace_changed: true

# Embeddings (multi-provider)
embedding:
  provider: ollama
  model: qwen3-embedding:0.6b
  base_url: "http://localhost:11434"
  fallback_model: bge-m3
  dimensions: 1024

# LLM (multi-provider)
llm:
  provider: ollama
  model: qwen2.5:1.5b
  base_url: "http://localhost:11434"

# Vector store
vectorstore:
  url: "http://localhost:6333"
  collection: memex
  use_sparse: true

# Retrieval
retriever:
  search_type: hybrid
  top_k: 5
  auto_filter: false
  rerank:
    provider: cross_encoder
    model: Qwen/Qwen3-Reranker-0.6B
    fallback: BAAI/bge-reranker-base
    fetch_k: 20

# Query expansion
expansion:
  enabled: true
  hyde: true
  multi_query: true
  multi_query_count: 3
  rewrite: true

# Contextual retrieval
context:
  enabled: true
  strategy: summary

# Metadata extraction
metadata:
  enabled: true
  entity_extraction: true
  doc_classification: true
  topic_tagging: true
  language_detection: true

# Answer generation
generation:
  enabled: true
  max_context_chars: 12000

# Caching
cache:
  enabled: true
  redis_url: "redis://localhost:6379"
  ttl:
    embedding: 86400
    search: 3600

# Sources (for sync)
sources:
  - type: local
    name: docs
    path: /mnt/documents
    extensions: [".pdf", ".docx", ".md"]
    recursive: true

# Evaluation
evaluation:
  golden_set_path: null
  top_k: 5

# Logging
logging:
  level: INFO
  colored: true
```

---

## 6. Implementation Phases

**Phase 1: Restructure** — Move `rag/` → `memex/engine/`, `memex/*.py` → `memex/mcp/`. Update all imports. Tests must pass.

**Phase 2: Multi-provider** — Add `engine/llm/` providers. Wire into pipeline. Tests per provider.

**Phase 3: Docker consolidate** — Remove ML Services Dockerfile. Move BM25 + reranker in-process. Eliminate Redis Docker requirement (in-memory fallback). 5→3 containers.

**Phase 4: Bug fixes** — All 15 fixes from section 4. Tests for each.

**Phase 5: Config consolidate** — Redesign config.yaml to match new structure. Single source of truth.
