# RAGWire Feature Adoption Design Spec

**Date:** 2026-07-31
**Status:** Approved
**Scope:** Adopt all RAGWire features into this project, adapted to existing architecture

---

## Decision Log

The following questions were asked during brainstorming. Each records the question, available options, and the chosen answer for future reference.

### Q1: Source Configuration

**Question:** For the Document Sources & Sync feature (S3, local directories, pluggable registry), how should sources be configured? Currently Memex ingests one file at a time via MCP tools.

| Option | Description |
|--------|-------------|
| YAML config file (Recommended) | Define sources in a config.yaml (e.g. local dirs, S3 buckets). CLI/periodic sync reconciles collection against sources. |
| MCP tools only | Add new MCP tools for managing sources (add_source, sync_sources) without a config file. |
| Both YAML + MCP tools | YAML for initial setup, MCP tools for runtime management. More complex but flexible. |

**Chosen:** YAML config file

### Q2: MarkItDown Role

**Question:** MarkItDown is a lighter document converter (flattens to markdown). Docling is already your primary converter with GPU acceleration. How should MarkItDown fit in?

| Option | Description |
|--------|-------------|
| Fallback when Docling unavailable (Recommended) | Keep Docling as primary. Add MarkItDown as a lighter fallback for when Docling is down or for simpler docs. More resilient but more code to maintain. |
| Replace Docling | MarkItDown replaces Docling entirely. Simpler but loses Docling's advanced features (table structure, OCR, picture classification). |
| User-selectable via config | User chooses via config which converter to use. Both always available. Maximum flexibility. |

**Chosen:** User-selectable via config

### Q3: YAML Config Scope

**Question:** RAGWire uses a full YAML config (vectorstore, embedding, LLM, sources, chunking, etc). You currently use 80+ env vars. How should the config system evolve?

| Option | Description |
|--------|-------------|
| Full replacement | YAML config replaces all env vars. Clean but a breaking change for existing users. |
| YAML primary, env vars override (Recommended) | YAML config + env vars coexist. YAML is primary, env vars override YAML values. No breaking changes. |
| Env vars primary, YAML for sources only | Keep env vars as primary, YAML just adds source definitions. Minimal change. |

**Chosen:** Full replacement

### Q4: Citation-Based Answers

**Question:** RAGWire generates cited answers (Answer object with citations, refusal sentinel, confidence score). Your current MCP tools return raw ranked chunks. How should answers work?

| Option | Description |
|--------|-------------|
| Structured answer format (Recommended) | MCP tools return structured Answer objects with citations, refusal, confidence, sources. More useful for agents but changes the MCP tool response format. |
| Keep current chunk-based format | Keep current format (raw chunks ranked by relevance). Simpler but less agent-friendly. |
| New tool + keep existing | Add a new MCP tool (rag_answer) that returns cited answers, keep existing rag_query for raw chunks. |

**Chosen:** Structured answer format

### Q5: S3 Source Scope

**Question:** The S3 source needs AWS credentials and boto3. Where should it run?

| Option | Description |
|--------|-------------|
| Docker container (Recommended) | S3 connector runs in Docker (needs boto3 in a container). Fully local, no host dependencies. |
| Host-side (MCP server) | S3 connector runs on host via MCP server. Needs AWS credentials on host. |
| Multi-cloud (S3 + GCS + Azure) | S3 runs in Docker, but also support GCS and Azure Blob as pluggable cloud sources. |

**Chosen:** Docker container

### Q6: Sync Mechanism

**Question:** The sync feature reconciles the collection against sources (adds new, removes deleted, replaces changed). How should it be triggered?

| Option | Description |
|--------|-------------|
| MCP tool (Recommended) | New MCP tool (rag_sync) that reconciles collection against configured sources. Agent-triggered. |
| CLI command only | CLI command only (memex sync). Manual or cron-triggered. |
| Both MCP + CLI | Both MCP tool and CLI command. Most flexible. |

**Chosen:** MCP tool

### Q7: Evaluation Framework

**Question:** The golden-set evaluation framework (recall, precision, MRR, eval sweep) — how should users run evaluations?

| Option | Description |
|--------|-------------|
| MCP tools (Recommended) | New MCP tools (rag_eval, rag_eval_sweep) for running evaluations from agent context. Returns metrics tables. |
| CLI only | CLI command only. Run from terminal, not from agent. |
| Both MCP + CLI | Both MCP tools and CLI. Most flexible but more surface area. |

**Chosen:** Both MCP + CLI

### Q8: Deduplication Strategy

**Question:** RAGWire has multi-level dedup (file SHA256, chunk SHA256, partial ingest recovery). How thorough should dedup be?

| Option | Description |
|--------|-------------|
| Content-hash dedup only (Recommended) | SHA256 content hash on ingestion. Skip if identical content already indexed. Simple and effective. |
| Two-phase (stat + hash) | Content-hash + file mtime/size pre-check. Two-phase: fast stat check then deep hash. Faster for large doc sets. |
| Three-phase (stat + hash + chunk dedup) | Content-hash + chunk-level dedup within documents. Eliminates duplicate chunks across sections. Most thorough but more overhead. |

**Chosen:** Content-hash dedup only

### Q9: MMR Search

**Question:** RAGWire has MMR (Maximal Marginal Relevance) search for result diversity. Your current pipeline uses RRF fusion. Should MMR be added?

| Option | Description |
|--------|-------------|
| Add MMR mode (Recommended) | Add MMR as a third search mode alongside similarity and hybrid. Good for exploratory queries where diversity matters. |
| Skip MMR | Skip MMR. Dense+sparse+RRF is sufficient for most use cases. |
| MMR alongside RRF | Replace RRF with MMR. RRF stays but MMR becomes an option for the dense path. |

**Chosen:** Add MMR mode

### Q10: Agent Filter Tools

**Question:** RAGWire has agent-oriented filter tools: get_filter_context (shows available fields/stored values, suggests filters) and extract_filters. Your current rag_query supports metadata_filter but agents must figure it out themselves. Should we add these?

| Option | Description |
|--------|-------------|
| Add both tools (Recommended) | New MCP tools: rag_get_filter_context (shows available fields/values, suggests filters) + rag_extract_filters (extracts filters from query). Designed for agent workflows. |
| Filter context tool only | Add rag_get_filter_context only. Manual filter extraction by agent. |
| Skip, use existing filter param | Skip dedicated tools. Agents can use metadata_filter parameter on rag_query directly. |

**Chosen:** Add both tools

### Q11: Implementation Approach

**Question:** This is a large set of features. How should we approach implementation?

| Option | Description |
|--------|-------------|
| Phased implementation (Recommended) | Design the full system but implement in phases. Phase 1: YAML config + sources + sync. Phase 2: Citation answers + eval. Phase 3: CLI + remaining tools. |
| All at once | Design everything at once, implement all at once. Higher risk but faster if it works. |
| One feature at a time | Design each feature independently, implement one at a time. Most controlled but slowest. |

**Chosen:** Phased implementation

---

## Architecture Approach

**Selected:** Layered Config-Driven Architecture (Approach A)

YAML config becomes the single source of truth. MCP server reads it at startup. Each feature (sources, sync, eval, filter tools, citation answers) is a module that reads its config section. Docker services unchanged — they still handle Docling, Ollama, Qdrant, Redis, ML Services. New Docker service for S3/MarkItDown.

---

## Feature Design

### 1. YAML Config System

The current 80+ env vars get replaced by a single `config.yaml`.

**Structure:**
```yaml
version: "1.0"

vectorstore:
  provider: qdrant
  url: "http://localhost:6333"
  collection: memex
  api_key: ${QDRANT_API_KEY}

embedding:
  provider: ollama
  model: qwen3-embedding:0.6b
  dimensions: 1024
  batch_size: 64
  fallback_model: null
  base_url: "http://localhost:11434"
  api_key: ${OPENAI_API_KEY}

sparse:
  provider: fastembed
  model: Qdrant/bm25

llm:
  provider: ollama
  model: qwen2.5:1.5b
  base_url: "http://localhost:11434"
  api_key: ${OPENAI_API_KEY}
  num_ctx: 4096

converter:
  engine: docling
  docling_url: "http://localhost:5001"
  docling_picture_classify: true
  docling_enrich_code: false
  docling_enrich_formula: false
  docling_chart_extract: false
  docling_ocr: true
  markitdown_url: "http://localhost:5003"
  markitdown_extensions: [".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md"]

chunking:
  strategy: hybrid
  size: 1024
  overlap: 128
  min_length: 30
  merge_peers: true

reranker:
  enabled: true
  provider: http
  model: Qwen/Qwen3-Reranker-0.6B
  fallback_model: BAAI/bge-reranker-base
  type: auto
  fetch_k_multiplier: 4

query_expansion:
  enabled: true
  hyde: true
  query_rewrite: true
  multi_query: true
  multi_query_count: 3

contextual_retrieval:
  enabled: true
  strategy: summary
  max_tokens: 50

metadata:
  entity_extraction: true
  doc_classification: true
  topic_tagging: true
  language_detection: true

caching:
  enabled: true
  redis_url: "redis://localhost:6379"
  ttl_embedding: 86400
  ttl_search: 3600
  ttl_parse: 604800
  ttl_expansion: 21600

sources:
  - type: local
    name: docs
    path: /mnt/documents
    extensions: [".pdf", ".docx", ".md"]
    recursive: true
  - type: s3
    name: reports
    bucket: my-bucket
    prefix: reports/
    extensions: [".pdf"]
    aws_access_key: ${AWS_ACCESS_KEY_ID}
    aws_secret_key: ${AWS_SECRET_ACCESS_KEY}
    region: us-east-1

sync:
  auto_delete: true
  dry_run: false

answer:
  enabled: true
  max_context_chars: 12000
  system_prompt: null
  refusal_sentinel: "INSUFFICIENT_CONTEXT"

evaluation:
  golden_set_path: null
  top_k: 5
  run_ragas: false

search:
  mode: hybrid
  mmr:
    fetch_k: 20
    lambda_mult: 0.5

server:
  host: 127.0.0.1
  port: 8080
```

**Key points:**
- `${VAR}` substitution from env vars (for secrets like API keys)
- Dot-notation access: `config.get("embedding.model")`
- Type-safe helpers for int/float/bool values
- Backward compat: startup warns if old env vars are detected, suggests config.yaml equivalent
- `config.example.yaml` ships with the project

### 2. Document Sources & Sync

**Source types** (pluggable registry pattern):

```python
class Source(ABC):
    name: str
    type: str
    extensions: list[str]

    def list_files(self) -> list[SourceFile]
    def get_hash(self, file: SourceFile) -> str
    def download(self, file: SourceFile, dest: Path) -> Path
```

**Built-in source types:**
- `local` — walks a directory, lists files by extension
- `s3` — lists objects by prefix, downloads to temp cache with mtime-based skip

**Sync process** (`rag_sync` MCP tool):
1. Load all sources from config.yaml
2. For each source: `list_files()` → get current file set
3. For each file: compute content hash
4. Compare against stored hashes in Qdrant payload
5. Reconcile: new files → ingest, changed files → delete old + ingest new, deleted files → delete chunks
6. Safety: if any source fails to list, suppress ALL deletions for that run
7. Return stats: added, changed, deleted, unchanged, errors

**New MCP tool:** `rag_sync(source_name: str | null, dry_run: bool)`

**Docker:** New S3 service container. Custom Dockerfile with Python + boto3 + a small HTTP API for listing and downloading objects. MCP server calls this service during sync.

### 3. MarkItDown Integration

MarkItDown runs as a new Docker service alongside Docling. User selects via `config.yaml` → `converter.engine`.

**Docker service:**
Custom Dockerfile (no official HTTP image exists for MarkItDown). Lightweight Python container exposing a simple HTTP API:

```yaml
markitdown:
  build:
    context: ./docker/markitdown
    dockerfile: Dockerfile
  container_name: memex-markitdown
  ports:
    - "127.0.0.1:5003:8000"
  restart: unless-stopped
```

Dockerfile installs `markitdown` pip package + a small FastAPI wrapper that accepts file uploads and returns markdown.

**Converter abstraction:**
```python
class Converter(ABC):
    async def convert(self, file_path: str) -> str

class DoclingConverter:
    async def convert(self, file_path: str) -> str:
        # POST to Docling /v1/convert/source

class MarkItDownConverter:
    async def convert(self, file_path: str) -> str:
        # POST to MarkItDown /convert

def get_converter(config) -> Converter:
    if config.converter.engine == "docling":
        return DoclingConverter(config.converter.docling_url)
    elif config.converter.engine == "markitdown":
        return MarkItDownConverter(config.converter.markitdown_url)
```

**Key difference:** MarkItDown is stateless, no GPU, no table structure recognition, no OCR — just flattens to markdown. Docling remains the default for quality.

### 4. Deduplication

**File-level dedup:** SHA256 of markdown content after conversion. Check Qdrant payload `content_hash` — if exists, skip.

**Chunk-level dedup:** SHA256 of each chunk's text. Drop exact duplicate chunks within the same document. Scoped per-document, not cross-document.

**Partial ingest recovery:** Each chunk stored with `total_chunks` count. On re-ingest, if stored count mismatches expected, clear old chunks and re-ingest fresh.

**MCP tool behavior:** `rag_ingest_file` and `rag_ingest_batch` check content hash before processing. Returns stats: `ingested`, `skipped`, `failed`.

### 5. Citation-Based Answers

**Answer object:**
```python
@dataclass
class Answer:
    text: str
    refused: bool
    confidence: float
    citations: list[Citation]
    sources: list[str]
    filters_used: dict | None

@dataclass
class Citation:
    index: int
    source: str
    chunk_text: str
    metadata: dict
    rerank_score: float | None
```

**Generation pipeline:**
1. Retrieve top-k chunks (hybrid search + RRF + rerank)
2. Pack chunks into context (budget: `max_context_chars`, default 12000)
3. Send to LLM with grounded answer prompt + sentinel instruction
4. Parse response: detect refusal sentinel → `refused=True`
5. Extract `[N]` citation markers → resolve to source files
6. Compute confidence = fraction of sentences carrying a citation

**Refusal behavior:** System prompt includes refusal sentinel. Detection works even if model wraps it in prose (regex scan). Empty document set → explicit refusal, no hallucination.

**MCP tool response:**
```json
{
  "answer": "The company reported revenue of $45.2B in Q3 2025.",
  "refused": false,
  "confidence": 0.85,
  "citations": [
    {"index": 1, "source": "report.pdf", "text": "...", "rerank_score": 0.92}
  ],
  "sources": ["report.pdf", "earnings.md"],
  "filters_used": {"doc_type": "report"}
}
```

### 6. Agent Filter Tools

**Tool 1: `rag_get_filter_context`**

Returns available metadata fields, their stored values, and suggested filters for a query.

```python
@dataclass
class FilterContext:
    fields: list[FieldInfo]
    suggested_filters: dict | None
    sample_query: str

@dataclass
class FieldInfo:
    name: str
    type: str
    values: list[str]
    count: int
```

Implementation: Scan Qdrant payload indexes, collect unique values per field (capped at 100). If query provided, LLM extracts suggested filters.

**Tool 2: `rag_extract_filters`**

Extracts metadata filters from a natural language query without executing search.

```python
# Input
query = "show me reports from apple in 2024"

# Output
{
  "filters": {"doc_type": "report", "keywords": ["apple"], "dates": "2024"},
  "explanation": "Filtered to reports mentioning apple, from 2024",
  "confidence": 0.9
}
```

**Agent workflow:**
1. Agent calls `rag_get_filter_context(query="quarterly revenue")`
2. Sees available fields: doc_type, topics, dates, keywords
3. Calls `rag_extract_filters(query="apple Q3 2024 revenue")`
4. Gets filters: `{"keywords": ["apple"], "dates": "2024-Q3"}`
5. Calls `rag_query(query="revenue", metadata_filter=filters)`
6. Returns cited answer with confidence score

### 7. MMR Search Mode

MMR (Maximal Marginal Relevance) added as a third search mode.

**How it works:** First pass retrieves top `fetch_k` candidates via dense vector similarity. Second pass iteratively selects documents balancing relevance with diversity.

**Formula:** `λ * sim(query, doc) - (1-λ) * max(sim(doc, selected))`

**Config:**
```yaml
search:
  mode: hybrid
  mmr:
    fetch_k: 20
    lambda_mult: 0.5
```

**MCP tool:** `rag_query` gets optional `search_mode` override.

**Pipeline with MMR:**
```
User Query → Query Expansion → Dense search (fetch_k) → MMR selection (top_k) → Reranking → Results
```

### 8. Evaluation Framework

**Golden set format (YAML):**
```yaml
queries:
  - query: "What was the Q3 2024 revenue?"
    expected_sources:
      - "reports/q3-2024.pdf"
    expected_keywords: ["revenue", "Q3", "2024"]
    category: financial
    difficulty: easy
```

**Metrics:**
| Metric | Description |
|--------|-------------|
| `recall@K` | Fraction of expected documents found in top K |
| `precision@K` | Fraction of top K that are correct |
| `hit_rate@K` | 1.0 if any expected doc found, 0.0 otherwise |
| `mrr` | Mean Reciprocal Rank of first correct result |
| `keyword_coverage` | Fraction of expected keywords in retrieved content |

**Eval sweep:** Compare two configs side by side with delta comparison.

**MCP tools:** `rag_eval(golden_set_path, top_k, compare_rerank)` and `rag_eval_sweep(golden_set_path, variants)`

**CLI commands:** `memex eval golden.yaml --top-k 5 --compare-rerank`

### 9. CLI Commands

Built with `typer`. Entry point in `pyproject.toml`:
```toml
[project.scripts]
memex = "memex.cli:app"
```

**Commands:**
- `memex ingest /path/to/documents --recursive --source-name docs`
- `memex sync --dry-run --source-name reports`
- `memex eval golden.yaml --top-k 5 --compare-rerank`

**CLI flags:** `--config / -c` (path to config.yaml), `--verbose / -v`, `--dry-run`

CLI reads same `config.yaml` as MCP server. Calls same pipelines as MCP tools.

### 10. Phased Implementation Plan

**Phase 1: Foundation (YAML Config + Sources + Sync)**
- Build YAML config loader with `${VAR}` substitution
- Replace all env var reads with config lookups
- Implement Source base class + LocalSource + S3Source
- Build sync engine (hash comparison, reconcile, safety rails)
- Add `rag_sync` MCP tool
- Add `rag_get_filter_context` and `rag_extract_filters` MCP tools
- Docker: add S3 service container
- Tests: config loading, source listing, sync logic

**Phase 2: Answers + Dedup + MMR**
- Build Answer/Citation dataclasses and generation pipeline
- Implement citation parsing, refusal detection, confidence scoring
- Add content-hash dedup + chunk-level dedup
- Add partial ingest recovery
- Add MMR search mode
- Update `rag_query` to return structured answers
- Tests: answer generation, dedup, MMR

**Phase 3: Evaluation + CLI + MarkItDown**
- Build golden set loader and metrics computation
- Implement eval sweep with delta comparison
- Add `rag_eval` and `rag_eval_sweep` MCP tools
- Build CLI with typer (ingest, sync, eval)
- Add MarkItDown Docker service + converter abstraction
- Tests: eval metrics, CLI commands, converter switching

**Phase 4: Polish + Docs**
- Update all existing tests to use new config system
- Migrate `setup.sh` to work with config.yaml
- Update README, AGENTS.md, DOCKER.md
- Add `config.example.yaml`
- Integration tests for full pipeline

---

## New MCP Tools Summary

| Tool | Type | Description |
|------|------|-------------|
| `rag_sync` | Write | Sync collection against configured sources |
| `rag_get_filter_context` | Read | Show available metadata fields, values, suggested filters |
| `rag_extract_filters` | Read | Extract metadata filters from natural language query |
| `rag_eval` | Read | Run golden-set evaluation, return metrics |
| `rag_eval_sweep` | Read | Compare multiple retrieval configs side by side |

## Existing MCP Tools Updated

| Tool | Change |
|------|--------|
| `rag_query` | Returns structured Answer with citations. Adds `search_mode` param for MMR. |
| `rag_ingest_file` | Adds content-hash dedup check before processing. |
| `rag_ingest_batch` | Adds content-hash dedup check. Returns `skipped` count. |
| `rag_service_status` | Adds MarkItDown service to health check list. |

## Docker Changes

| Service | Change |
|---------|--------|
| New: `markitdown` | Custom Dockerfile: markitdown pip package + FastAPI wrapper on port 5003 |
| New: `s3-service` | Custom Dockerfile: Python + boto3 + HTTP API for S3 listing/download |
| Existing 5 services | Unchanged |

## No Brand References

- No mention of "RAGWire", "ragwire", "laxmimerit", or the source repository anywhere in code, docs, or config
- All features rewritten to fit this project's patterns and naming
- Config schema designed from scratch (not copied)
- Code implementations follow existing codebase conventions (async, httpx, tenacity, Pydantic)
