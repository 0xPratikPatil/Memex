# Implementation Plan: Feature Adoption

**Spec:** `docs/superpowers/specs/2026-07-31-feature-adoption-design.md`
**Date:** 2026-07-31

---

## Phase 1: Foundation (YAML Config + Sources + Sync)

### Task 1.1: YAML Config Loader

**Files to create:**
- `rag/yaml_config.py` — Config loader with `${VAR}` substitution, dot-notation access, type-safe helpers

**Files to modify:**
- `rag/config.py` — Refactor to read from YAML config instead of env vars directly. Keep `_env()` helpers as fallback for secrets.

**Implementation:**
```python
# rag/yaml_config.py
class YamlConfig:
    def __init__(self, path: str = "config.yaml"):
        self._data = self._load(path)

    def _load(self, path: str) -> dict:
        # Read YAML, resolve ${VAR} substitutions from env

    def get(self, dotpath: str, default=None):
        # "embedding.model" → self._data["embedding"]["model"]

    def get_str(self, dotpath: str, default: str) -> str: ...
    def get_int(self, dotpath: str, default: int) -> int: ...
    def get_float(self, dotpath: str, default: float) -> float: ...
    def get_bool(self, dotpath: str, default: bool) -> bool: ...
    def get_list(self, dotpath: str, default: list) -> list: ...
```

**Steps:**
1. Create `rag/yaml_config.py` with `YamlConfig` class
2. Implement `${VAR}` substitution with recursive resolution
3. Implement dot-notation access with type-safe helpers
4. Update `rag/config.py` to instantiate `YamlConfig` and expose same module-level variables
5. Add backward compat: detect old env vars, log warnings with config.yaml equivalents
6. Create `config.example.yaml` with all defaults
7. Write unit tests for config loading, var substitution, dot-notation, type helpers

### Task 1.2: Document Source Abstraction

**Files to create:**
- `rag/sources/__init__.py` — Source base class, SourceFile dataclass, registry
- `rag/sources/local.py` — LocalSource implementation
- `rag/sources/s3.py` — S3Source implementation

**Implementation:**
```python
# rag/sources/__init__.py
@dataclass
class SourceFile:
    name: str
    path: str  # full path or S3 key
    size: int
    modified_at: float  # timestamp


class Source(ABC):
    name: str
    type: str
    extensions: list[str]

    @abstractmethod
    def list_files(self) -> list[SourceFile]: ...

    @abstractmethod
    def get_content_hash(self, file: SourceFile) -> str: ...

    @abstractmethod
    def download(self, file: SourceFile, dest: Path) -> Path: ...


# Registry
_SOURCES: dict[str, type[Source]] = {}


def register_source(cls: type[Source]) -> type[Source]:
    _SOURCES[cls.type] = cls
    return cls


def get_source(type_name: str, config: dict) -> Source:
    return _SOURCES[type_name](**config)
```

**Steps:**
1. Create `rag/sources/__init__.py` with `SourceFile`, `Source` ABC, registry
2. Create `rag/sources/local.py` — `LocalSource`: walks directory, filters by extension, computes SHA256
3. Create `rag/sources/s3.py` — `S3Source`: lists objects by prefix, downloads to temp cache
4. Write unit tests for both source types

### Task 1.3: Sync Engine

**Files to create:**
- `rag/sync.py` — Sync engine: reconcile sources against collection

**Implementation:**
```python
@dataclass
class SyncStats:
    added: int
    changed: int
    deleted: int
    unchanged: int
    errors: list[str]

async def sync(
    config: YamlConfig,
    source_name: str | None = None,
    dry_run: bool = False,
) -> SyncStats:
    # 1. Load sources from config
    # 2. For each source: list_files()
    # 3. For each file: compute content hash
    # 4. Query Qdrant for stored hashes
    # 5. Reconcile: new → ingest, changed → delete+ingest, deleted → delete
    # 6. Safety: if any source fails, suppress deletions
    # 7. Return stats
```

**Steps:**
1. Create `rag/sync.py` with `SyncStats` and `sync()` function
2. Implement hash comparison against Qdrant payload
3. Implement reconcile logic (new, changed, deleted)
4. Implement safety rails (suppress deletions on source failure)
5. Write unit tests with mock Qdrant and sources

### Task 1.4: MCP Sync Tool

**Files to modify:**
- `memex/server.py` — Add `rag_sync` tool

**Steps:**
1. Add `rag_sync` tool with `source_name: str | null` and `dry_run: bool` params
2. Call `rag.sync.sync()` with config
3. Return `SyncStats` as JSON
4. Write integration test

### Task 1.5: MCP Filter Tools

**Files to create:**
- `rag/filter_tools.py` — Filter context and filter extraction logic

**Files to modify:**
- `memex/server.py` — Add `rag_get_filter_context` and `rag_extract_filters` tools

**Steps:**
1. Create `rag/filter_tools.py` with `get_filter_context()` and `extract_filters()`
2. Implement Qdrant payload field scanning for available fields/values
3. Implement LLM-based filter extraction from natural language
4. Add both tools to MCP server
5. Write unit tests

### Task 1.6: S3 Docker Service

**Files to create:**
- `docker/s3-service/Dockerfile` — Python + boto3 + FastAPI wrapper
- `docker/s3-service/server.py` — HTTP API for S3 listing/download
- `docker/s3-service/requirements.txt`

**Steps:**
1. Create Dockerfile with Python 3.12, boto3, fastapi, uvicorn
2. Create FastAPI app with `/list` and `/download` endpoints
3. Add to `docker-compose.yml` with health check, resource limits, logging
4. Test locally

### Task 1.7: Phase 1 Tests

**Files to create/modify:**
- `tests/unit/test_yaml_config.py`
- `tests/unit/test_sources.py`
- `tests/unit/test_sync.py`
- `tests/unit/test_filter_tools.py`
- `tests/integration/test_sync_integration.py`

**Steps:**
1. Write unit tests for YAML config (loading, substitution, dot-notation)
2. Write unit tests for sources (local file listing, hash computation)
3. Write unit tests for sync engine (mock Qdrant, mock sources)
4. Write unit tests for filter tools (mock Qdrant, mock LLM)
5. Write integration test for sync with real Qdrant

---

## Phase 2: Answers + Dedup + MMR

### Task 2.1: Answer/Citation Dataclasses

**Files to create:**
- `rag/answer.py` — Answer, Citation dataclasses + generation pipeline

**Implementation:**
```python
@dataclass
class Citation:
    index: int
    source: str
    chunk_text: str
    metadata: dict
    rerank_score: float | None

@dataclass
class Answer:
    text: str
    refused: bool
    confidence: float
    citations: list[Citation]
    sources: list[str]
    filters_used: dict | None

    def formatted(self) -> str:
        # Answer text with numbered source list

async def generate_answer(
    query: str,
    chunks: list,
    config: YamlConfig,
    filters: dict | None = None,
) -> Answer:
    # 1. Pack chunks into context budget
    # 2. Send to LLM with grounded prompt + sentinel
    # 3. Parse: detect refusal, extract citations
    # 4. Compute confidence
    # 5. Return Answer
```

**Steps:**
1. Create `rag/answer.py` with dataclasses
2. Implement context budget management (max_context_chars)
3. Implement system prompt with refusal sentinel
4. Implement citation parsing (extract [N] markers, resolve to sources)
5. Implement confidence scoring (fraction of sentences with citations)
6. Write unit tests

### Task 2.2: Content-Hash Dedup

**Files to modify:**
- `rag/ingestion.py` — Add content hash check before processing
- `rag/pipeline.py` — Store content_hash in Qdrant payload

**Steps:**
1. Compute SHA256 of markdown content after conversion
2. Query Qdrant for existing `content_hash` — skip if exists
3. Store `content_hash` in Qdrant payload on ingest
4. Add chunk-level dedup: SHA256 per chunk text, drop duplicates within document
5. Write unit tests

### Task 2.3: Partial Ingest Recovery

**Files to modify:**
- `rag/ingestion.py` — Check total_chunks count on re-ingest

**Steps:**
1. Store `total_chunks` in Qdrant payload
2. On re-ingest, count stored chunks for the file
3. If count mismatches expected, clear old chunks and re-ingest
4. Write unit tests

### Task 2.4: MMR Search Mode

**Files to create:**
- `rag/search/mmr.py` — MMR selection algorithm

**Files to modify:**
- `rag/pipeline.py` — Add MMR as search mode option
- `memex/server.py` — Add `search_mode` param to `rag_query`

**Steps:**
1. Create `rag/search/mmr.py` with MMR algorithm
2. Implement: fetch_k dense candidates → iterative MMR selection → top_k
3. Add `search_mode` parameter to `rag_query` tool
4. Wire MMR into search pipeline alongside similarity and hybrid
5. Write unit tests

### Task 2.5: Update rag_query for Structured Answers

**Files to modify:**
- `memex/server.py` — Update `rag_query` to return Answer object

**Steps:**
1. After retrieval + reranking, call `generate_answer()`
2. Return structured Answer JSON with citations, confidence, sources
3. Support `response_format` toggle (markdown vs json)
4. Write integration test

### Task 2.6: Phase 2 Tests

**Files to create/modify:**
- `tests/unit/test_answer.py`
- `tests/unit/test_dedup.py`
- `tests/unit/test_mmr.py`
- `tests/integration/test_answer_integration.py`

---

## Phase 3: Evaluation + CLI + MarkItDown

### Task 3.1: Golden Set Loader

**Files to create:**
- `rag/evaluation/golden.py` — GoldenSet, GoldenQuery dataclasses + YAML loader

**Steps:**
1. Create `rag/evaluation/golden.py`
2. Implement YAML/JSON loading
3. Implement match modes (basename, exact, contains)
4. Write unit tests

### Task 3.2: Metrics Computation

**Files to create:**
- `rag/evaluation/metrics.py` — recall, precision, hit_rate, MRR, keyword_coverage

**Steps:**
1. Create `rag/evaluation/metrics.py`
2. Implement all 5 metrics
3. Implement `EvalResult` with aggregate and per-query scores
4. Write unit tests

### Task 3.3: Eval Sweep

**Files to create:**
- `rag/evaluation/sweep.py` — Compare multiple configs side by side

**Steps:**
1. Implement `sweep()` function: run eval with different configs
2. Compute delta and delta_pct for all metrics
3. Format as plain-text table
4. Write unit tests

### Task 3.4: MCP Eval Tools

**Files to modify:**
- `memex/server.py` — Add `rag_eval` and `rag_eval_sweep` tools

**Steps:**
1. Add `rag_eval` tool: load golden set, run retrieval, compute metrics
2. Add `rag_eval_sweep` tool: run multiple configs, return delta table
3. Write integration tests

### Task 3.5: CLI Commands

**Files to modify:**
- `memex/cli.py` — Add typer app with ingest, sync, eval commands
- `pyproject.toml` — Add `[project.scripts]` entry point

**Steps:**
1. Refactor `memex/cli.py` to use typer
2. Add `memex ingest` command — calls ingestion pipeline
3. Add `memex sync` command — calls sync engine
4. Add `memex eval` command — calls evaluation framework
5. Add `--config`, `--verbose`, `--dry-run` flags
6. Add `[project.scripts]` to pyproject.toml
7. Write CLI tests

### Task 3.6: MarkItDown Integration

**Files to create:**
- `docker/markitdown/Dockerfile` — markitdown pip package + FastAPI wrapper
- `docker/markitdown/server.py` — HTTP API for document conversion
- `docker/markitdown/requirements.txt`
- `rag/converters/__init__.py` — Converter ABC
- `rag/converters/docling.py` — DoclingConverter (existing logic extracted)
- `rag/converters/markitdown.py` — MarkItDownConverter

**Files to modify:**
- `docker-compose.yml` — Add markitdown service
- `rag/pipeline.py` — Use converter abstraction

**Steps:**
1. Create `rag/converters/__init__.py` with `Converter` ABC and `get_converter()` factory
2. Extract existing Docling logic into `rag/converters/docling.py`
3. Create `rag/converters/markitdown.py` calling MarkItDown HTTP API
4. Create MarkItDown Dockerfile + FastAPI wrapper
5. Add markitdown service to docker-compose.yml
6. Update pipeline to use `get_converter(config)`
7. Update `rag_service_status` to include MarkItDown
8. Write unit tests for converter switching

### Task 3.7: Phase 3 Tests

**Files to create/modify:**
- `tests/unit/test_golden_set.py`
- `tests/unit/test_metrics.py`
- `tests/unit/test_sweep.py`
- `tests/unit/test_cli.py`
- `tests/unit/test_converters.py`
- `tests/integration/test_eval_integration.py`

---

## Phase 4: Polish + Docs

### Task 4.1: Migrate Existing Tests

**Steps:**
1. Update all existing unit tests to use `YamlConfig` instead of env vars
2. Update integration tests to use config.yaml
3. Ensure all 310+ tests pass with new config system

### Task 4.2: Migrate setup.sh

**Steps:**
1. Update `setup.sh` to generate `config.yaml` from `config.example.yaml`
2. Keep `.env` for secrets only (API keys)
3. Update Docker health checks to work with new config

### Task 4.3: Documentation

**Files to modify:**
- `README.md` — Update with config.yaml usage, new MCP tools, CLI commands
- `AGENTS.md` — Update with new features, config, MCP tools
- `DOCKER.md` — Add MarkItDown and S3 services
- `CONTRIBUTING.md` — Update development setup

**Steps:**
1. Update README with config.yaml examples
2. Document all 13 MCP tools (8 existing + 5 new)
3. Document CLI commands
4. Document Docker services (7 total)
5. Add config.example.yaml with comments

### Task 4.4: Integration Tests

**Steps:**
1. Write end-to-end test: config → ingest → sync → query → answer → eval
2. Test with both Docling and MarkItDown converters
3. Test S3 source with localstack or mock
4. Verify all Docker services start and pass health checks

### Task 4.5: Clean Up

**Steps:**
1. Remove old env var references from code (replaced by config.yaml)
2. Update `.env.example` to only contain secrets
3. Run full test suite: `make test`
4. Run linter: `make lint`
5. Run type checker: `make typecheck`

---

## Dependency Graph

```
Phase 1 (Foundation)
  ├── 1.1 YAML Config ─────────────────────┐
  ├── 1.2 Sources ─────────────────────────┤
  ├── 1.3 Sync Engine (depends on 1.1, 1.2)┤
  ├── 1.4 MCP Sync Tool (depends on 1.3)   │
  ├── 1.5 Filter Tools (depends on 1.1)    │
  ├── 1.6 S3 Docker (independent)          │
  └── 1.7 Tests ───────────────────────────┘
                                           │
Phase 2 (Answers + Dedup + MMR) ◄──────────┘
  ├── 2.1 Answer/Citation (depends on 1.1)
  ├── 2.2 Content-Hash Dedup (depends on 1.1)
  ├── 2.3 Partial Ingest Recovery (depends on 2.2)
  ├── 2.4 MMR Search (depends on 1.1)
  ├── 2.5 Update rag_query (depends on 2.1)
  └── 2.6 Tests
                                           │
Phase 3 (Eval + CLI + MarkItDown) ◄────────┘
  ├── 3.1 Golden Set Loader (depends on 1.1)
  ├── 3.2 Metrics (depends on 3.1)
  ├── 3.3 Eval Sweep (depends on 3.2)
  ├── 3.4 MCP Eval Tools (depends on 3.3)
  ├── 3.5 CLI (depends on 1.1, 1.3, 3.3)
  ├── 3.6 MarkItDown (depends on 1.1)
  └── 3.7 Tests
                                           │
Phase 4 (Polish) ◄────────────────────────┘
  ├── 4.1 Migrate Tests
  ├── 4.2 Migrate setup.sh
  ├── 4.3 Documentation
  ├── 4.4 Integration Tests
  └── 4.5 Clean Up
```

---

## New Files Summary

| Phase | File | Purpose |
|-------|------|---------|
| 1 | `rag/yaml_config.py` | YAML config loader |
| 1 | `config.example.yaml` | Example config with all defaults |
| 1 | `rag/sources/__init__.py` | Source ABC + registry |
| 1 | `rag/sources/local.py` | Local directory source |
| 1 | `rag/sources/s3.py` | S3 source |
| 1 | `rag/sync.py` | Sync engine |
| 1 | `rag/filter_tools.py` | Filter context + extraction |
| 1 | `docker/s3-service/Dockerfile` | S3 service container |
| 1 | `docker/s3-service/server.py` | S3 HTTP API |
| 2 | `rag/answer.py` | Answer/Citation + generation |
| 2 | `rag/search/mmr.py` | MMR search algorithm |
| 3 | `rag/evaluation/golden.py` | Golden set loader |
| 3 | `rag/evaluation/metrics.py` | Evaluation metrics |
| 3 | `rag/evaluation/sweep.py` | Eval sweep comparison |
| 3 | `rag/converters/__init__.py` | Converter ABC |
| 3 | `rag/converters/docling.py` | Docling converter |
| 3 | `rag/converters/markitdown.py` | MarkItDown converter |
| 3 | `docker/markitdown/Dockerfile` | MarkItDown container |
| 3 | `docker/markitdown/server.py` | MarkItDown HTTP API |

## Modified Files Summary

| Phase | File | Change |
|-------|------|--------|
| 1 | `rag/config.py` | Refactor to use YamlConfig |
| 1 | `memex/server.py` | Add rag_sync, rag_get_filter_context, rag_extract_filters |
| 1 | `docker-compose.yml` | Add s3-service |
| 2 | `rag/ingestion.py` | Add content-hash dedup |
| 2 | `rag/pipeline.py` | Store content_hash, use MMR |
| 2 | `memex/server.py` | Update rag_query for structured answers, add search_mode |
| 3 | `memex/server.py` | Add rag_eval, rag_eval_sweep |
| 3 | `memex/cli.py` | Refactor to typer, add ingest/sync/eval |
| 3 | `pyproject.toml` | Add [project.scripts], typer dependency |
| 3 | `docker-compose.yml` | Add markitdown service |
| 4 | `README.md`, `AGENTS.md`, `DOCKER.md` | Documentation updates |
