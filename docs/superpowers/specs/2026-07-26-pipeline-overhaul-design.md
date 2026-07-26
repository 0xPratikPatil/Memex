# RAG Pipeline Overhaul — Extraction, Chunking, Embedding, Search

**Date**: 2026-07-26
**Status**: Draft
**Author**: Opencode

---

## Problem Statement

Memex v0.4.0 has a solid RAG foundation but leaves significant extraction quality and retrieval performance on the table:

1. **Docling underused**: Only requests `["md", "json"]` output. Returns `html_content` and `text_content` but discards them. Enrichment features (code, formula, picture classification, chart extraction) are available in the v1 API but never enabled.

2. **Chunking is post-hoc regex on flat text**: `_recursive_chunk()` splits markdown strings with regex patterns, throwing away the DoclingDocument structure (table boundaries, captions, list groupings, image positions, heading hierarchy). Token estimation uses a crude `len(text)//4` approximation.

3. **Advanced features disabled by default**: Contextual retrieval, query expansion (HyDE, rewrite, multi-query), metadata extraction, and embedding cache all exist in code but default to `false`. The pipeline operates at a fraction of its potential.

4. **Docker uses pip**: The ML services Dockerfile uses `pip install` instead of `uv`, making builds slower than necessary.

5. **No multi-format embedding**: All chunks are embedded as flat markdown, regardless of content type (tables lose structure, code loses language tags, images lose captions).

---

## Solution Overview

A comprehensive pipeline upgrade across all five stages, maximizing what's already available:

1. **Extraction**: Enable all free Docling enrichments, request all output formats
2. **Chunking**: Replace custom regex chunker with Docling's `HybridChunker` operating on the `DoclingDocument` structure
3. **Embedding**: Multi-format chunk serialization, contextual retrieval on by default, cache always on
4. **Search**: Enable HyDE, query rewrite, and multi-query expansion
5. **Infrastructure**: `uv` for Docker builds

All changes are config-gated. Every toggle can be disabled per-environment.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        INGESTION FLOW                            │
│                                                                  │
│  File/URL                                                       │
│    │                                                             │
│    ▼                                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Docling Serve (Docker, GPU)                              │   │
│  │                                                          │   │
│  │  Options:                                                │   │
│  │    • do_code_enrichment      (language-tagged code)      │   │
│  │    • do_formula_enrichment   (LaTeX extraction)          │   │
│  │    • do_picture_classification (chart/diagram/logo)      │   │
│  │    • do_chart_extraction     (chart→table)               │   │
│  │    • image_export_mode=embedded (base64 inline)          │   │
│  │    • to_formats: md, json, html, text                    │   │
│  │                                                          │   │
│  │  Returns: ConversionResult                               │   │
│  │    • markdown  (for text chunks)                         │   │
│  │    • json_content → DoclingDocument (for HybridChunker)  │   │
│  │    • html_content (for table chunks)                     │   │
│  │    • text_content (fallback)                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│    │                                                             │
│    ▼                                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  New: rag/chunking.py                                    │   │
│  │                                                          │   │
│  │  Docling HybridChunker                                   │   │
│  │    • Operates on DoclingDocument (JSON)                  │   │
│  │    • Tokenizer-aware (aligned to bge-m3)                 │   │
│  │    • Two-pass: split oversized + merge undersized peers  │   │
│  │    • Repeats table headers across chunk boundaries       │   │
│  │    • Preserves headings, captions, list structure        │   │
│  │                                                          │   │
│  │  Returns: List[Chunk] with:                              │   │
│  │    • content (serialized per chunk type)                 │   │
│  │    • section_header, heading_level                       │   │
│  │    • chunk_type (text|table|code|image_desc)             │   │
│  │    • provenance metadata                                │   │
│  └──────────────────────────────────────────────────────────┘   │
│    │                                                             │
│    ▼                                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Contextual Retrieval (existing, now enabled)            │   │
│  │                                                          │   │
│  │  • Generate document summary via LLM                     │   │
│  │  • Prefix each chunk: "[Context: ...] chunk content"     │   │
│  │  • Strategy: "summary" (LLM-based)                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│    │                                                             │
│    ▼                                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Embedding + Storage                                     │   │
│  │                                                          │   │
│  │  • dense → bge-m3 (1024d) via Ollama                     │   │
│  │  • contextual_dense → bge-m3 on raw content (no prefix)  │   │
│  │  • sparse → BM25 via fastembed (Qdrant/bm25)             │   │
│  │  • Embedding cache via Redis (always on)                 │   │
│  │  • Store in Qdrant with full payload                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│                        QUERY FLOW                              │
│                                                                  │
│  User query                                                    │
│    │                                                             │
│    ▼                                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Query Expansion (existing, now enabled)                 │   │
│  │                                                          │   │
│  │  Step 1: Query Rewrite                                   │   │
│  │    "that thing with numbers" →                           │   │
│  │    "Q3 2025 revenue figures and financial metrics"        │   │
│  │                                                          │   │
│  │  Step 2: HyDE                                           │   │
│  │    Generate hypothetical answer document                 │   │
│  │    "The Q3 2025 revenue was $2.3B, up 15% YoY..."       │   │
│  │    → Embed to dense vector                               │   │
│  │                                                          │   │
│  │  Step 3: Multi-Query (3 paraphrases)                     │   │
│  │    "Q3 2025 financial results"                           │   │
│  │    "third quarter revenue growth 2025"                   │   │
│  │    "Q3 2025 earnings report"                             │   │
│  │    → Embed each to dense vectors                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│    │                                                             │
│    ▼                                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Hybrid Search (existing RRF pipeline)                   │   │
│  │                                                          │   │
│  │  • original dense (rewritten query)                      │   │
│  │  • HyDE dense                                            │   │
│  │  • 3× paraphrase dense                                   │   │
│  │  • sparse BM25                                           │   │
│  │  → RRF fusion (k=60) → candidates                        │   │
│  │  → cross-encoder rerank (bge-reranker-base)             │   │
│  │  → top-k results                                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Components

### Component 1: Docling Extraction (`rag/docling_client.py`)

**What changes**: `_build_options()` returns the full set of enrichment flags and additional output formats.

```python
def _build_options() -> dict[str, Any]:
    return {
        "from_formats": ["docx", "pptx", "html", "image", "pdf", "md", "csv", "xlsx"],
        "to_formats": ["md", "json", "html", "text"],
        "do_ocr": config.ENABLE_OCR,
        "table_mode": "accurate",
        "do_table_structure": True,
        "image_export_mode": config.DOCLING_IMAGE_EXPORT,
        "do_code_enrichment": config.DOCLING_ENRICH_CODE,
        "do_formula_enrichment": config.DOCLING_ENRICH_FORMULA,
        "do_picture_classification": config.DOCLING_PICTURE_CLASSIFY,
        "do_chart_extraction": config.DOCLING_CHART_EXTRACT,
    }
```

**New config keys** (`config.py`):

| Variable | Default | Description |
|---|---|---|
| `DOCLING_ENRICH_CODE` | `true` | Code language detection on code blocks |
| `DOCLING_ENRICH_FORMULA` | `true` | LaTeX extraction from formulas |
| `DOCLING_PICTURE_CLASSIFY` | `true` | Image type classification (chart, diagram, logo, ...) |
| `DOCLING_CHART_EXTRACT` | `true` | Convert charts to tables/code |
| `DOCLING_IMAGE_EXPORT` | `embedded` | How images are emitted: `placeholder`, `embedded`, `referenced` |
| `DOCLING_PDF_BACKEND` | `DLPARSE_V1` | PDF parsing backend. `DLPARSE_V4` available as opt-in (requires `docling-parse` in the serve image; test before enabling) |

**No schema changes**: `ConversionResult` already has `json_content`, `html_content`, `text_content` fields. They just need population from the serve response.

**`ConversionResult` usage changes**: Previously only `markdown` was consumed downstream. Now:
- `json_content` → passed to `HybridChunker` for structure-aware chunking
- `html_content` → used for table-type chunks (preserves cell/column structure in embeddings)
- `markdown` → used for text-type chunks
- `text_content` → fallback plain-text source

---

### Component 2: Chunking (`rag/chunking.py` — new file)

**What it does**: Wraps Docling's `HybridChunker` to produce structure-aware, tokenizer-aligned chunks from the `DoclingDocument` JSON returned by Docling Serve.

**Interface**:

```python
from docling_core.transforms.chunker import HybridChunker
from docling_core.transforms.chunker.base import BaseChunk


def chunk_docling_document(
    docling_json: dict[str, Any],
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    merge_peers: bool = True,
    repeat_table_header: bool = True,
    tokenizer_name: str = "BAAI/bge-m3",
) -> list[dict[str, Any]]: ...
```

**Design decisions**:

1. **Tokenizer**: Uses HuggingFace tokenizer for `BAAI/bge-m3` to get exact token counts, not `len/4` estimates. Imported lazily to avoid overhead when chunking is disabled.

2. **Chunk serialization by type**: After HybridChunker produces chunks, each chunk's `chunk_type` metadata drives the serialization format used for embedding:
   - `table` → HTML (preserves `<table>`, `<tr>`, `<td>` structure)
   - `code` → Markdown fenced code block with language tag
   - `image_description` → `[Image: caption text]`
   - `text` → Markdown (default)

3. **Fallback path**: When `CHUNK_STRATEGY` is `recursive` or `fixed`, or when Docling JSON is unavailable (plain text ingest), the existing `_recursive_chunk()` and `_fixed_chunk()` functions remain as fallbacks.

4. **Dependency**: `docling` package added as an optional extra `chunking` in `pyproject.toml`. Only needed on the host side (MCP server). Not added to the Docker ML services image.

**New config keys**:

| Variable | Default | Description |
|---|---|---|
| `CHUNK_STRATEGY` | `hybrid` | `hybrid`, `recursive`, `fixed` |
| `CHUNK_SIZE` | `1024` | Target chunk size in tokens |
| `CHUNK_OVERLAP` | `128` | Overlap between chunks in tokens |
| `CHUNK_MERGE_PEERS` | `true` | HybridChunker: merge undersized adjacent chunks with same heading |
| `CHUNK_REPEAT_TABLE_HEADER` | `true` | Repeat table headers across chunk boundaries |
| `CHUNK_TYPE_FORMAT` | `true` | Use type-specific serialization (table→HTML, code→md fence, etc.) |

---

### Component 3: Embedding & Storage (`rag/pipeline.py`)

**What changes**: `ingest_text()` is refactored to accept pre-chunked data from the new chunker when available. The embedding path (dense + sparse + contextual_dense + metadata) remains the same. Key behavioral changes:

1. **Contextual retrieval now on by default**: `ENABLE_CONTEXTUAL_RETRIEVAL=true` with `CONTEXT_STRATEGY=summary`. The existing `ContextGenerator` in `rag/services/contextual_retrieval.py` is unchanged — it already works.

2. **Embedding cache always on**: `ENABLE_CACHE=true`. Redis is already in docker-compose, the cache layer (`rag/services/cache.py`) is already implemented. The toggle just needs flipping.

3. **Metadata extraction enabled**: All metadata extractors enabled by default (entities, doc classification, topic tagging, language detection). The `MetadataExtractor` in `rag/services/metadata_extractor.py` is already implemented.

4. **Chunk-type-aware payload**: When `CHUNK_TYPE_FORMAT=true`, the Qdrant payload includes `chunk_type` (text, table, code, image_description) for potential future filtering.

**Config changes**:

| Variable | Old | New |
|---|---|---|
| `ENABLE_CACHE` | `false` | `true` |
| `ENABLE_CONTEXTUAL_RETRIEVAL` | `false` | `true` |
| `CONTEXT_STRATEGY` | `header` | `summary` |
| `ENABLE_METADATA_EXTRACTION` | `false` | `true` |
| `ENABLE_ENTITY_EXTRACTION` | `false` | `true` |
| `ENABLE_DOC_CLASSIFICATION` | `false` | `true` |
| `ENABLE_TOPIC_TAGGING` | `false` | `true` |
| `ENABLE_LANGUAGE_DETECTION` | `true` | `true` (no change) |
| `EMBED_BATCH_SIZE` | `32` | `64` |

**Model stack unchanged**:
- Dense: `bge-m3` (1024d) via Ollama
- Sparse: `Qdrant/bm25` via fastembed (Docker ML service)
- Reranker: `BAAI/bge-reranker-base` via sentence-transformers (Docker ML service)
- Chat: `qwen2.5:0.5b` via Ollama (for context, metadata, and query expansion)

**How the models work together**: Dense (bge-m3) provides semantic recall — finds conceptually related chunks even without keyword overlap. Sparse (BM25) provides lexical precision — catches exact terms, API names, error codes. RRF (Reciprocal Rank Fusion, k=60) merges both rankings, boosting chunks that score well in both. The cross-encoder reranker (bge-reranker-base) then reads query+document pairs through full cross-attention, re-scoring the top candidates — this catches false positives where a chunk uses the right keywords but isn't actually *about* the query.

---

### Component 4: Search & Query Expansion (`rag/pipeline.py` + `rag/services/query_expansion.py`)

**What changes**: Enable the already-implemented query expansion techniques. No new code in the expansion module — only config defaults change. The search pipeline (`hybrid_search()`) already handles `ExpandedQuery` with HyDE vectors, paraphrases, and rewritten queries — this is already wired in.

1. **HyDE enabled**: `ENABLE_HYDE=true`. Before search, the LLM generates a hypothetical answer document, embeds it, and searches alongside the original query. HyDE vectors improve recall for conceptual/narrative queries by embedding a "fake answer" that sits closer to real answer chunks than the short query does.

2. **Query rewrite enabled**: `ENABLE_QUERY_REWRITE=true`. Takes ambiguous or conversational queries and expands them into well-formed search queries before embedding. The rewritten query replaces the original for the main dense search.

3. **Multi-query enabled**: `ENABLE_MULTI_QUERY=true`, `MULTI_QUERY_COUNT=3`. Generates 3 paraphrases of the query, each searched independently via dense, all fused through RRF. The embedding cache prevents redundant Ollama calls for common paraphrases.

**Execution order** (sequential, each gated by its flag):
```
Query Rewrite → HyDE generation → Multi-Query paraphrases → Search + RRF → Rerank
```

Each step is independent. Failures are logged and skipped — the pipeline degrades gracefully.

**Config changes**:

| Variable | Old | New |
|---|---|---|
| `ENABLE_QUERY_EXPANSION` | `false` | `true` |
| `ENABLE_HYDE` | `false` | `true` |
| `ENABLE_QUERY_REWRITE` | `false` | `true` |
| `ENABLE_MULTI_QUERY` | `false` | `true` |
| `SEARCH_TOP_K` | `20` | `30` |

**Cache coverage**: With `ENABLE_CACHE=true`:
- Embedding cache: re-uses dense vectors for repeated text (chunks, queries, HyDE answers)
- Search cache (`CACHE_TTL_SEARCH=3600`): caches full result sets for repeated queries
- Parse cache (`CACHE_TTL_PARSE=604800`): caches Docling conversion results for 7 days

---

### Component 5: Docker ML Services (`Dockerfile`)

**What changes**: Replace `pip install` with `uv pip install` for faster, reproducible builds.

```dockerfile
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime AS deps
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH"
RUN uv pip install --no-cache \
    "fastembed>=0.4,<1" \
    "sentence-transformers>=3,<5" \
    "fastapi[standard]>=0.115,<1" \
    "uvicorn[standard]>=0.30,<1" \
    "pydantic>=2,<3" \
    "httpx>=0.27,<1"
```

The preload stage and runtime stage remain the same apart from `VIRTUAL_ENV` path changes.

**What stays the same**: `docling-serve` (pre-built ghcr image), `qdrant/qdrant`, `ollama/ollama`, `redis` images are all pre-built pulls — no changes.

---

### Component 6: Dependencies (`pyproject.toml`)

**New extra**: `chunking` for host-side HybridChunker usage:

```toml
[project.optional-dependencies]
chunking = ["docling>=2"]
```

Not a core dependency — the chunker only runs on the host MCP server, not in Docker.

---

## Files Changed

| File | Change |
|---|---|
| `rag/docling_client.py` | `_build_options()` adds all enrichment flags, requests html+text formats |
| `rag/chunking.py` | **New** — wraps Docling HybridChunker with multi-format serialization |
| `rag/pipeline.py` | `ingest_text()` integrates new chunker; old chunker retained as fallback |
| `rag/config.py` | All new env vars (~20 new settings) |
| `.env.example` | Updated defaults matching new config |
| `Dockerfile` | `uv`-based build for ML services |
| `pyproject.toml` | `chunking` extra: `docling>=2` |
| `README.md` | Updated feature table, config reference, chunking docs |

---

## Error Handling

- Docling enrichment flags are additive — if a flag isn't supported by the running serve version, it's silently ignored (request extras, not requirements)
- `DLPARSE_V4` defaults to off — if enabled and not available, V1 fallback is automatic
- HybridChunker import is lazy; if `docling` package is not installed and `CHUNK_STRATEGY=hybrid`, falls back to `recursive` with a warning
- Query expansion steps are individually wrapped in try/except — HyDE failure doesn't prevent rewrite, multi-query failure doesn't prevent HyDE
- Embedding cache is fallback-tolerant — if Redis is down, embeddings proceed without caching
- All feature toggles are independently gated — a misconfiguration in one stage doesn't cascade

---

## Testing Strategy

- **Unit tests**: New tests for `rag/chunking.py` (HybridChunker integration, fallback paths, chunk-type serialization)
- **Unit tests**: Updated tests for `rag/docling_client.py` (new options payload)
- **Integration tests**: Verify end-to-end ingest with enrichments enabled on live Docling serve
- **Integration tests**: Verify search with HyDE + multi-query expansion
- **Regression**: All 234 existing tests (181 unit + 53 integration) must pass with new defaults
- **Config tests**: Verify each new flag gates its behavior correctly

---

## Migration Notes

- **Existing Qdrant collections**: No migration needed. New config applies to new ingests. Re-ingest to benefit from new chunking and enrichments.
- **Existing .env files**: New keys are additive. Old `.env` files work with old defaults; new `.env.example` shows recommended values.
- **Docker**: Rebuild ml-services image with `docker compose build ml-services` after Dockerfile change.
- **`uv sync`**: Run `uv sync --extra chunking` for development if using HybridChunker.
