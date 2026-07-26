# Metadata Enhancement Design

**Date**: 2026-07-26
**Status**: Draft
**Author**: Opencode

---

## Problem Statement

Current chunk metadata is minimal (`pipeline.py:414-423`):

```python
point_meta = {
    "source": source_identifier,
    "chunk_index": idx,
    "total_chunks": len(raw_chunks),
    "content": chunk["content"],
    "section_header": chunk.get("section_header", ""),
    "ingested_at": now,
    "content_hash": content_hash,
}
```

This limits filtering and retrieval capabilities:

1. **No entity information**: Can't filter by people, organizations, dates mentioned in documents.
2. **No document classification**: Can't distinguish reports, emails, code, articles by type.
3. **No topic tagging**: Can't filter by subject area (finance, engineering, legal).
4. **No language detection**: Can't filter by language.
5. **No structural metadata**: No page numbers, heading hierarchy, or document position.

---

## Solution Overview

Enhance metadata extraction pipeline:

1. **Entity Extraction**: Extract people, organizations, dates, locations from chunks.
2. **Document Classification**: Classify document type (report, email, code, article, etc.).
3. **Topic Tagging**: Assign topic labels using LLM or zero-shot classification.
4. **Structural Metadata**: Capture heading hierarchy, chunk position, document structure.
5. **Language Detection**: Detect and store document language.

All metadata is extracted during ingestion and stored in Qdrant payload, enabling rich filtering.

---

## Architecture

```
┌──────────────┐
│ Document Text │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Chunking     │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│  Metadata Extractor   │  (new)
│                       │
│  ┌─────────────────┐  │
│  │ Entity Extractor │  │  spaCy / LLM
│  └────────┬────────┘  │
│           │           │
│  ┌────────▼────────┐  │
│  │ Doc Classifier   │  │  LLM / rules
│  └────────┬────────┘  │
│           │           │
│  ┌────────▼────────┐  │
│  │ Topic Tagger     │  │  LLM / zero-shot
│  └────────┬────────┘  │
│           │           │
│  ┌────────▼────────┐  │
│  │ Language Detect  │  │  fasttext / langdetect
│  └────────┬────────┘  │
│           │           │
│  ┌────────▼────────┐  │
│  │ Structural Meta  │  │  regex / parse
│  └────────┬────────┘  │
│           │           │
└───────────┼───────────┘
            │
            ▼
┌──────────────────────┐
│  Enriched Metadata    │
│  (stored in Qdrant)   │
└──────────────────────┘
```

---

## Files to Modify/Create

| File | Action | Purpose |
|------|--------|---------|
| `src/metadata_extractor.py` | **Create** | All metadata extraction logic |
| `src/config.py` | **Modify** | Add metadata feature toggles |
| `src/pipeline.py` | **Modify** | Integrate metadata extraction into `ingest_text` |
| `tests/unit/test_metadata_extractor.py` | **Create** | Unit tests for each extractor |
| `tests/integration/test_metadata_ingest.py` | **Create** | Test metadata in Qdrant |
| `pyproject.toml` | **Modify** | Add optional dependencies |

---

## Implementation Details

### 1. Configuration (`config.py` additions)

```python
# ── Metadata Enhancement ────────────────────────────────────────────────────
ENABLE_METADATA_EXTRACTION: bool = _env_bool("ENABLE_METADATA_EXTRACTION", False)
ENABLE_ENTITY_EXTRACTION: bool = _env_bool("ENABLE_ENTITY_EXTRACTION", False)
ENABLE_DOC_CLASSIFICATION: bool = _env_bool("ENABLE_DOC_CLASSIFICATION", False)
ENABLE_TOPIC_TAGGING: bool = _env_bool("ENABLE_TOPIC_TAGGING", False)
ENABLE_LANGUAGE_DETECTION: bool = _env_bool("ENABLE_LANGUAGE_DETECTION", True)

METADATA_MODEL: str = _env("METADATA_MODEL", "")  # empty = use EMBED_MODEL via Ollama
MAX_ENTITIES_PER_CHUNK: int = _env_int("MAX_ENTITIES_PER_CHUNK", 10)
MAX_TOPICS_PER_CHUNK: int = _env_int("MAX_TOPICS_PER_CHUNK", 5)
```

### 2. Metadata Extractor Module (`src/metadata_extractor.py`)

```python
"""Metadata extraction: entities, classification, topics, language."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any

import httpx
from . import config

logger = logging.getLogger("metadata-extractor")


class MetadataExtractor:
    """Extracts rich metadata from document chunks."""

    def __init__(self, ollama_client: httpx.Client | None = None):
        self._ollama = ollama_client
        self._lang_detector = None

    def extract_all(
        self,
        chunk: dict[str, Any],
        document_text: str = "",
        source_identifier: str = "",
    ) -> dict[str, Any]:
        """Extract all configured metadata from a chunk."""
        metadata: dict[str, Any] = {}

        if config.ENABLE_ENTITY_EXTRACTION:
            metadata["entities"] = self.extract_entities(chunk["content"])

        if config.ENABLE_DOC_CLASSIFICATION and chunk.get("chunk_index", 0) == 0:
            metadata["doc_type"] = self.classify_document(document_text or chunk["content"])

        if config.ENABLE_TOPIC_TAGGING:
            metadata["topics"] = self.extract_topics(chunk["content"])

        if config.ENABLE_LANGUAGE_DETECTION:
            metadata["language"] = self.detect_language(chunk["content"])

        metadata["structural"] = self.extract_structural(chunk, source_identifier)

        return metadata

    def extract_entities(self, text: str) -> dict[str, list[str]]:
        """Extract named entities from text using LLM."""
        prompt = (
            "Extract named entities from this text. Return JSON with keys: "
            "people, organizations, dates, locations, products. "
            "Each value is a list of unique strings. Only output JSON.\n\n"
            f"Text: {text[:1000]}"
        )
        try:
            response = self._chat(prompt)
            entities = json.loads(response)
            # Limit entity count
            return {k: v[: config.MAX_ENTITIES_PER_CHUNK] for k, v in entities.items()}
        except (json.JSONDecodeError, Exception) as exc:
            logger.debug("Entity extraction failed: %s", exc)
            return {}

    def classify_document(self, text: str) -> str:
        """Classify document type using LLM."""
        prompt = (
            "Classify this document into one of these types: "
            "report, email, article, code, documentation, presentation, "
            "resume, contract, invoice, meeting_notes, other. "
            "Only output the type.\n\n"
            f"Text: {text[:2000]}"
        )
        try:
            return self._chat(prompt).strip().lower()
        except Exception as exc:
            logger.debug("Document classification failed: %s", exc)
            return "unknown"

    def extract_topics(self, text: str) -> list[str]:
        """Extract topic labels from text."""
        prompt = (
            f"Extract up to {config.MAX_TOPICS_PER_CHUNK} topic labels from this text. "
            "Return as JSON array of strings. Only output JSON.\n\n"
            f"Text: {text[:1000]}"
        )
        try:
            response = self._chat(prompt)
            topics = json.loads(response)
            return topics[: config.MAX_TOPICS_PER_CHUNK]
        except (json.JSONDecodeError, Exception) as exc:
            logger.debug("Topic extraction failed: %s", exc)
            return []

    def detect_language(self, text: str) -> str:
        """Detect text language."""
        try:
            if self._lang_detector is None:
                from langdetect import detect

                self._lang_detector = detect
            return self._lang_detector(text[:500])
        except Exception:
            return "unknown"

    def extract_structural(
        self,
        chunk: dict[str, Any],
        source_identifier: str,
    ) -> dict[str, Any]:
        """Extract structural metadata from chunk position."""
        content = chunk.get("content", "")
        header = chunk.get("section_header", "")

        # Parse heading level from header
        heading_level = 0
        if header:
            match = re.match(r"^(#{1,6})\s+", header)
            if match:
                heading_level = len(match.group(1))

        # Detect list items, code blocks, tables
        is_list = bool(re.match(r"^\s*[-*\d]+[.)]\s", content))
        is_code = "```" in content or content.startswith("    ")
        is_table = "|" in content and content.count("|") >= 3

        return {
            "chunk_index": chunk.get("chunk_index", 0),
            "total_chunks": chunk.get("total_chunks", 0),
            "heading_level": heading_level,
            "section_header": header,
            "is_list": is_list,
            "is_code": is_code,
            "is_table": is_table,
            "char_count": len(content),
            "word_count": len(content.split()),
        }

    def _chat(self, prompt: str) -> str:
        """Call Ollama chat API."""
        if self._ollama is None:
            raise RuntimeError("Ollama client not available for metadata extraction")
        resp = self._ollama.post(
            config.OLLAMA_EMBED_URL.replace("/api/embeddings", "/api/chat"),
            json={
                "model": config.METADATA_MODEL or config.EMBED_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
```

### 3. Pipeline Integration (`pipeline.py` changes)

```python
def ingest_text(self, text, source_identifier, metadata=None, content_hash="", progress_cb=None):
    # ... existing chunking ...
    raw_chunks = create_chunks(text)

    # NEW: Metadata extraction
    if config.ENABLE_METADATA_EXTRACTION:
        from src.metadata_extractor import MetadataExtractor

        extractor = MetadataExtractor(self._get_ollama())

        _progress("Extracting metadata...", 73)
        for chunk in raw_chunks:
            chunk_meta = extractor.extract_all(
                chunk=chunk,
                document_text=text,
                source_identifier=source_identifier,
            )
            chunk["metadata"] = chunk_meta

    # ... existing embedding and storage ...

    # In the point creation loop:
    point_meta = {
        "source": source_identifier,
        "chunk_index": idx,
        "total_chunks": len(raw_chunks),
        "content": chunk["content"],
        "section_header": chunk.get("section_header", ""),
        "ingested_at": now,
        "content_hash": content_hash,
        **(chunk.get("metadata", {})),  # NEW: merge extracted metadata
        **base_meta,
    }
```

### 4. Enhanced Filtering in Search

Update `hybrid_search` to support metadata filters:

```python
def hybrid_search(
    self,
    query: str,
    top_k: int = 5,
    rerank: bool = True,
    source_filter: str | None = None,
    metadata_filter: dict[str, Any] | None = None,  # NEW
) -> list[dict[str, Any]]:
    # Build filter conditions
    filter_conditions = []
    if source_filter:
        filter_conditions.append(FieldCondition(key="source", match=MatchValue(value=source_filter)))
    if metadata_filter:
        for key, value in metadata_filter.items():
            if isinstance(value, list):
                # Multi-value match
                filter_conditions.append(FieldCondition(key=f"metadata.{key}", match=MatchAny(values=value)))
            else:
                filter_conditions.append(FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value)))

    qdrant_filter = Filter(must=filter_conditions) if filter_conditions else None
    # ... rest of search logic
```

### 5. MCP Server Enhancement

Add metadata filter parameter to `rag_query`:

```python
@mcp.tool(...)
async def rag_query(
    query: str,
    top_k: int = 5,
    use_reranking: bool = True,
    source_filter: str | None = None,
    metadata_filter: dict[str, str] | None = None,  # NEW
    response_format: ResponseFormat = ResponseFormat.MARKDOWN,
) -> Any:
    results = engine.hybrid_search(
        query=query,
        top_k=top_k,
        rerank=use_reranking,
        source_filter=source_filter,
        metadata_filter=metadata_filter,
    )
    # ...
```

---

## Testing Strategy

### Unit Tests (`tests/unit/test_metadata_extractor.py`)

- `test_extract_entities_returns_dict`: Verify entity extraction returns structured dict.
- `test_classify_document_returns_string`: Verify classification returns valid type.
- `test_extract_topics_returns_list`: Verify topic extraction returns list.
- `test_detect_language_returns_code`: Verify language detection returns ISO code.
- `test_extract_structural_heading_level`: Verify heading level parsed correctly.
- `test_extract_structural_flags`: Verify is_list, is_code, is_table detected.
- `test_extract_all_merges_metadata`: Verify all metadata fields present.
- `test_disabled_features_not_extracted`: When flags off, metadata is empty.

### Integration Tests (`tests/integration/test_metadata_ingest.py`)

- `test_ingest_with_metadata`: Ingest document, verify metadata stored in Qdrant payload.
- `test_search_with_metadata_filter`: Search with entity/type filter, verify filtered results.
- `test_metadata_in_list_documents`: Verify metadata visible in document listing.
- `test_metadata_on_reingest`: Re-ingest same document, verify metadata updated.

### Evaluation

Compare retrieval with/without metadata filtering:
- Ingest 20 documents of different types
- Run queries with type/entity filters
- Measure precision improvement from filtering

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM calls slow down ingestion significantly | Slow ingest | Extract metadata in parallel; use header strategy (no LLM) as fallback |
| Entity extraction hallucinates entities | Incorrect metadata | Validate against known entities; use rule-based extraction as fallback |
| Metadata increases storage significantly | Higher Qdrant storage | Each chunk adds ~200-500 bytes; negligible for typical collections |
| Language detection fails on short text | Wrong language tag | Fall back to "unknown"; only detect on chunks > 50 chars |
| Metadata extraction errors break ingestion | Ingestion failure | Wrap all extraction in try/except; metadata is optional enrichment |
| LLM model unavailable for extraction | Extraction skipped | Graceful fallback to empty metadata; don't block ingestion |

---

## Priority & Effort

- **Priority**: Medium (enables advanced filtering but not required for basic RAG)
- **Estimated effort**: 2-3 days
- **Dependencies**: Optional: `langdetect` for language detection
- **Rollback**: Feature flag `ENABLE_METADATA_EXTRACTION` defaults to `False`
