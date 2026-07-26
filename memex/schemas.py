"""Pydantic models for MCP tool input validation and structured output."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ResponseFormat(StrEnum):
    MARKDOWN = "markdown"
    JSON = "json"


# ── Input Schemas ──────────────────────────────────────────────────────────────


class IngestFileInput(BaseModel):
    file_path_or_url: str = Field(
        ...,
        description="Local file path or URL to the document to ingest",
        min_length=1,
    )


class IngestUrlInput(BaseModel):
    url: str = Field(
        ...,
        description="URL of the document to ingest",
        min_length=1,
    )


class IngestBatchInput(BaseModel):
    items: list[str] = Field(
        ...,
        description="List of file paths or URLs to ingest",
        min_length=1,
    )


class QueryInput(BaseModel):
    query: str = Field(
        ...,
        description="Natural language search query",
        min_length=2,
        max_length=500,
    )
    top_k: int = Field(
        default=5,
        description="Max results to fetch from backend",
        ge=1,
        le=50,
    )
    use_reranking: bool = Field(
        default=True,
        description="Apply cross-encoder reranking",
    )
    source_filter: str | None = Field(
        default=None,
        description="Filter results to a specific document source",
    )
    use_query_expansion: bool | None = Field(
        default=None,
        description="Override server default for query expansion (null = use server setting)",
    )
    use_contextual_search: bool | None = Field(
        default=None,
        description="Use contextual embeddings for search (null = use server setting)",
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )
    metadata_filter: dict[str, str | list[str]] | None = Field(
        default=None,
        description="Filter by metadata fields (doc_type, topics, language, keywords, entities.people, dates)",
    )
    offset: int = Field(
        default=0,
        description="Pagination offset (skip first N results)",
        ge=0,
    )
    limit: int = Field(
        default=10,
        description="Max results to include in response",
        ge=1,
        le=50,
    )


class DeleteDocumentInput(BaseModel):
    source_identifier: str = Field(
        ...,
        description="File path or URL of the document to delete",
        min_length=1,
    )


class ListDocumentsInput(BaseModel):
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format",
    )
    offset: int = Field(
        default=0,
        description="Pagination offset (skip first N documents)",
        ge=0,
    )
    limit: int = Field(
        default=20,
        description="Max documents to include in response",
        ge=1,
        le=100,
    )


# ── Output Schemas ─────────────────────────────────────────────────────────────


class SearchResult(BaseModel):
    """A single search result chunk."""

    id: str
    rrf_score: float | None = None
    rerank_score: float | None = None
    source: str
    content: str
    section_header: str = ""
    context_prefix: str = ""
    doc_type: str = ""
    topics: list[str] = Field(default_factory=list)
    language: str = ""
    keywords: list[str] = Field(default_factory=list)


class QueryOutput(BaseModel):
    """Structured output from rag_query."""

    total: int
    count: int
    results: list[SearchResult]


class DocumentInfo(BaseModel):
    """Metadata for an ingested document."""

    source: str
    chunk_count: int
    total_chunks: int = 0
    ingested_at: str = ""
    sections: list[str] = Field(default_factory=list)
    doc_type: str = ""
    topics: list[str] = Field(default_factory=list)
    language: str = ""
    keywords: list[str] = Field(default_factory=list)


class ListDocumentsOutput(BaseModel):
    """Structured output from rag_list_documents."""

    total: int
    documents: list[DocumentInfo]


class ServiceStatusEntry(BaseModel):
    """Status of a single backend service."""

    healthy: bool
    url: str
    latency_ms: float | None = None
    error: str | None = None


class CollectionStatsOutput(BaseModel):
    """Structured output from rag_collection_stats."""

    collection_name: str
    total_points: int
    total_vectors: int
    status: str = ""
    optimizer_status: str = ""
    config: dict[str, Any] = Field(default_factory=dict)
