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
        description="Filter by metadata fields (doc_type, topics, language, keywords, entities.people, entities.dates)",
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
    search_mode: str | None = Field(
        default=None,
        description="Override search mode: 'similarity', 'hybrid', 'mmr' (null = use config default)",
    )
    generate_answer: bool | None = Field(
        default=None,
        description="Override answer generation (null = use config setting)",
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
    entities: dict[str, Any] = Field(default_factory=dict)
    dates: list[str] = Field(default_factory=list)


class CitationInfo(BaseModel):
    """A citation referencing a search result chunk."""

    index: int
    source: str
    snippet: str = ""
    score: float | None = None


class AnswerOutput(BaseModel):
    """Structured answer with citations from rag_query."""

    text: str
    refused: bool
    confidence: float
    citations: list[CitationInfo]
    sources: list[str]
    search_mode: str = "hybrid"
    results: list[SearchResult] = Field(default_factory=list)


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


# ── Sync Tool Schemas ────────────────────────────────────────────────────────


class SyncInput(BaseModel):
    source_name: str | None = Field(
        default=None,
        description="Sync a specific source by name. Null = sync all sources.",
    )
    dry_run: bool = Field(
        default=False,
        description="Report what would change without writing",
    )


class SyncStatsOutput(BaseModel):
    """Output from rag_sync."""

    added: int = 0
    changed: int = 0
    deleted: int = 0
    unchanged: int = 0
    errors: list[str] = Field(default_factory=list)
    dry_run: bool = False


# ── Filter Tool Schemas ──────────────────────────────────────────────────────


class FilterContextInput(BaseModel):
    query: str | None = Field(
        default=None,
        description="Optional query to get filter suggestions for",
    )


class FieldInfoOutput(BaseModel):
    """A discovered metadata field."""

    name: str
    type: str
    values: list[str]
    count: int


class FilterContextOutput(BaseModel):
    """Output from rag_get_filter_context."""

    fields: list[FieldInfoOutput]
    suggested_filters: dict[str, str | list[str]] | None = None
    sample_query: str = ""


class ExtractFiltersInput(BaseModel):
    query: str = Field(
        ...,
        description="Natural language query to extract metadata filters from",
        min_length=2,
        max_length=500,
    )


class ExtractedFiltersOutput(BaseModel):
    """Output from rag_extract_filters."""

    filters: dict[str, str | list[str]]
    explanation: str
    confidence: float


# ── Eval Tool Schemas ───────────────────────────────────────────────────────


class EvalInput(BaseModel):
    """Input for rag_eval — golden-set evaluation."""

    golden_set_path: str = Field(
        ...,
        description="Path to golden set YAML/JSON file",
        min_length=1,
    )
    top_k: int = Field(
        default=5,
        description="Results per query",
        ge=1,
        le=50,
    )
    compare_rerank: bool = Field(
        default=False,
        description="Compare with/without reranking",
    )
    source_match_mode: str = Field(
        default="basename",
        description="Source matching mode: basename, exact, contains",
    )


class EvalQueryResult(BaseModel):
    """Per-query evaluation result."""

    query: str
    recall: float
    precision: float
    hit_rate: float
    mrr: float
    keyword_coverage: float
    expected_sources: list[str]
    retrieved_sources: list[str]


class EvalOutput(BaseModel):
    """Output from rag_eval — aggregate and per-query metrics."""

    total_queries: int
    avg_recall: float
    avg_precision: float
    avg_hit_rate: float
    avg_mrr: float
    avg_keyword_coverage: float
    queries: list[EvalQueryResult]


class EvalSweepInput(BaseModel):
    """Input for rag_eval_sweep — compare multiple retrieval configs."""

    golden_set_path: str = Field(
        ...,
        description="Path to golden set YAML/JSON file",
        min_length=1,
    )
    variants: list[dict[str, Any]] = Field(
        ...,
        description="List of variant configs to compare",
        min_length=1,
    )
    top_k: int = Field(
        default=5,
        description="Results per query",
        ge=1,
        le=50,
    )


class EvalSweepOutput(BaseModel):
    """Output from rag_eval_sweep — comparison table with deltas."""

    variants: list[EvalOutput]
    delta_table: str
