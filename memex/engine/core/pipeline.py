"""RAG pipeline: embeddings, Qdrant storage, hybrid search, reranking, MMR.

Uses ``httpx`` for connection-pooled Ollama calls, ``tenacity`` for retries,
and lazy-init for heavy models (sparse embeddings, reranker).

Key improvements over v1:
- Async I/O throughout
- Docling HybridChunker for structure-aware tokenizer-aligned chunking
- Legacy recursive/fixed chunking as fallback
- Reciprocal Rank Fusion (RRF) for dense+sparse combination
- Batch embedding via Ollama API
- Rich per-chunk metadata
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    Modifier,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from memex.engine.core import config
from memex.engine.core.errors import ConfigError
from memex.engine.core.progress import PipelineStage
from memex.engine.ingestion.context import ContextGenerator, strip_context_prefix
from memex.engine.ingestion.embedding import EmbeddingService
from memex.engine.llm.base import EmbedProvider, LLMProvider
from memex.engine.metadata.extractor import MetadataExtractor
from memex.engine.retrieval.expansion import ExpandedQuery

logger = logging.getLogger("rag-pipeline")

# ── Evaluation hooks ────────────────────────────────────────────────────────

_eval_timings: dict[str, list[float]] = {}
_eval_timings_lock = threading.Lock()
_EVAL_TIMINGS_MAX_PER_STAGE = 1000  # cap per-stage entries to prevent unbounded growth


def _record_eval_timing(stage: str, elapsed_ms: float) -> None:
    """Record pipeline stage timing when evaluation logging is enabled."""
    if not config.EVAL_ENABLED or not config.EVAL_LOG_TIMING:
        return
    with _eval_timings_lock:
        if stage not in _eval_timings:
            _eval_timings[stage] = []
        _eval_timings[stage].append(elapsed_ms)
        if len(_eval_timings[stage]) > _EVAL_TIMINGS_MAX_PER_STAGE:
            _eval_timings[stage] = _eval_timings[stage][-_EVAL_TIMINGS_MAX_PER_STAGE:]
    logger.debug("eval-timing|%s|%.2fms", stage, elapsed_ms)


def get_eval_timings() -> dict[str, list[float]]:
    """Return accumulated evaluation timing data."""
    with _eval_timings_lock:
        return dict(_eval_timings)


def reset_eval_timings() -> None:
    """Clear accumulated evaluation timing data."""
    with _eval_timings_lock:
        _eval_timings.clear()


# ── Chunking ──────────────────────────────────────────────────────────────────


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return max(1, len(text) // 4)


def _recursive_chunk(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> list[dict[str, Any]]:
    """Split text recursively by semantic boundaries.

    Hierarchy: headers > paragraphs > sentences > words.
    Each chunk carries its nearest section header as metadata.
    """
    chunks: list[dict[str, Any]] = []
    current_header = ""

    def _split_by_pattern(t: str, pattern: str) -> list[str]:
        parts = re.split(pattern, t)
        result = []
        for i in range(0, len(parts), 2):
            segment = parts[i]
            if i + 1 < len(parts):
                segment += parts[i + 1]
            if segment.strip():
                result.append(segment.strip())
        return result

    def _chunk_section(section_text: str, header: str) -> None:
        est = _estimate_tokens(section_text)
        if est <= max_tokens:
            if section_text.strip():
                chunks.append(
                    {
                        "content": section_text.strip(),
                        "section_header": header,
                    }
                )
            return

        # Try splitting by paragraphs
        paragraphs = re.split(r"\n\s*\n", section_text)
        if len(paragraphs) > 1:
            buffer = ""
            for para in paragraphs:
                combined = f"{buffer}\n\n{para}".strip() if buffer else para.strip()
                if _estimate_tokens(combined) <= max_tokens:
                    buffer = combined
                else:
                    if buffer:
                        chunks.append(
                            {
                                "content": buffer,
                                "section_header": header,
                            }
                        )
                    if _estimate_tokens(para.strip()) > max_tokens:
                        _chunk_section(para.strip(), header)
                    else:
                        buffer = para.strip()
            if buffer:
                chunks.append(
                    {
                        "content": buffer,
                        "section_header": header,
                    }
                )
            return

        # Try splitting by sentences
        sentences = re.split(r"(?<=[.!?])\s+", section_text)
        if len(sentences) > 1:
            buffer = ""
            for sent in sentences:
                combined = f"{buffer} {sent}".strip() if buffer else sent.strip()
                if _estimate_tokens(combined) <= max_tokens:
                    buffer = combined
                else:
                    if buffer:
                        chunks.append(
                            {
                                "content": buffer,
                                "section_header": header,
                            }
                        )
                    buffer = sent.strip()
            if buffer:
                chunks.append(
                    {
                        "content": buffer,
                        "section_header": header,
                    }
                )
            return

        # Fallback: hard split by words with overlap
        words = section_text.split()
        step = max(1, max_tokens - overlap_tokens)
        word_step = max(1, step // 4)
        for i in range(0, len(words), word_step):
            chunk_words = words[i : i + max_tokens]
            chunk_text = " ".join(chunk_words)
            if _estimate_tokens(chunk_text) >= 10:
                chunks.append(
                    {
                        "content": chunk_text,
                        "section_header": header,
                    }
                )

    # Split by markdown headers first
    header_pattern = r"(^#{1,6}\s+.+$)"
    sections = re.split(header_pattern, text, flags=re.MULTILINE)

    i = 0
    while i < len(sections):
        section = sections[i]
        if re.match(r"^#{1,6}\s+", section):
            current_header = section.strip()
            i += 1
            if i < len(sections):
                _chunk_section(sections[i], current_header)
        else:
            _chunk_section(section, current_header)
        i += 1

    # Apply overlap by duplicating tail/head between adjacent chunks.
    if overlap_tokens > 0 and len(chunks) > 1:
        overlapped: list[dict[str, Any]] = [chunks[0]]
        for idx in range(1, len(chunks)):
            prev_words = chunks[idx - 1]["content"].split()
            overlap_word_count = min(overlap_tokens // 4, len(prev_words))
            if overlap_word_count > 0:
                overlap_text = " ".join(prev_words[-overlap_word_count:])
                combined = f"{overlap_text} {chunks[idx]['content']}"
                if _estimate_tokens(combined) <= int(max_tokens * 1.5):
                    chunks[idx]["content"] = combined
            overlapped.append(chunks[idx])
        chunks = overlapped

    return chunks


def _fixed_chunk(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> list[dict[str, Any]]:
    """Simple fixed-size word splitting (fallback)."""
    words = text.split()
    step = max(1, max_tokens - overlap_tokens)
    chunks: list[dict[str, Any]] = []
    for i in range(0, len(words), step):
        chunk_words = words[i : i + max_tokens]
        chunk_text = " ".join(chunk_words)
        if len(chunk_text.strip()) >= config.MIN_CHUNK_LEN:
            chunks.append(
                {
                    "content": chunk_text,
                    "section_header": "",
                }
            )
    return chunks


def create_chunks(
    text: str = "",
    source_identifier: str = "",
) -> list[dict[str, Any]]:
    """Create chunks from a document.

    When *source_identifier* is provided and ``CHUNK_STRATEGY`` is ``hybrid``,
    uses Docling Serve's ``/v1/chunk/hybrid/source`` API for structure-aware
    chunking.  Falls back to legacy recursive/fixed chunking otherwise.
    """
    strategy = config.CHUNK_STRATEGY.lower()

    if strategy == "hybrid" and source_identifier:
        try:
            from memex.engine.ingestion.splitter import chunk_file

            result = chunk_file(source_identifier)
            chunks = result.get("chunks", [])
            return [c for c in chunks if len(c["content"].strip()) >= config.MIN_CHUNK_LEN]
        except ImportError:
            pass
        except Exception:
            logger.warning("Docling chunking API failed, falling back to recursive", exc_info=True)

    if not text.strip():
        return []

    max_tokens = config.CHUNK_SIZE
    overlap_tokens = config.CHUNK_OVERLAP

    if strategy == "recursive" or strategy == "hybrid":
        raw = _recursive_chunk(text, max_tokens, overlap_tokens)
    elif strategy == "fixed":
        raw = _fixed_chunk(text, max_tokens, overlap_tokens)
    else:
        raw = _recursive_chunk(text, max_tokens, overlap_tokens)

    return [c for c in raw if len(c["content"].strip()) >= config.MIN_CHUNK_LEN]


# ── RAG Engine ───────────────────────────────────────────────────────────────


class RAGEngine:
    """Manages Qdrant storage, embeddings, hybrid search and reranking."""

    def __init__(self) -> None:
        from memex.engine.llm import get_embedder, get_llm

        self._qdrant: QdrantClient | None = None
        self._ml_services: httpx.Client | None = None
        self._embedding_svc: EmbeddingService | None = None
        self._llm: LLMProvider = get_llm()
        self._embedder: EmbedProvider = get_embedder()
        self._dim_validated: bool = False

    # ── Lazy singletons ───────────────────────────────────────────────────

    def _get_qdrant(self) -> QdrantClient:
        if self._qdrant is None:
            self._qdrant = QdrantClient(
                url=config.QDRANT_URL,
                timeout=int(config.QDRANT_TIMEOUT),
            )
            try:
                self._ensure_collection()
            except Exception:
                self._qdrant = None
                raise
        return self._qdrant

    def _get_ml_services(self) -> httpx.Client:
        if not hasattr(config, "ML_SERVICES_URL"):
            raise ConfigError(
                "ML Services URL not configured",
                hint="Set sparse.url in config.yaml or use the local provider "
                "(sparse.provider=local, reranker.provider=local).",
            )
        if self._ml_services is None or self._ml_services.is_closed:
            self._ml_services = httpx.Client(
                base_url=config.ML_SERVICES_URL,
                timeout=httpx.Timeout(30.0, connect=5.0),
                limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
            )
        return self._ml_services

    # ── Collection setup ──────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        qdrant = self._get_qdrant()
        if not qdrant.collection_exists(config.COLLECTION_NAME):
            vectors_config: dict[str, VectorParams] = {
                "dense": VectorParams(size=config.DENSE_DIM, distance=Distance.COSINE),
            }
            if config.ENABLE_CONTEXTUAL_RETRIEVAL:
                vectors_config["contextual_dense"] = VectorParams(
                    size=config.DENSE_DIM,
                    distance=Distance.COSINE,
                )
            qdrant.create_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config=vectors_config,
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(),
                        modifier=Modifier.IDF,
                    )
                },
            )
            logger.info("Created Qdrant collection: %s", config.COLLECTION_NAME)

            # Wait for collection optimizer to finish initial indexing
            import time as _time

            for _attempt in range(30):
                try:
                    info = qdrant.get_collection(config.COLLECTION_NAME)
                    if str(info.optimizer_status) == "ok":
                        break
                except Exception:
                    pass
                _time.sleep(0.5)
            else:
                logger.warning("Collection optimizer did not complete within 15s, proceeding anyway")

            # Create payload indexes for filtered searches
            for _field_name, _field_type in [
                ("source", "keyword"),
                ("content_hash", "keyword"),
                ("doc_type", "keyword"),
                ("language", "keyword"),
            ]:
                try:
                    qdrant.create_payload_index(
                        collection_name=config.COLLECTION_NAME,
                        field_name=_field_name,
                        field_schema=_field_type,  # type: ignore[arg-type]
                    )
                except Exception:
                    logger.debug("Payload index already exists for %s", _field_name)
            for _field_name in ["topics", "keywords"]:
                try:
                    qdrant.create_payload_index(
                        collection_name=config.COLLECTION_NAME,
                        field_name=_field_name,
                        field_schema="keyword",  # type: ignore[arg-type]
                    )
                except Exception:
                    logger.debug("Payload index already exists for %s", _field_name)

        # Warn if contextual retrieval is enabled but collection has no contextual_dense vector
        if config.ENABLE_CONTEXTUAL_RETRIEVAL:
            try:
                info = qdrant.get_collection(config.COLLECTION_NAME)
                vectors = info.config.params.vectors
                vectors_dict: dict[str, Any] = {}
                if vectors is not None and (isinstance(vectors, dict) or hasattr(vectors, "items")):
                    vectors_dict = {k: v for k, v in vectors.items()}
                if "contextual_dense" not in vectors_dict:
                    logger.warning(
                        "contextual_retrieval is enabled but collection '%s' has no "
                        "contextual_dense vector. Existing chunks will not benefit "
                        "from contextual search. Re-ingest affected documents.",
                        config.COLLECTION_NAME,
                    )
            except Exception:
                logger.debug("Could not check collection vectors for contextual_dense", exc_info=True)

    # ── Embedding helpers ─────────────────────────────────────────────────

    def _get_embedding_service(self) -> EmbeddingService:
        """Lazy-init the EmbeddingService singleton."""
        if self._embedding_svc is None:
            self._embedding_svc = EmbeddingService(self._embedder)
        return self._embedding_svc

    def _validate_embedding_dim(self) -> None:
        """Verify embedding dimensions match DENSE_DIM at first use."""
        if self._dim_validated:
            return
        try:
            test_vec = self._dense_embed_batch(["dimension validation test"])[0]
            if len(test_vec) != config.DENSE_DIM:
                logger.warning(
                    "Embedding dimension mismatch: model returned %d, config expects %d. "
                    "Update embedding.dimensions in config.yaml.",
                    len(test_vec),
                    config.DENSE_DIM,
                )
        except Exception as exc:
            logger.warning("Unable to validate embedding dimensions: %s", exc)
        self._dim_validated = True

    def _dense_embed_batch(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Embed a batch of texts via the EmbeddingService.

        Delegates to ``EmbeddingService.embed()`` which handles caching,
        batched transport via /api/embed, and model fallback.
        """
        return self._get_embedding_service().embed(texts, model=model)

    def _dense_embed(self, text: str) -> list[float]:
        """Embed a single text via the EmbeddingService."""
        self._validate_embedding_dim()
        return self._dense_embed_batch([text])[0]

    def _sparse_embed(self, texts: list[str]) -> list[dict[str, float]]:
        """Get sparse embeddings via configured provider (http or local)."""
        if config.SPARSE_PROVIDER in ("http", "docker") and hasattr(config, "ML_SERVICES_URL"):
            try:
                client = self._get_ml_services()
                resp = client.post("/sparse/embed", json={"texts": texts})
                resp.raise_for_status()
                return resp.json()["vectors"]
            except Exception:
                logger.warning("HTTP sparse embedding failed, falling back to local", exc_info=True)
        # Local provider (or fallback) — import fastembed on demand
        from fastembed import SparseTextEmbedding

        if not hasattr(self, "_sparse_model_local"):
            logger.info("Loading sparse model locally: %s", config.SPARSE_MODEL)
            self._sparse_model_local = SparseTextEmbedding(model_name=config.SPARSE_MODEL)
        return [{str(k): float(v) for k, v in emb.as_dict().items()} for emb in self._sparse_model_local.embed(texts)]

    def _rerank(self, query: str, documents: list[str], top_k: int = 10) -> tuple[list[float], list[int]]:
        """Rerank documents via configured provider (http, local, or ollama)."""
        if config.RERANK_PROVIDER in ("http", "docker") and hasattr(config, "ML_SERVICES_URL"):
            try:
                client = self._get_ml_services()
                resp = client.post(
                    "/rerank",
                    json={"query": query, "documents": documents, "top_k": top_k},
                )
                resp.raise_for_status()
                data = resp.json()
                return data["scores"], data["indices"]
            except Exception:
                logger.warning("HTTP reranking failed, falling back to local", exc_info=True)
        if config.RERANK_PROVIDER in ("http", "local"):
            from sentence_transformers import CrossEncoder

            if not hasattr(self, "_reranker_local"):
                logger.info("Loading reranker locally: %s", config.RERANK_MODEL)
                self._reranker_local = CrossEncoder(config.RERANK_MODEL)
            pairs = [(query, doc) for doc in documents]
            scores = self._reranker_local.predict(pairs)
            indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            top = indexed[:top_k]
            return [float(s) for _, s in top], [int(i) for i, _ in top]
        else:
            # Unknown provider — return unsorted (identity rerank)
            logger.warning("Unknown rerank provider '%s', returning unsorted", config.RERANK_PROVIDER)
            n = min(top_k, len(documents))
            return [0.0] * n, list(range(n))

    # ── Public API ────────────────────────────────────────────────────────

    def compute_file_hash(self, data: bytes) -> str:
        """Compute SHA256 hash of file content."""
        return hashlib.sha256(data).hexdigest()

    def is_already_ingested(self, source_identifier: str, content_hash: str) -> tuple[bool, int]:
        """Check if a file with the same hash is already ingested.

        Returns (already_ingested, chunk_count).
        """
        qdrant = self._get_qdrant()
        result = qdrant.scroll(
            collection_name=config.COLLECTION_NAME,
            limit=1,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="source", match=MatchValue(value=source_identifier)),
                    FieldCondition(key="content_hash", match=MatchValue(value=content_hash)),
                ]
            ),
            with_payload=["total_chunks"],
            with_vectors=False,
        )
        points, _ = result
        if points:
            return True, (points[0].payload or {}).get("total_chunks", 0)
        return False, 0

    def source_exists(self, source_identifier: str) -> tuple[bool, int, float | None, int | None]:
        """Check if a source exists in Qdrant (Phase 1 check).

        Returns (exists, chunk_count, file_mtime, file_size).
        Only the first matching point is queried for metadata.
        """
        qdrant = self._get_qdrant()
        result = qdrant.scroll(
            collection_name=config.COLLECTION_NAME,
            limit=1,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="source", match=MatchValue(value=source_identifier)),
                ]
            ),
            with_payload=["total_chunks", "file_mtime", "file_size"],
            with_vectors=False,
        )
        points, _ = result
        if points:
            payload = points[0].payload or {}
            return True, payload.get("total_chunks", 0), payload.get("file_mtime"), payload.get("file_size")
        return False, 0, None, None

    def check_unmodified_local(self, source_identifier: str) -> tuple[bool, int]:
        """Phase 2 check: is a local file unchanged since last ingestion?

        Compares current os.stat mtime+size against stored payload values.
        Returns (can_skip, chunk_count). If any stat call fails, returns False.
        """
        exists, chunk_count, stored_mtime, stored_size = self.source_exists(source_identifier)
        if not exists:
            return False, 0
        try:
            st = os.stat(source_identifier)
            if (
                stored_mtime is not None
                and stored_size is not None
                and abs(st.st_mtime - stored_mtime) < 1.0
                and st.st_size == stored_size
            ):
                return True, chunk_count
        except OSError:
            pass
        return False, 0

    def _ingest_chunks(
        self,
        raw_chunks: list[dict[str, Any]],
        source_identifier: str,
        metadata: dict[str, Any] | None = None,
        content_hash: str = "",
        progress_cb: Callable[[str, int], None] | None = None,
        document_text: str = "",
    ) -> int:
        """Common ingestion logic for chunks. Returns number of chunks ingested.

        Handles: dedup, contextual retrieval, metadata extraction, embedding, Qdrant upsert.
        """

        def _progress(msg: str, pct: int) -> None:
            if progress_cb:
                progress_cb(msg, pct)
            logger.info("ingest [%d%%] %s", pct, msg)
            self._record_stage(source_identifier, pct)

        # Remove exact duplicate chunks within the document
        from memex.engine.ingestion.hashing import dedup_chunks

        raw_chunks = dedup_chunks(raw_chunks)
        if not raw_chunks:
            raise ValueError("No valid text chunks after deduplication.")

        # ── Contextual retrieval ──────────────────────────────────────────
        ctx_gen: ContextGenerator | None = None
        document_summary = ""
        if config.ENABLE_CONTEXTUAL_RETRIEVAL:
            ctx_gen = ContextGenerator(self._llm)
            if config.CONTEXT_STRATEGY == "summary":
                _progress("Generating document summary...", 71)
                try:
                    document_summary = ctx_gen.generate_document_summary(document_text or "")
                except Exception:
                    logger.warning("Document summary generation failed, falling back to header strategy", exc_info=True)
            _progress("Adding context to chunks...", 73)
            raw_chunks = ctx_gen.enrich_chunks(raw_chunks, document_summary=document_summary)

        # ── Metadata extraction ──────────────────────────────────────────
        metadata_extractor: MetadataExtractor | None = None
        if config.ENABLE_METADATA_EXTRACTION:
            metadata_extractor = MetadataExtractor(self._llm)
            _progress("Extracting metadata...", 74)
            batch_meta = metadata_extractor.extract_batch(
                chunks=raw_chunks,
                document_text=document_text,
                source_identifier=source_identifier,
            )
            for chunk, meta in zip(raw_chunks, batch_meta, strict=True):
                chunk["metadata"] = meta

        _progress(f"Generating embeddings ({len(raw_chunks)} chunks)...", 75)
        chunk_texts = [c["content"] for c in raw_chunks]
        raw_texts = [strip_context_prefix(c["content"]) for c in raw_chunks]

        # Dense (raw), sparse (enriched), and contextual dense (enriched) are
        # independent — run all three concurrently to minimise latency
        import concurrent.futures

        contextual_vecs: list[list[float]] | None = None
        max_workers = 3 if config.ENABLE_CONTEXTUAL_RETRIEVAL else 2
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            dense_future = pool.submit(self._dense_embed_batch, raw_texts)
            sparse_future = pool.submit(self._sparse_embed, chunk_texts)
            if config.ENABLE_CONTEXTUAL_RETRIEVAL:
                contextual_future = pool.submit(self._dense_embed_batch, chunk_texts)
            else:
                contextual_future = None

            dense_vecs = dense_future.result()
            sparse_vecs = sparse_future.result()
            if contextual_future is not None:
                contextual_vecs = contextual_future.result()

        _progress("Storing in Qdrant...", 90)
        now = datetime.now(UTC).isoformat()
        base_meta = metadata or {}

        # Capture file stat if source is a local file
        file_mtime: float | None = None
        file_size: int | None = None
        if os.path.isfile(source_identifier):
            try:
                st = os.stat(source_identifier)
                file_mtime = st.st_mtime
                file_size = st.st_size
            except OSError:
                pass

        points: list[PointStruct] = []
        for idx, (chunk, dense_vec, sparse_dict) in enumerate(zip(raw_chunks, dense_vecs, sparse_vecs, strict=True)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source_identifier}_{idx}"))
            point_meta = {
                "source": source_identifier,
                "chunk_index": idx,
                "total_chunks": len(raw_chunks),
                "content": chunk["content"],
                "section_header": chunk.get("section_header", ""),
                "context_prefix": chunk.get("context_prefix", ""),
                "ingested_at": now,
                "content_hash": content_hash,
                **(chunk.get("metadata", {})),
                **base_meta,
            }
            if file_mtime is not None:
                point_meta["file_mtime"] = file_mtime
            if file_size is not None:
                point_meta["file_size"] = file_size

            # Clear stale metadata when extraction is disabled
            if not config.ENABLE_METADATA_EXTRACTION:
                point_meta.setdefault("doc_type", "")
                point_meta.setdefault("topics", [])
                point_meta.setdefault("language", "")
                point_meta.setdefault("keywords", [])
                point_meta.setdefault("entities", {})

            vectors: dict[str, Any] = {
                "dense": dense_vec,
                "sparse": SparseVector(
                    indices=[int(k) for k in sparse_dict],
                    values=list(sparse_dict.values()),
                ),
            }
            if contextual_vecs is not None:
                vectors["contextual_dense"] = contextual_vecs[idx]

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vectors,
                    payload=point_meta,
                )
            )

        qdrant = self._get_qdrant()
        try:
            for i in range(0, len(points), config.EMBED_BATCH_SIZE):
                qdrant.upsert(
                    collection_name=config.COLLECTION_NAME,
                    points=points[i : i + config.EMBED_BATCH_SIZE],
                )
        except Exception:
            logger.error(
                "Write failed for '%s', rolling back by content_hash=%s",
                source_identifier,
                content_hash,
            )
            try:
                qdrant.delete(
                    collection_name=config.COLLECTION_NAME,
                    points_selector=Filter(
                        must=[FieldCondition(key="content_hash", match=MatchValue(value=content_hash))],
                    ),
                )
                logger.info("Rolled back chunks for content_hash=%s", content_hash)
            except Exception as rollback_exc:
                logger.error("Rollback also failed: %s", rollback_exc)
            raise

        logger.info("Ingested %d chunks for '%s'", len(points), source_identifier)
        return len(points)

    def ingest_text(
        self,
        text: str,
        source_identifier: str,
        metadata: dict[str, Any] | None = None,
        content_hash: str = "",
        progress_cb: Callable[[str, int], None] | None = None,
    ) -> int:
        """Chunk, embed, and upsert into Qdrant. Returns number of chunks.

        When ``CHUNK_STRATEGY`` is ``hybrid``, uses the Docling Serve chunking
        API for structure-aware chunking. Falls back to legacy recursive/fixed
        chunking when the API is unavailable.
        """

        def _progress(msg: str, pct: int) -> None:
            if progress_cb:
                progress_cb(msg, pct)
            logger.info("ingest [%d%%] %s", pct, msg)

        _progress("Chunking document...", 70)
        raw_chunks = create_chunks(text=text, source_identifier=source_identifier)
        if not raw_chunks:
            raise ValueError("No valid text chunks to ingest.")

        return self._ingest_chunks(
            raw_chunks=raw_chunks,
            source_identifier=source_identifier,
            metadata=metadata,
            content_hash=content_hash,
            progress_cb=progress_cb,
            document_text=text,
        )

    def ingest_prechunked(
        self,
        chunks: list[dict[str, Any]],
        markdown: str,
        source_identifier: str,
        metadata: dict[str, Any] | None = None,
        content_hash: str = "",
        progress_cb: Callable[[str, int], None] | None = None,
    ) -> int:
        """Ingest pre-chunked data, skipping the create_chunks() call.

        Used when chunking was done externally (e.g., by HybridChunker in a
        single Docling API call). Skips the double Docling call problem.

        Embeddings are stored to Qdrant IMMEDIATELY at the end of this call —
        no batching across files. Each file's vectors are persisted as soon as
        all stages complete for that file.
        """
        if not chunks:
            raise ValueError("No chunks to ingest.")

        raw_chunks = [c for c in chunks if len(c.get("content", "").strip()) >= config.MIN_CHUNK_LEN]
        if not raw_chunks:
            raise ValueError("No chunks above MIN_CHUNK_LEN after filtering.")

        return self._ingest_chunks(
            raw_chunks=raw_chunks,
            source_identifier=source_identifier,
            metadata=metadata,
            content_hash=content_hash,
            progress_cb=progress_cb,
            document_text=markdown,
        )

    def _record_stage(self, source_identifier: str, pct: int) -> None:
        """Record the pipeline stage for a source in the file status store.

        Maps the ingest progress percentage to a PipelineStage and writes it
        as a ``processing`` self-loop. Failures here are non-fatal — status
        tracking must never break ingestion.
        """
        if pct <= 72:
            stage = PipelineStage.CONVERTING
        elif pct <= 74:
            stage = PipelineStage.CONTEXT
        elif pct <= 76:
            stage = PipelineStage.METADATA
        elif pct <= 89:
            stage = PipelineStage.EMBEDDING
        else:
            stage = PipelineStage.STORING
        try:
            from memex.engine.ingestion.status import FileStatusStore

            store = FileStatusStore(self._get_qdrant())
            store.update_stage(source_identifier, stage)
        except Exception:
            logger.debug("Status stage recording skipped for %s", source_identifier, exc_info=True)

    def _build_search_filter(
        self,
        source_filter: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> Filter | None:
        """Build Qdrant filter from source and metadata constraints."""
        filter_conditions: list[FieldCondition] = []
        if source_filter:
            filter_conditions.append(FieldCondition(key="source", match=MatchValue(value=source_filter)))
        if metadata_filter:
            for key, value in metadata_filter.items():
                if isinstance(value, list):
                    filter_conditions.append(FieldCondition(key=key, match=MatchAny(any=value)))  # type: ignore[arg-type]
                else:
                    filter_conditions.append(FieldCondition(key=key, match=MatchValue(value=str(value).lower())))
        return Filter(must=filter_conditions) if filter_conditions else None  # type: ignore[arg-type]

    def _get_dense_vector_name(self, use_contextual_search: bool | None = None) -> str:
        """Determine which dense vector to search based on config."""
        if use_contextual_search is not None:
            effective_contextual = use_contextual_search
        else:
            effective_contextual = config.ENABLE_CONTEXTUAL_RETRIEVAL
        return "contextual_dense" if effective_contextual else "dense"

    def _hit_to_result(self, hit: Any, dense_vector_name: str = "dense") -> dict[str, Any]:
        """Convert a Qdrant hit to a result dict."""
        doc_id = str(hit.id)
        payload = hit.payload or {}
        score = hit.score if hit.score else 0.0
        return {
            "id": doc_id,
            "source": payload.get("source", ""),
            "content": payload.get("content", ""),
            "section_header": payload.get("section_header", ""),
            "context_prefix": payload.get("context_prefix", ""),
            "dense_score": score,
            "sparse_score": 0.0,
            "doc_type": payload.get("doc_type", ""),
            "topics": payload.get("topics", []),
            "language": payload.get("language", ""),
            "keywords": payload.get("keywords", []),
            "entities": payload.get("entities", {}),
            "dates": payload.get("entities", {}).get("dates", []),
            "structural": payload.get("structural", {}),
        }

    def _rerank_results(self, query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply reranking to results. Returns sorted results."""
        if not results or not config.ENABLE_RERANKING:
            return results
        t_rerank = time.monotonic()
        contents = [item["content"] for item in results]
        try:
            scores, indices = self._rerank(query, contents, top_k=len(contents))
            for score, idx in zip(scores, indices, strict=False):
                if idx < len(results):
                    results[idx]["rerank_score"] = float(score)
            results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        except Exception:
            logger.warning("Reranking failed, skipping rerank step", exc_info=True)
        _record_eval_timing("rerank", (time.monotonic() - t_rerank) * 1000)
        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = True,
        source_filter: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        expanded_query: ExpandedQuery | None = None,
        use_contextual_search: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Dense + sparse hybrid search with RRF fusion and optional reranking.

        When *expanded_query* is provided, additional dense searches are performed
        using the HyDE vector and/or paraphrase vectors, then merged via RRF.

        When *use_contextual_search* is True and contextual retrieval is enabled,
        the search uses the ``contextual_dense`` vector instead of ``dense``.

        *metadata_filter* is a dict of key→value constraints applied to the
        Qdrant payload. Lists are treated as multi-value (MatchAny).
        """
        from memex.engine.utils.cache import cache_search_results, get_cached_search_results

        cached = get_cached_search_results(query, top_k, source_filter, metadata_filter)
        if cached is not None:
            _record_eval_timing("search_cache_hit", 0.0)
            return cached

        t_search_start = time.monotonic()

        dense_query = query
        if expanded_query and expanded_query.rewritten:
            dense_query = expanded_query.rewritten

        t_embed = time.monotonic()
        query_dense = self._dense_embed(dense_query)
        sparse_vec = self._sparse_embed([query])[0]
        q_indices = [int(k) for k in sparse_vec]
        q_values = list(sparse_vec.values())
        _record_eval_timing("embed_query", (time.monotonic() - t_embed) * 1000)

        qdrant = self._get_qdrant()
        candidate_k = min(config.SEARCH_TOP_K, 50)

        dense_vector_name = self._get_dense_vector_name(use_contextual_search)
        qdrant_filter = self._build_search_filter(source_filter, metadata_filter)

        # ── Prepare all search tasks ────────────────────────────────────────
        import concurrent.futures

        # Pre-compute paraphrase embeddings if multi-query expansion is active
        para_vecs: list[list[float]] = []
        if expanded_query and expanded_query.paraphrases:
            try:
                para_vecs = self._dense_embed_batch(expanded_query.paraphrases)
            except Exception:
                logger.warning("Paraphrase batch embedding failed, skipping", exc_info=True)

        # ── Launch all independent searches in parallel ─────────────────────
        t_qdrant = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            # Primary searches: dense + sparse
            dense_future = pool.submit(
                qdrant.query_points,
                collection_name=config.COLLECTION_NAME,
                query=query_dense,
                using=dense_vector_name,
                limit=candidate_k,
                query_filter=qdrant_filter,
            )
            sparse_future = pool.submit(
                qdrant.query_points,
                collection_name=config.COLLECTION_NAME,
                query=SparseVector(indices=q_indices, values=q_values),
                using="sparse",
                limit=candidate_k,
                query_filter=qdrant_filter,
            )

            # HyDE search (if expansion enabled)
            hyde_future = None
            if expanded_query and expanded_query.hyde_vector:
                hyde_future = pool.submit(
                    qdrant.query_points,
                    collection_name=config.COLLECTION_NAME,
                    query=expanded_query.hyde_vector,
                    using=dense_vector_name,
                    limit=candidate_k,
                    query_filter=qdrant_filter,
                )

            # Multi-query paraphrase searches (if expansion enabled)
            para_futures: list = []
            if para_vecs:
                for para_dense in para_vecs:
                    para_futures.append(
                        pool.submit(
                            qdrant.query_points,
                            collection_name=config.COLLECTION_NAME,
                            query=para_dense,
                            using=dense_vector_name,
                            limit=candidate_k,
                            query_filter=qdrant_filter,
                        )
                    )

            # Collect all results
            dense_hits = dense_future.result().points
            sparse_hits = sparse_future.result().points

            hyde_hits: list = []
            if hyde_future is not None:
                try:
                    hyde_hits = hyde_future.result().points
                except Exception:
                    logger.warning("HyDE dense search failed, skipping", exc_info=True)

            para_hits_all: list[list] = []
            for f in para_futures:
                try:
                    para_hits_all.append(f.result().points)
                except Exception:
                    para_hits_all.append([])

        _record_eval_timing("dense_search", 0)
        _record_eval_timing("sparse_search", 0)
        _record_eval_timing("all_qdrant_searches", (time.monotonic() - t_qdrant) * 1000)

        # ── Reciprocal Rank Fusion ────────────────────────────────────────
        k = 60  # RRF constant (standard default from original paper)
        rrf_scores: dict[str, float] = {}
        result_map: dict[str, dict[str, Any]] = {}

        def _merge_hits(hits: list[Any], rrf_offset: int = 0) -> None:
            for rank, hit in enumerate(hits):
                doc_id = str(hit.id)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + rrf_offset + 1)
                if doc_id not in result_map:
                    result_map[doc_id] = self._hit_to_result(hit, dense_vector_name)

        _merge_hits(dense_hits)
        _merge_hits(sparse_hits, rrf_offset=len(dense_hits))

        # Merge HyDE results (already searched in parallel above)
        hyde_offset = len(dense_hits) + len(sparse_hits)
        if hyde_hits:
            _merge_hits(hyde_hits, rrf_offset=hyde_offset)

        # Merge multi-query paraphrase results (already searched in parallel above)
        if para_hits_all:
            para_offset = hyde_offset + (len(hyde_hits) if hyde_hits else candidate_k)
            for idx, para_hits in enumerate(para_hits_all):
                if para_hits:
                    _merge_hits(para_hits, rrf_offset=para_offset + idx * candidate_k)

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        results: list[dict[str, Any]] = []
        for doc_id in sorted_ids:
            entry = result_map[doc_id]
            entry["rrf_score"] = rrf_scores[doc_id]
            results.append(entry)

        # ── Reranking ─────────────────────────────────────────────────────
        if rerank:
            results = self._rerank_results(query, results)

        final_results = results[:top_k]
        _record_eval_timing("total_search", (time.monotonic() - t_search_start) * 1000)
        cache_search_results(query, top_k, source_filter, final_results, metadata_filter)
        return final_results

    def mmr_search(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        rerank: bool = True,
        source_filter: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        use_contextual_search: bool | None = None,
    ) -> list[dict[str, Any]]:
        """MMR search: dense similarity + diversity-aware selection.

        Fetches *fetch_k* candidates via dense search, then applies MMR
        selection to return *top_k* diverse results. Optionally reranks
        the MMR-selected results.
        """
        from memex.engine.retrieval.mmr import mmr_select

        t_search_start = time.monotonic()

        t_embed = time.monotonic()
        query_dense = self._dense_embed(query)
        _record_eval_timing("embed_query", (time.monotonic() - t_embed) * 1000)

        qdrant = self._get_qdrant()

        dense_vector_name = self._get_dense_vector_name(use_contextual_search)
        qdrant_filter = self._build_search_filter(source_filter, metadata_filter)

        # Dense search for fetch_k candidates
        t_qdrant = time.monotonic()
        dense_hits = qdrant.query_points(
            collection_name=config.COLLECTION_NAME,
            query=query_dense,
            using=dense_vector_name,
            limit=fetch_k,
            query_filter=qdrant_filter,
        ).points
        _record_eval_timing("dense_search", (time.monotonic() - t_qdrant) * 1000)

        if not dense_hits:
            _record_eval_timing("total_search", (time.monotonic() - t_search_start) * 1000)
            return []

        # Build candidate list with embeddings for MMR
        candidate_embeddings: list[list[float]] = []
        candidate_scores: list[float] = []
        result_map: dict[str, dict[str, Any]] = {}

        for hit in dense_hits:
            doc_id = str(hit.id)
            score = hit.score if hit.score else 0.0

            # Retrieve vector for MMR diversity computation
            point = qdrant.retrieve(
                collection_name=config.COLLECTION_NAME,
                ids=[hit.id],
                with_vectors=True,
                with_payload=False,
            )
            if point:
                vec_payload = point[0].vector
                if isinstance(vec_payload, dict):
                    vec = vec_payload.get(dense_vector_name, [])
                else:
                    vec = vec_payload if vec_payload else []
                candidate_embeddings.append(vec)  # type: ignore[arg-type]
            else:
                candidate_embeddings.append([])

            candidate_scores.append(score)
            result_map[doc_id] = self._hit_to_result(hit, dense_vector_name)

        # MMR selection
        t_mmr = time.monotonic()
        selected_indices = mmr_select(
            query_embedding=query_dense,
            candidate_embeddings=candidate_embeddings,
            candidate_scores=candidate_scores,
            top_k=top_k,
            lambda_mult=lambda_mult,
        )
        _record_eval_timing("mmr_select", (time.monotonic() - t_mmr) * 1000)

        # Build results from MMR selection
        all_ids = [str(h.id) for h in dense_hits]
        results: list[dict[str, Any]] = []
        for idx in selected_indices:
            if idx < len(all_ids):
                doc_id = all_ids[idx]
                entry = result_map[doc_id]
                entry["dense_score"] = candidate_scores[idx]
                results.append(entry)

        # Optional reranking
        if rerank:
            results = self._rerank_results(query, results)

        final_results = results[:top_k]
        _record_eval_timing("total_search", (time.monotonic() - t_search_start) * 1000)
        return final_results

    def list_documents(self) -> list[dict[str, Any]]:
        """List all unique documents with chunk counts and metadata.

        Uses minimal payload retrieval to avoid loading full content.
        For large collections (>100K chunks), this is O(N) but with small constant.
        """
        qdrant = self._get_qdrant()

        sources: dict[str, dict[str, Any]] = {}
        offset = None
        while True:
            result = qdrant.scroll(
                collection_name=config.COLLECTION_NAME,
                limit=500,
                offset=offset,
                with_payload=[
                    "source",
                    "total_chunks",
                    "ingested_at",
                    "doc_type",
                    "language",
                ],
                with_vectors=False,
            )
            points, next_offset = result
            for point in points:
                payload = point.payload or {}
                src = payload.get("source", "unknown")
                if src not in sources:
                    sources[src] = {
                        "source": src,
                        "chunk_count": 0,
                        "total_chunks": payload.get("total_chunks", 0),
                        "ingested_at": payload.get("ingested_at", ""),
                        "doc_type": payload.get("doc_type", ""),
                        "language": payload.get("language", ""),
                    }
                sources[src]["chunk_count"] += 1

            if next_offset is None:
                break
            offset = next_offset

        docs = list(sources.values())
        return sorted(docs, key=lambda x: x["ingested_at"], reverse=True)

    def get_collection_stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        qdrant = self._get_qdrant()
        info = qdrant.get_collection(config.COLLECTION_NAME)
        vectors_count = getattr(info, "vectors_count", None) or info.points_count or 0
        return {
            "collection_name": config.COLLECTION_NAME,
            "total_points": info.points_count or 0,
            "total_vectors": vectors_count,
            "status": str(info.status),
            "optimizer_status": str(info.optimizer_status),
            "config": {
                "dense_dim": config.DENSE_DIM,
                "distance": "cosine",
                "sparse_enabled": True,
            },
        }

    def delete_by_source(self, source_identifier: str) -> bool:
        """Delete all chunks whose source matches source_identifier."""
        from memex.engine.utils.cache import invalidate_for_document

        qdrant = self._get_qdrant()
        qdrant.delete(
            collection_name=config.COLLECTION_NAME,
            points_selector=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_identifier))]),
        )
        invalidate_for_document(source_identifier)
        logger.info("Deleted chunks for source: %s", source_identifier)
        return True

    def get_chunker_status(self) -> dict[str, Any]:
        """Return chunker configuration and availability."""
        try:
            from memex.engine.ingestion.splitter import is_hybrid_chunker_available

            hybrid_available = is_hybrid_chunker_available()
        except Exception:
            hybrid_available = False
        active_chunker = (
            "Docling HybridChunker"
            if (config.CHUNK_STRATEGY == "hybrid" and hybrid_available)
            else config.CHUNK_STRATEGY.title()
        )
        result = {
            "strategy": config.CHUNK_STRATEGY,
            "chunk_size": config.CHUNK_SIZE,
            "hybrid_available": hybrid_available,
            "active_chunker": active_chunker,
            "merge_peers": config.CHUNK_MERGE_PEERS,
        }
        if active_chunker != "Docling HybridChunker":
            result["chunk_overlap"] = config.CHUNK_OVERLAP
        return result

    def close(self) -> None:
        """Release HTTP clients and Qdrant connection."""
        if self._ml_services is not None and not self._ml_services.is_closed:
            self._ml_services.close()
            self._ml_services = None
        if self._qdrant is not None:
            with contextlib.suppress(Exception):
                self._qdrant.close()
            self._qdrant = None
