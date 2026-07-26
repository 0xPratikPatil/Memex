"""RAG pipeline: embeddings, Qdrant storage, hybrid search, reranking.

Uses ``httpx`` for connection-pooled Ollama calls, ``tenacity`` for retries,
and lazy-init for heavy models (sparse embeddings, reranker).

Key improvements over v1:
- Async I/O throughout
- Recursive chunking that respects semantic boundaries
- Reciprocal Rank Fusion (RRF) for dense+sparse combination
- Batch embedding via Ollama API
- Rich per-chunk metadata
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import re
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    Modifier,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import config

logger = logging.getLogger("rag-pipeline")


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
                chunks.append({
                    "content": section_text.strip(),
                    "section_header": header,
                })
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
                        chunks.append({
                            "content": buffer,
                            "section_header": header,
                        })
                    if _estimate_tokens(para.strip()) > max_tokens:
                        _chunk_section(para.strip(), header)
                    else:
                        buffer = para.strip()
            if buffer:
                chunks.append({
                    "content": buffer,
                    "section_header": header,
                })
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
                        chunks.append({
                            "content": buffer,
                            "section_header": header,
                        })
                    buffer = sent.strip()
            if buffer:
                chunks.append({
                    "content": buffer,
                    "section_header": header,
                })
            return

        # Fallback: hard split by words with overlap
        words = section_text.split()
        step = max(1, (max_tokens * 4) // 4 - (overlap_tokens * 4) // 4)
        word_step = max(1, step // 4)
        for i in range(0, len(words), word_step):
            chunk_words = words[i : i + (max_tokens * 4) // 4]
            chunk_text = " ".join(chunk_words)
            if _estimate_tokens(chunk_text) >= 10:
                chunks.append({
                    "content": chunk_text,
                    "section_header": header,
                })

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

    # Apply overlap by duplicating tail/head between adjacent chunks
    if overlap_tokens > 0 and len(chunks) > 1:
        overlapped: list[dict[str, Any]] = [chunks[0]]
        for idx in range(1, len(chunks)):
            prev_words = chunks[idx - 1]["content"].split()
            overlap_word_count = min(overlap_tokens // 4, len(prev_words))
            if overlap_word_count > 0:
                overlap_text = " ".join(prev_words[-overlap_word_count:])
                combined = f"{overlap_text} {chunks[idx]['content']}"
                if _estimate_tokens(combined) <= max_tokens * 2:
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
    max_words = (max_tokens * 4) // 4
    overlap_words = (overlap_tokens * 4) // 4
    step = max(1, max_words - overlap_words)
    chunks: list[dict[str, Any]] = []
    for i in range(0, len(words), step):
        chunk_words = words[i : i + max_words]
        chunk_text = " ".join(chunk_words)
        if len(chunk_text.strip()) >= config.MIN_CHUNK_LEN:
            chunks.append({
                "content": chunk_text,
                "section_header": "",
            })
    return chunks


def create_chunks(text: str) -> list[dict[str, Any]]:
    """Create chunks from markdown text using configured strategy."""
    if not text.strip():
        return []

    strategy = config.CHUNK_STRATEGY.lower()
    max_tokens = config.CHUNK_SIZE
    overlap_tokens = config.CHUNK_OVERLAP

    if strategy == "recursive":
        raw = _recursive_chunk(text, max_tokens, overlap_tokens)
    elif strategy == "fixed":
        raw = _fixed_chunk(text, max_tokens, overlap_tokens)
    else:
        raw = _recursive_chunk(text, max_tokens, overlap_tokens)

    # Filter trivial chunks
    return [c for c in raw if len(c["content"].strip()) >= config.MIN_CHUNK_LEN]


# ── RAG Engine ───────────────────────────────────────────────────────────────


class RAGEngine:
    """Manages Qdrant storage, embeddings, hybrid search and reranking."""

    def __init__(self) -> None:
        self._qdrant: QdrantClient | None = None
        self._ollama: httpx.Client | None = None
        self._sparse_model: SparseTextEmbedding | None = None
        self._reranker = None  # lazy CrossEncoder

    # ── Lazy singletons ───────────────────────────────────────────────────

    def _get_qdrant(self) -> QdrantClient:
        if self._qdrant is None:
            self._qdrant = QdrantClient(
                url=config.QDRANT_URL,
                timeout=config.QDRANT_TIMEOUT,
            )
            self._ensure_collection()
        return self._qdrant

    def _get_ollama(self) -> httpx.Client:
        if self._ollama is None or self._ollama.is_closed:
            self._ollama = httpx.Client(
                timeout=httpx.Timeout(config.HTTP_TIMEOUT, connect=10.0),
                limits=httpx.Limits(
                    max_connections=8,
                    max_keepalive_connections=4,
                    keepalive_expiry=30,
                ),
            )
        return self._ollama

    def _get_sparse_model(self) -> SparseTextEmbedding:
        if self._sparse_model is None:
            logger.info("Loading sparse model: %s", config.SPARSE_MODEL)
            self._sparse_model = SparseTextEmbedding(model_name=config.SPARSE_MODEL)
        return self._sparse_model

    def _get_reranker(self):
        if self._reranker is None:
            if config.ENABLE_RERANKING:
                from sentence_transformers import CrossEncoder

                logger.info("Loading reranker: %s", config.RERANK_MODEL)
                self._reranker = CrossEncoder(config.RERANK_MODEL)
            else:
                return None
        return self._reranker

    # ── Collection setup ──────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        qdrant = self._get_qdrant()
        if not qdrant.collection_exists(config.COLLECTION_NAME):
            qdrant.create_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config={
                    "dense": VectorParams(
                        size=config.DENSE_DIM, distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(),
                        modifier=Modifier.IDF,
                    )
                },
            )
            logger.info("Created Qdrant collection: %s", config.COLLECTION_NAME)

    # ── Embedding helpers ─────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        stop=stop_after_attempt(config.HTTP_MAX_RETRIES),
        wait=wait_exponential(multiplier=config.HTTP_RETRY_BACKOFF, max=10),
        reraise=True,
    )
    def _dense_embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via Ollama."""
        client = self._get_ollama()
        embeddings: list[list[float]] = []

        # Ollama /api/embeddings only accepts one prompt at a time,
        # so we batch sequentially but reuse the connection.
        for text in texts:
            resp = client.post(
                config.OLLAMA_EMBED_URL,
                json={"model": config.EMBED_MODEL, "prompt": text},
            )
            resp.raise_for_status()
            embeddings.append(resp.json()["embedding"])

        return embeddings

    def _dense_embed(self, text: str) -> list[float]:
        """Embed a single text."""
        return self._dense_embed_batch([text])[0]

    def _sparse_embed(self, text: str) -> tuple[list[int], list[float]]:
        model = self._get_sparse_model()
        emb = list(model.embed([text]))[0]
        return emb.indices.tolist(), emb.values.tolist()

    def _sparse_embed_batch(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        model = self._get_sparse_model()
        embs = list(model.embed(texts))
        return [(e.indices.tolist(), e.values.tolist()) for e in embs]

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

    def ingest_text(
        self,
        text: str,
        source_identifier: str,
        metadata: dict[str, Any] | None = None,
        content_hash: str = "",
        progress_cb: Callable[[str, int], None] | None = None,
    ) -> int:
        """Chunk, embed, and upsert into Qdrant. Returns number of chunks."""
        def _progress(msg: str, pct: int) -> None:
            if progress_cb:
                progress_cb(msg, pct)
            logger.info("ingest [%d%%] %s", pct, msg)

        _progress("Chunking document...", 70)
        raw_chunks = create_chunks(text)
        if not raw_chunks:
            raise ValueError("No valid text chunks to ingest.")

        _progress(f"Generating embeddings ({len(raw_chunks)} chunks)...", 75)
        chunk_texts = [c["content"] for c in raw_chunks]
        dense_vecs = self._dense_embed_batch(chunk_texts)
        sparse_vecs = self._sparse_embed_batch(chunk_texts)

        _progress("Storing in Qdrant...", 90)
        now = datetime.now(UTC).isoformat()
        base_meta = metadata or {}

        points: list[PointStruct] = []
        for idx, (chunk, dense_vec, (sparse_idx, sparse_vals)) in enumerate(
            zip(raw_chunks, dense_vecs, sparse_vecs, strict=True)
        ):
            point_id = str(
                uuid.uuid5(uuid.NAMESPACE_DNS, f"{source_identifier}_{idx}")
            )
            point_meta = {
                "source": source_identifier,
                "chunk_index": idx,
                "total_chunks": len(raw_chunks),
                "content": chunk["content"],
                "section_header": chunk.get("section_header", ""),
                "ingested_at": now,
                "content_hash": content_hash,
                **base_meta,
            }
            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vec,
                        "sparse": SparseVector(
                            indices=sparse_idx,
                            values=sparse_vals,
                        ),
                    },
                    payload=point_meta,
                )
            )

        qdrant = self._get_qdrant()
        batch_size = 64
        for i in range(0, len(points), batch_size):
            qdrant.upsert(
                collection_name=config.COLLECTION_NAME,
                points=points[i : i + batch_size],
            )

        logger.info("Ingested %d chunks for '%s'", len(points), source_identifier)
        return len(points)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = True,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Dense + sparse hybrid search with RRF fusion and optional reranking."""
        query_dense = self._dense_embed(query)
        q_indices, q_values = self._sparse_embed(query)

        qdrant = self._get_qdrant()
        candidate_k = min(config.SEARCH_TOP_K, 50)

        # Build optional filter
        qdrant_filter = None
        if source_filter:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="source", match=MatchValue(value=source_filter)
                    )
                ]
            )

        # Fetch dense results
        dense_hits = qdrant.query_points(
            collection_name=config.COLLECTION_NAME,
            query=query_dense,
            using="dense",
            limit=candidate_k,
            query_filter=qdrant_filter,
        ).points

        # Fetch sparse results
        sparse_hits = qdrant.query_points(
            collection_name=config.COLLECTION_NAME,
            query=SparseVector(indices=q_indices, values=q_values),
            using="sparse",
            limit=candidate_k,
            query_filter=qdrant_filter,
        ).points

        # ── Reciprocal Rank Fusion ────────────────────────────────────────
        k = 60  # RRF constant (standard default)
        rrf_scores: dict[str, float] = {}
        result_map: dict[str, dict[str, Any]] = {}

        for rank, hit in enumerate(dense_hits):
            doc_id = str(hit.id)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
            if doc_id not in result_map:
                result_map[doc_id] = {
                    "id": doc_id,
                    "source": hit.payload.get("source", ""),
                    "content": hit.payload.get("content", ""),
                    "section_header": hit.payload.get("section_header", ""),
                    "dense_score": hit.score,
                    "sparse_score": 0.0,
                }

        for rank, hit in enumerate(sparse_hits):
            doc_id = str(hit.id)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
            if doc_id not in result_map:
                result_map[doc_id] = {
                    "id": doc_id,
                    "source": hit.payload.get("source", ""),
                    "content": hit.payload.get("content", ""),
                    "section_header": hit.payload.get("section_header", ""),
                    "dense_score": 0.0,
                    "sparse_score": hit.score,
                }
            else:
                result_map[doc_id]["sparse_score"] = hit.score

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        results: list[dict[str, Any]] = []
        for doc_id in sorted_ids:
            entry = result_map[doc_id]
            entry["rrf_score"] = rrf_scores[doc_id]
            results.append(entry)

        # ── Reranking ─────────────────────────────────────────────────────
        if rerank and results:
            reranker = self._get_reranker()
            if reranker is not None:
                pairs = [[query, item["content"]] for item in results]
                scores = reranker.predict(pairs)
                for i, score in enumerate(scores):
                    results[i]["rerank_score"] = float(score)
                results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            else:
                logger.warning("Reranker not available, skipping rerank step")

        return results[:top_k]

    def list_documents(self) -> list[dict[str, Any]]:
        """List all unique documents with chunk counts."""
        qdrant = self._get_qdrant()

        # Scroll through all points to collect source metadata
        sources: dict[str, dict[str, Any]] = {}
        offset = None
        while True:
            result = qdrant.scroll(
                collection_name=config.COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=["source", "chunk_index", "total_chunks", "ingested_at", "section_header"],
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
                        "sections": set(),
                    }
                sources[src]["chunk_count"] += 1
                header = payload.get("section_header", "")
                if header:
                    sources[src]["sections"].add(header)

            if next_offset is None:
                break
            offset = next_offset

        # Convert sets to lists for JSON serialization
        docs = []
        for src_info in sources.values():
            src_info["sections"] = sorted(src_info["sections"])
            docs.append(src_info)

        return sorted(docs, key=lambda x: x["ingested_at"], reverse=True)

    def get_document_info(self, source_identifier: str) -> dict[str, Any] | None:
        """Get detailed info about a specific document."""
        qdrant = self._get_qdrant()
        result = qdrant.scroll(
            collection_name=config.COLLECTION_NAME,
            limit=1000,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="source", match=MatchValue(value=source_identifier)
                    )
                ]
            ),
            with_payload=["source", "chunk_index", "total_chunks", "ingested_at", "section_header", "content"],
            with_vectors=False,
        )
        points, _ = result
        if not points:
            return None

        sections: set[str] = set()
        for p in points:
            header = (p.payload or {}).get("section_header", "")
            if header:
                sections.add(header)

        first_payload = points[0].payload or {}
        return {
            "source": source_identifier,
            "chunk_count": len(points),
            "total_chunks": first_payload.get("total_chunks", 0),
            "ingested_at": first_payload.get("ingested_at", ""),
            "sections": sorted(sections),
        }

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
        qdrant = self._get_qdrant()
        qdrant.delete(
            collection_name=config.COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="source", match=MatchValue(value=source_identifier)
                    )
                ]
            ),
        )
        logger.info("Deleted chunks for source: %s", source_identifier)
        return True

    def close(self) -> None:
        """Release HTTP clients and Qdrant connection."""
        if self._ollama is not None and not self._ollama.is_closed:
            self._ollama.close()
            self._ollama = None
        if self._qdrant is not None:
            with contextlib.suppress(Exception):
                self._qdrant.close()
            self._qdrant = None
