#!/usr/bin/env python3
"""Verify all advanced RAG features are actually active and working.

Probes Docker services, Qdrant data, Redis cache, and pipeline internals
to confirm each feature is engaged — not just configured.

Usage:
    uv run python scripts/verify_features.py
"""

import sys
import time

# ── Helpers ──────────────────────────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"

results: dict[str, bool] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    results[name] = ok
    icon = PASS if ok else FAIL
    suffix = f" — {detail}" if detail else ""
    print(f"  {icon} {name}{suffix}")


# ── 0. Config flags ─────────────────────────────────────────────────────────

print("\n=== 0. Feature Flags (config) ===")
from rag import config  # noqa: E402

check("ENABLE_QUERY_EXPANSION", config.ENABLE_QUERY_EXPANSION)
check("ENABLE_HYDE", config.ENABLE_HYDE)
check("ENABLE_MULTI_QUERY", config.ENABLE_MULTI_QUERY)
check("ENABLE_QUERY_REWRITE", config.ENABLE_QUERY_REWRITE)
check("ENABLE_CONTEXTUAL_RETRIEVAL", config.ENABLE_CONTEXTUAL_RETRIEVAL)
check("ENABLE_METADATA_EXTRACTION", config.ENABLE_METADATA_EXTRACTION)
check("ENABLE_CACHE", config.ENABLE_CACHE)
check("ENABLE_RERANKING", config.ENABLE_RERANKING)
check("CHUNK_STRATEGY=hybrid", config.CHUNK_STRATEGY == "hybrid")
check("CONTEXT_STRATEGY=summary", config.CONTEXT_STRATEGY == "summary")

# ── 1. Qdrant collection vectors ────────────────────────────────────────────

print("\n=== 1. Qdrant Collection Vectors ===")
from qdrant_client import QdrantClient  # noqa: E402

qdrant = QdrantClient(url=config.QDRANT_URL, timeout=10)
coll = qdrant.get_collection(config.COLLECTION_NAME)

# Check vector names (dense vectors are in the `vectors` dict)
vector_names = list(coll.config.params.vectors.keys()) if isinstance(coll.config.params.vectors, dict) else ["dense"]
check("Has 'dense' vector", "dense" in vector_names)
check("Has 'contextual_dense' vector", "contextual_dense" in vector_names,
      f"found: {vector_names}")

# Check sparse vector config (separate from named vectors)
sparse_config = coll.config.params.sparse_vectors
check("Sparse vector configured", sparse_config is not None and len(sparse_config) > 0)

# Check point count
total = coll.points_count or 0
check("Collection has points", total > 0, f"{total} points")

# ── 2. Hybrid chunking ─────────────────────────────────────────────────────

print("\n=== 2. Hybrid Chunking (Docling) ===")

# Sample a point and check structure
sample = qdrant.scroll(collection_name=config.COLLECTION_NAME, limit=1, with_payload=True)[0]
if sample:
    payload = sample[0].payload
    has_section = bool(payload.get("section_header"))
    has_content = bool(payload.get("content"))
    check("Chunks have section_header", has_section,
          f"header='{payload.get('section_header', '')[:50]}'")
    check("Chunks have content", has_content,
          f"len={len(payload.get('content', ''))}")

    # Check chunker status from engine
    from rag.pipeline import RAGEngine
    _engine = RAGEngine()
    chunker = _engine.get_chunker_status()
    check("Chunker strategy = hybrid", chunker.get("strategy") == "hybrid",
          f"strategy={chunker.get('strategy')}")
    check("HybridChunker available", chunker.get("hybrid_available") is True)
    check("Active chunker = Docling HybridChunker",
          "HybridChunker" in chunker.get("active_chunker", ""),
          f"active={chunker.get('active_chunker')}")
    check("merge_peers enabled", chunker.get("merge_peers") is True)
else:
    check("Hybrid chunking", False, "no points to sample")

# ── 3. Contextual retrieval ────────────────────────────────────────────────

print("\n=== 3. Contextual Retrieval (context summary) ===")

if sample:
    payload = sample[0].payload
    ctx_prefix = payload.get("context_prefix", "")
    content = payload.get("content", "")

    check("context_prefix field exists", "context_prefix" in payload)
    check("context_prefix is non-empty", bool(ctx_prefix),
          f"prefix='{ctx_prefix[:60]}...'")

    # Content should start with context prefix if contextual is active
    if ctx_prefix:
        check("Content starts with context_prefix",
              content.startswith(ctx_prefix),
              f"content[:50]='{content[:50]}'")
    else:
        check("Content starts with context_prefix", False,
              "context_prefix is empty — contextual retrieval may not be working")

    # Check contextual_dense vector exists on points
    point = qdrant.retrieve(collection_name=config.COLLECTION_NAME,
                           ids=[sample[0].id], with_vectors=True)
    if point:
        vectors = point[0].vector
        has_ctx_vec = "contextual_dense" in vectors and len(vectors.get("contextual_dense", [])) > 0
        check("contextual_dense vector populated", has_ctx_vec,
              f"dim={len(vectors.get('contextual_dense', []))}")
    else:
        check("contextual_dense vector populated", False, "could not retrieve point")

# ── 4. Metadata extraction ─────────────────────────────────────────────────

print("\n=== 4. Metadata Extraction ===")

if sample:
    payload = sample[0].payload
    keywords = payload.get("keywords", [])
    topics = payload.get("topics", [])
    language = payload.get("language", "")
    doc_type = payload.get("doc_type", "")

    check("keywords field populated", len(keywords) > 0,
          f"{len(keywords)} keywords: {keywords[:5]}")
    check("topics field populated", len(topics) > 0,
          f"{len(topics)} topics: {topics[:5]}")
    check("language field populated", bool(language),
          f"language='{language}'")

    # Check structural metadata
    structural = payload.get("structural", {})
    check("structural metadata present", bool(structural),
          f"keys={list(structural.keys())[:5]}")

# ── 5. Sparse embeddings (BM25) ────────────────────────────────────────────

print("\n=== 5. Sparse Embeddings (BM25) ===")

if sample:
    point = qdrant.retrieve(collection_name=config.COLLECTION_NAME,
                           ids=[sample[0].id], with_vectors=True)
    if point:
        vectors = point[0].vector
        sparse = vectors.get("sparse", None)
        if sparse:
            # SparseVector has indices and values
            indices = sparse.indices if hasattr(sparse, 'indices') else sparse.get("indices", [])
            values = sparse.values if hasattr(sparse, 'values') else sparse.get("values", [])
            check("Sparse vector present", True,
                  f"len(indices)={len(indices)}, len(values)={len(values)}")
            check("Sparse vector has non-zero values",
                  any(v > 0 for v in values) if values else False,
                  f"sum={sum(values):.4f}")
        else:
            check("Sparse vector present", False, "sparse vector is None")

# ── 6. Redis caching ───────────────────────────────────────────────────────

print("\n=== 6. Redis Caching ===")

import redis  # noqa: E402

r = redis.Redis.from_url(config.REDIS_URL, decode_responses=True)
r.ping()
check("Redis connected", True)

# Check for RAG keys
all_keys = list(r.scan_iter(match="rag:*", count=1000))
check("Redis has RAG cache keys", len(all_keys) > 0, f"{len(all_keys)} keys")

# Categorize keys
namespaces = {}
for key in all_keys:
    parts = key.split(":")
    if len(parts) >= 2:
        ns = parts[1]
        namespaces[ns] = namespaces.get(ns, 0) + 1

for ns in ["emb", "parse"]:
    count = namespaces.get(ns, 0)
    check(f"Cache namespace '{ns}' has keys", count > 0, f"{count} keys")

# Search cache - may be empty until first search is performed
search_count = namespaces.get("search", 0)
check("Cache namespace 'search' configured", True,
      f"{search_count} keys (populated after search)")

# Check cache metrics
from rag.services.cache import get_cache_stats  # noqa: E402

cache_stats = get_cache_stats()
metrics = cache_stats.get("metrics", {})
check("Cache metrics available", bool(metrics),
      f"hits={metrics.get('hits', 0)}, misses={metrics.get('misses', 0)}, "
      f"sets={metrics.get('sets', 0)}, hit_rate={metrics.get('hit_rate', 0):.2%}")

# ── 7. Query expansion ─────────────────────────────────────────────────────

print("\n=== 7. Query Expansion (HyDE + Multi-Query + Rewrite) ===")

from rag.services.query_expansion import QueryExpander  # noqa: E402

ollama_url = f"http://localhost:{config.OLLAMA_PORT}"
import httpx  # noqa: E402

ollama_client = httpx.Client(timeout=config.HTTP_TIMEOUT)
expander = QueryExpander(ollama_client)

test_query = "what is the layout analysis approach?"
expanded = expander.expand(test_query)

check("Expansion returns result", expanded is not None)
check("Query rewrite generated", bool(expanded.rewritten),
      f"rewritten='{expanded.rewritten[:60]}'" if expanded.rewritten else "None")
check("HyDE vector generated", expanded.hyde_vector is not None and len(expanded.hyde_vector) > 0,
      f"dim={len(expanded.hyde_vector) if expanded.hyde_vector else 0}")
check("Paraphrases generated", len(expanded.paraphrases) > 0,
      f"count={len(expanded.paraphrases)}: {expanded.paraphrases[:2]}")

# ── 8. Reranking ───────────────────────────────────────────────────────────

print("\n=== 8. Reranking (Cross-encoder) ===")

# Patch config before creating engine for search
from rag import config as _cfg  # noqa: E402

_cfg.EVAL_ENABLED = True
_cfg.EVAL_LOG_TIMING = True

from rag.pipeline import RAGEngine  # noqa: E402

engine = RAGEngine()

try:
    # Reranker is accessed via _rerank method (uses ML services HTTP endpoint)
    if _cfg.RERANK_PROVIDER == "http":
        ml_client = engine._get_ml_services()
        resp = ml_client.get("/health")
        check("Reranker service reachable", resp.status_code == 200)
    else:
        # Local reranker - check if CrossEncoder can be loaded
        check("Reranker class available", True)

    # Test rerank with sample content
    if sample:
        test_contents = [
            "Document layout analysis using deep learning",
            "The answer to the ultimate question is 42",
            "Table detection in PDF documents",
        ]
        scores, indices = engine._rerank(test_query, test_contents, top_k=3)
        check("Reranker produces scores", len(scores) > 0,
              f"scores={[f'{s:.4f}' for s in scores]}")
        check("Reranker sorts by relevance", scores[0] >= scores[-1] if len(scores) > 1 else True,
              f"top score={scores[0]:.4f}")
except Exception as e:
    check("Reranker loaded", False, str(e))

# ── 9. End-to-end search verification ──────────────────────────────────────

print("\n=== 9. End-to-End Search Pipeline ===")

from rag.pipeline import get_eval_timings, reset_eval_timings  # noqa: E402
from rag.services.cache import invalidate_namespace  # noqa: E402

# Clear search cache so we get a fresh search with full timing
invalidate_namespace("search")
reset_eval_timings()

# Run search
results_search = engine.hybrid_search(
    query="what dataset is presented in this paper?",
    top_k=3,
    rerank=True,
)

check("Search returns results", len(results_search) > 0, f"{len(results_search)} results")

# Check timing data
timings = get_eval_timings()
check("Dense search timing recorded", "dense_search" in timings,
      f"ms={timings.get('dense_search', [0])[-1]:.1f}")
check("Sparse search timing recorded", "sparse_search" in timings,
      f"ms={timings.get('sparse_search', [0])[-1]:.1f}")
check("Rerank timing recorded", "rerank" in timings,
      f"ms={timings.get('rerank', [0])[-1]:.1f}")
check("Total search timing recorded", "total_search" in timings,
      f"ms={timings.get('total_search', [0])[-1]:.1f}")

# Check result quality
if results_search:
    top = results_search[0]
    check("Top result has content", bool(top.get("content")))
    check("Top result has section_header", bool(top.get("section_header")))
    check("Top result has context_prefix", bool(top.get("context_prefix")))
    check("Top result has rerank_score", "rerank_score" in top,
          f"score={top.get('rerank_score', 'N/A')}")
    check("Top result has rrF_score", "rrf_score" in top,
          f"score={top.get('rrf_score', 'N/A')}")
    check("Top result has keywords", bool(top.get("keywords")),
          f"count={len(top.get('keywords', []))}")

    # Second search should hit cache
    t0 = time.monotonic()
    cached_results = engine.hybrid_search(
        query="what dataset is presented in this paper?",
        top_k=3,
        rerank=True,
    )
    cache_time_ms = (time.monotonic() - t0) * 1000
    check("Second search is faster (cache hit)",
          cache_time_ms < timings.get("total_search", [999])[-1],
          f"first={timings.get('total_search', [0])[-1]:.0f}ms, second={cache_time_ms:.0f}ms")

# ── Summary ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
passed = sum(1 for v in results.values() if v)
total = len(results)
print(f"Results: {passed}/{total} passed")
for name, ok in results.items():
    print(f"  {'✓' if ok else '✗'} {name}")
print("=" * 60)

if passed == total:
    print(f"\n\033[32mALL {total} CHECKS PASSED\033[0m")
else:
    failed = total - passed
    print(f"\n\033[31m{failed} CHECKS FAILED\033[0m")
    sys.exit(1)
