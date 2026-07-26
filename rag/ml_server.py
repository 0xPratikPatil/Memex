"""ML Services — Sparse embeddings + Cross-encoder reranker.

Runs inside Docker with GPU access. MCP connects via HTTP.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml-services")

_sparse_model = None
_reranker = None

RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")


# ── Request/Response models ──────────────────────────────────────────────


class SparseRequest(BaseModel):
    texts: list[str]


class SparseResponse(BaseModel):
    vectors: list[dict[str, float]]  # [{token_id: weight}, ...]


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_k: int = 10


class RerankResponse(BaseModel):
    scores: list[float]
    indices: list[int]


# ── Startup / shutdown ────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _sparse_model, _reranker
    logger.info("Loading sparse model: %s", SPARSE_MODEL)
    from fastembed import SparseTextEmbedding

    _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    # Warm-up with a dummy embed so first real request is fast
    _ = list(_sparse_model.embed(["warmup"]))
    logger.info("Sparse model loaded")

    logger.info("Loading reranker: %s", RERANK_MODEL)
    from sentence_transformers import CrossEncoder

    _reranker = CrossEncoder(RERANK_MODEL, device="cuda")
    logger.info("Reranker loaded")

    yield
    logger.info("Shutting down")


app = FastAPI(title="Memex ML Services", lifespan=lifespan)


# ── Endpoints ─────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok", "sparse_model": SPARSE_MODEL, "reranker_model": RERANK_MODEL}


@app.post("/sparse/embed", response_model=SparseResponse)
def sparse_embed(req: SparseRequest):
    if _sparse_model is None:
        raise HTTPException(status_code=503, detail="Sparse model not loaded")
    vectors: list[dict[str, float]] = []
    for emb in _sparse_model.embed(req.texts):
        # SparseEmbedding has .indices (numpy int array) and .values (numpy float array)
        d = {str(k): float(v) for k, v in zip(emb.indices, emb.values, strict=False)}
        vectors.append(d)
    return SparseResponse(vectors=vectors)


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    if _reranker is None:
        raise HTTPException(status_code=503, detail="Reranker not loaded")
    pairs = [(req.query, doc) for doc in req.documents]
    scores = _reranker.predict(pairs)
    # Sort by score descending
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top = indexed[:req.top_k]
    return RerankResponse(
        scores=[float(s) for _, s in top],
        indices=[int(i) for i, _ in top],
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5002)
