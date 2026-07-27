"""ML Services — Sparse embeddings + Reranker (multi-provider).

Runs inside Docker with GPU access. MCP connects via HTTP.

Supported reranker types (set RERANK_TYPE env var):
  cross-encoder  — sentence-transformers CrossEncoder (bge-reranker-base, mxbai-rerank, etc.)
  causal-lm      — Qwen3-Reranker style: P("yes") via chat template
  ollama         — proxy to Ollama /api/embed for reranking (future)
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
_rerank_type = "cross-encoder"

RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_TYPE = os.getenv("RERANK_TYPE", "cross-encoder")
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


# ── Causal-LM Reranker (Qwen3-Reranker) ────────────────────────────────


class CausalLMReranker:
    """Reranker using causal-LM P("yes") scoring.

    Used by Qwen3-Reranker models. Formats input as a chat template,
    then extracts the logit for the "yes" token as relevance score.
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading causal-LM reranker: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

        # Find the "yes" token id
        self._yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        if self._yes_token_id == self.tokenizer.unk_token_id:
            # Fallback: try "True" or first token
            self._yes_token_id = self.tokenizer.convert_tokens_to_ids("True")

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score query-document pairs using P("yes")."""
        import torch

        scores = []
        for query, doc in pairs:
            messages = [
                {
                    "role": "system",
                    "content": "Determine whether the document is relevant to the query. "
                    "Output only 'yes' or 'no'.",
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\nDocument: {doc}\n\n"
                    "Is this document relevant to the query?",
                },
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Get the logit for the "yes" token at the last position
                logits = outputs.logits[:, -1, :]
                yes_logit = logits[0, self._yes_token_id].float()
                score = torch.sigmoid(yes_logit).item()
                scores.append(score)
        return scores


# ── Startup / shutdown ────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _sparse_model, _reranker, _rerank_type
    logger.info("Loading sparse model: %s", SPARSE_MODEL)
    from fastembed import SparseTextEmbedding

    _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    # Warm-up with a dummy embed so first real request is fast
    _ = list(_sparse_model.embed(["warmup"]))
    logger.info("Sparse model loaded")

    _rerank_type = RERANK_TYPE.lower()
    logger.info("Loading reranker: %s (type=%s)", RERANK_MODEL, _rerank_type)

    if _rerank_type == "causal-lm":
        _reranker = CausalLMReranker(RERANK_MODEL, device="cuda")
    elif _rerank_type == "cross-encoder":
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANK_MODEL, device="cuda")
    else:
        raise ValueError(f"Unknown RERANK_TYPE: {_rerank_type}. Use 'cross-encoder' or 'causal-lm'.")

    logger.info("Reranker loaded")

    yield
    logger.info("Shutting down")


app = FastAPI(title="Memex ML Services", lifespan=lifespan)


# ── Endpoints ─────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok", "sparse_model": SPARSE_MODEL, "reranker_model": RERANK_MODEL, "rerank_type": _rerank_type}


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
    top = indexed[: req.top_k]
    return RerankResponse(
        scores=[float(s) for _, s in top],
        indices=[int(i) for i, _ in top],
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5002)
