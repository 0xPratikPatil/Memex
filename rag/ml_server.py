"""ML Services — Sparse embeddings + Reranker (multi-provider).

Runs inside Docker with GPU access. MCP connects via HTTP.

Supported reranker types (set RERANK_TYPE env var):
  cross-encoder  — sentence-transformers CrossEncoder (bge-reranker-base, mxbai-rerank, etc.)
  causal-lm      — Qwen3-Reranker style: P("yes") via chat template
  ollama         — proxy to Ollama /api/embed for reranking (future)

If the preferred model fails to load, falls back to the *_FALLBACK model.
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

RERANK_MODEL = os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-0.6B")
RERANK_MODEL_FALLBACK = os.getenv("RERANK_MODEL_FALLBACK", "BAAI/bge-reranker-base")
RERANK_TYPE = os.getenv("RERANK_TYPE", "auto")
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
            dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

        # Find the "yes" token id
        self._yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        if self._yes_token_id == self.tokenizer.unk_token_id:
            self._yes_token_id = self.tokenizer.convert_tokens_to_ids("True")

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score query-document pairs using P("yes").

        Batches all pairs into a single forward pass by concatenating
        prompts with padding, avoiding per-pair Python loop overhead.
        """
        import torch

        if not pairs:
            return []

        # Build all prompts at once
        prompts = []
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
            prompts.append(
                self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )

        # Tokenize all prompts in one batch call
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
            yes_logits = logits[:, self._yes_token_id].float()
            scores = torch.sigmoid(yes_logits).cpu().tolist()

        return [float(s) for s in scores]


# ── Model loading with fallback ─────────────────────────────────────────


def _load_reranker(model_name: str, model_type: str, device: str = "cuda"):
    """Try loading a reranker; raise on failure so caller can try fallback."""
    if model_type == "causal-lm":
        return CausalLMReranker(model_name, device=device)
    elif model_type == "cross-encoder":
        from sentence_transformers import CrossEncoder

        return CrossEncoder(model_name, device=device)
    else:
        raise ValueError(f"Unknown RERANK_TYPE: {model_type}. Use 'cross-encoder' or 'causal-lm'.")


def _infer_rerank_type(model_name: str) -> str:
    """Infer rerank_type from model name if RERANK_TYPE env not set."""
    lower = model_name.lower()
    if "qwen3-reranker" in lower or "causal" in lower:
        return "causal-lm"
    return "cross-encoder"


# ── Startup / shutdown ────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _sparse_model, _reranker, _rerank_type

    # ── Sparse model (BM25) ──
    logger.info("Loading sparse model: %s", SPARSE_MODEL)
    from fastembed import SparseTextEmbedding

    _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    _ = list(_sparse_model.embed(["warmup"]))
    logger.info("Sparse model loaded")

    # ── Reranker with fallback ──
    _rerank_type = RERANK_TYPE.lower()
    if _rerank_type == "auto":
        _rerank_type = _infer_rerank_type(RERANK_MODEL)

    logger.info("Loading reranker: %s (type=%s)", RERANK_MODEL, _rerank_type)
    try:
        _reranker = _load_reranker(RERANK_MODEL, _rerank_type)
        logger.info("Reranker loaded: %s", RERANK_MODEL)
    except Exception as e:
        logger.warning("Failed to load reranker %s: %s", RERANK_MODEL, e)
        fallback_type = _infer_rerank_type(RERANK_MODEL_FALLBACK)
        logger.info("Falling back to: %s (type=%s)", RERANK_MODEL_FALLBACK, fallback_type)
        _reranker = _load_reranker(RERANK_MODEL_FALLBACK, fallback_type)
        _rerank_type = fallback_type
        logger.info("Fallback reranker loaded: %s", RERANK_MODEL_FALLBACK)

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
        d = {str(k): float(v) for k, v in zip(emb.indices, emb.values, strict=False)}
        vectors.append(d)
    return SparseResponse(vectors=vectors)


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    if _reranker is None:
        raise HTTPException(status_code=503, detail="Reranker not loaded")
    pairs = [(req.query, doc) for doc in req.documents]
    scores = _reranker.predict(pairs)
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top = indexed[: req.top_k]
    return RerankResponse(
        scores=[float(s) for _, s in top],
        indices=[int(i) for i, _ in top],
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5002)
