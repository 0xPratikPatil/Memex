"""Maximal Marginal Relevance search for result diversity."""

from __future__ import annotations

import logging
import math

log = logging.getLogger(__name__)


def mmr_select(
    query_embedding: list[float],
    candidate_embeddings: list[list[float]],
    candidate_scores: list[float],
    top_k: int,
    lambda_mult: float = 0.5,
) -> list[int]:
    """Select documents using MMR.

    Args:
        query_embedding: Query vector
        candidate_embeddings: List of candidate vectors
        candidate_scores: Similarity scores for each candidate
        top_k: Number of results to return
        lambda_mult: Balance between relevance (1.0) and diversity (0.0)

    Returns:
        Indices of selected documents in MMR order
    """
    n = len(candidate_embeddings)
    if n == 0:
        return []
    top_k = min(top_k, n)
    if top_k <= 0:
        return []

    # Try numpy path for performance
    try:
        import numpy as np

        query_vec = np.array(query_embedding, dtype=np.float32)
        cand_matrix = np.array(candidate_embeddings, dtype=np.float32)
        scores = np.array(candidate_scores, dtype=np.float32)

        # Compute query-candidate similarities (reuse provided scores if dims match)
        if len(scores) == n:
            query_sims = scores
        else:
            q_norm = np.linalg.norm(query_vec)
            if q_norm == 0.0:
                query_sims = np.zeros(n, dtype=np.float32)
            else:
                c_norms = np.linalg.norm(cand_matrix, axis=1)
                c_norms = np.where(c_norms == 0.0, 1.0, c_norms)
                query_sims = (cand_matrix @ query_vec) / (c_norms * q_norm)

        # Precompute pairwise cosine similarity matrix
        c_norms = np.linalg.norm(cand_matrix, axis=1, keepdims=True)
        c_norms = np.where(c_norms == 0.0, 1.0, c_norms)
        normalized = cand_matrix / c_norms
        sim_matrix = normalized @ normalized.T

        selected: list[int] = []
        remaining = set(range(n))

        for _ in range(top_k):
            best_idx = -1
            best_mmr = -math.inf

            for i in remaining:
                max_sim = float(sim_matrix[i, selected].max()) if selected else 0.0

                mmr_val = lambda_mult * float(query_sims[i]) - (1.0 - lambda_mult) * max_sim
                if mmr_val > best_mmr:
                    best_mmr = mmr_val
                    best_idx = i

            if best_idx == -1:
                break
            selected.append(best_idx)
            remaining.discard(best_idx)

        return selected

    except ImportError:
        log.debug("numpy not available, using pure-Python MMR fallback")

    # Pure-Python fallback
    _query_sims = _compute_query_sims(query_embedding, candidate_embeddings, candidate_scores)
    _sim_matrix = _pairwise_sim_matrix(candidate_embeddings)

    _selected: list[int] = []
    _remaining = set(range(n))

    for _ in range(top_k):
        best_idx = -1
        best_mmr = -math.inf

        for i in _remaining:
            max_sim = max((_sim_matrix[i][j] for j in _selected), default=0.0)
            mmr_val = lambda_mult * _query_sims[i] - (1.0 - lambda_mult) * max_sim
            if mmr_val > best_mmr:
                best_mmr = mmr_val
                best_idx = i

        if best_idx == -1:
            break
        _selected.append(best_idx)
        _remaining.discard(best_idx)

    return _selected


def _compute_query_sims(
    query: list[float],
    candidates: list[list[float]],
    scores: list[float],
) -> list[float]:
    """Compute cosine similarity between query and each candidate.

    Uses provided scores if they represent the correct similarities,
    otherwise computes from scratch.
    """
    if len(scores) == len(candidates):
        return list(scores)

    q_norm = math.sqrt(sum(x * x for x in query))
    if q_norm == 0.0:
        return [0.0] * len(candidates)

    result = []
    for vec in candidates:
        dot = sum(x * y for x, y in zip(query, vec, strict=False))
        v_norm = math.sqrt(sum(x * x for x in vec))
        result.append(dot / (q_norm * v_norm) if v_norm != 0.0 else 0.0)
    return result


def _pairwise_sim_matrix(vectors: list[list[float]]) -> list[list[float]]:
    """Compute NxN pairwise cosine similarity matrix (pure Python)."""
    n = len(vectors)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            s = _cosine_sim(vectors[i], vectors[j])
            matrix[i][j] = s
            matrix[j][i] = s
    return matrix


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
