"""Cosine similarity utilities."""

from __future__ import annotations

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_similarity_matrix(vectors: list[list[float]]) -> list[list[float]]:
    """Compute pairwise cosine similarity matrix.

    Returns an N-by-N matrix where matrix[i][j] is the cosine similarity
    between vectors[i] and vectors[j].
    """
    n = len(vectors)
    # Try numpy for speed
    try:
        import numpy as np

        mat = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0.0, 1.0, norms)
        normalized = mat / norms
        sim = normalized @ normalized.T
        return sim.tolist()
    except ImportError:
        pass

    # Fallback: pure Python
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            s = cosine_similarity(vectors[i], vectors[j])
            matrix[i][j] = s
            matrix[j][i] = s
    return matrix
