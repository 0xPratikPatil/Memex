"""Unit tests for MMR search and cosine similarity utilities."""

from __future__ import annotations

import math

import pytest

from memex.engine.retrieval.cosine import cosine_similarity, cosine_similarity_matrix
from memex.engine.retrieval.mmr import mmr_select

# ── Cosine similarity ────────────────────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_known_value(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        expected = (1 * 4 + 2 * 5 + 3 * 6) / (
            math.sqrt(1 + 4 + 9) * math.sqrt(16 + 25 + 36)
        )
        assert cosine_similarity(a, b) == pytest.approx(expected)

    def test_zero_vector(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == pytest.approx(0.0)

    def test_empty_vectors(self) -> None:
        assert cosine_similarity([], []) == 0.0


class TestCosineSimilarityMatrix:
    def test_single_vector(self) -> None:
        mat = cosine_similarity_matrix([[1.0, 2.0]])
        assert mat[0][0] == pytest.approx(1.0)

    def test_diagonal_is_one(self) -> None:
        vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        mat = cosine_similarity_matrix(vectors)
        for i in range(3):
            assert mat[i][i] == pytest.approx(1.0)

    def test_symmetric(self) -> None:
        vectors = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        mat = cosine_similarity_matrix(vectors)
        for i in range(3):
            for j in range(3):
                assert mat[i][j] == pytest.approx(mat[j][i])

    def test_orthogonal(self) -> None:
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        mat = cosine_similarity_matrix(vectors)
        assert mat[0][1] == pytest.approx(0.0)
        assert mat[1][0] == pytest.approx(0.0)


# ── MMR select ───────────────────────────────────────────────────────────────


class TestMmrSelect:
    def test_basic_selection(self) -> None:
        query = [1.0, 0.0]
        candidates = [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]
        scores = [1.0, 0.8, 0.0]
        result = mmr_select(query, candidates, scores, top_k=2)
        assert len(result) == 2
        # First should be most relevant (index 0), second is either 1 or 2
        assert result[0] == 0

    def test_diversity_with_low_lambda(self) -> None:
        query = [1.0, 0.0]
        candidates = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
        scores = [1.0, 0.99, 0.0]
        result = mmr_select(query, candidates, scores, top_k=2, lambda_mult=0.1)
        # With low lambda, diversity dominates — should pick the diverse doc
        assert 2 in result

    def test_relevance_with_high_lambda(self) -> None:
        query = [1.0, 0.0]
        candidates = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
        scores = [1.0, 0.99, 0.0]
        result = mmr_select(query, candidates, scores, top_k=2, lambda_mult=1.0)
        # With lambda=1.0, pure relevance — pick top 2 by score
        assert result == [0, 1]

    def test_top_k_geq_candidates(self) -> None:
        candidates = [[1.0, 0.0], [0.0, 1.0]]
        scores = [1.0, 0.5]
        result = mmr_select([1.0, 0.0], candidates, scores, top_k=5)
        assert len(result) == 2

    def test_single_candidate(self) -> None:
        result = mmr_select([1.0, 0.0], [[1.0, 0.0]], [1.0], top_k=5)
        assert result == [0]

    def test_empty_candidates(self) -> None:
        result = mmr_select([1.0, 0.0], [], [], top_k=5)
        assert result == []

    def test_top_k_zero(self) -> None:
        result = mmr_select([1.0, 0.0], [[1.0, 0.0]], [1.0], top_k=0)
        assert result == []

    def test_mmr_differs_from_pure_relevance(self) -> None:
        query = [1.0, 0.0]
        # Two near-identical docs + one diverse doc
        candidates = [[1.0, 0.0], [0.98, 0.02], [0.0, 1.0]]
        scores = [1.0, 0.98, 0.0]

        # Pure relevance (lambda=1): top 2 most relevant
        rel_result = mmr_select(query, candidates, scores, top_k=2, lambda_mult=1.0)
        # MMR with diversity (lambda=0.3): should include diverse doc
        mmr_result = mmr_select(query, candidates, scores, top_k=2, lambda_mult=0.3)

        assert rel_result == [0, 1]
        assert 2 in mmr_result

    def test_zero_score_candidates(self) -> None:
        query = [1.0, 0.0]
        candidates = [[0.0, 0.0], [1.0, 0.0]]
        scores = [0.0, 1.0]
        result = mmr_select(query, candidates, scores, top_k=1)
        assert result == [1]
