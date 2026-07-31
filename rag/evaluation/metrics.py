"""Evaluation metrics for RAG retrieval quality."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QueryMetrics:
    """Metrics for a single query evaluation.

    Attributes:
        recall_at_k: Fraction of expected documents found in top k.
        precision_at_k: Fraction of top k that are correct.
        hit_rate_at_k: 1.0 if any expected doc found in top k, 0.0 otherwise.
        mrr: 1/rank of the first correct result.
        keyword_coverage: Fraction of expected keywords found in retrieved content.
    """

    recall_at_k: float
    precision_at_k: float
    hit_rate_at_k: float
    mrr: float
    keyword_coverage: float


@dataclass
class EvalResult:
    """Aggregated evaluation results across all queries.

    Attributes:
        queries: Per-query metric results.
        avg_recall: Average recall across all queries.
        avg_precision: Average precision across all queries.
        avg_hit_rate: Average hit rate across all queries.
        avg_mrr: Average MRR across all queries.
        avg_keyword_coverage: Average keyword coverage across all queries.
        total_queries: Number of queries evaluated.
    """

    queries: list[dict] = field(default_factory=list)
    avg_recall: float = 0.0
    avg_precision: float = 0.0
    avg_hit_rate: float = 0.0
    avg_mrr: float = 0.0
    avg_keyword_coverage: float = 0.0
    total_queries: int = 0

    def summary_table(self) -> str:
        """Format aggregate metrics as a plain-text table.

        Returns:
            A multi-line string with one row per metric.
        """
        lines = [
            f"{'Metric':<24} {'Value':>8}",
            "-" * 34,
            f"{'recall':<24} {self.avg_recall:>8.3f}",
            f"{'precision':<24} {self.avg_precision:>8.3f}",
            f"{'hit_rate':<24} {self.avg_hit_rate:>8.3f}",
            f"{'mrr':<24} {self.avg_mrr:>8.3f}",
            f"{'keyword_coverage':<24} {self.avg_keyword_coverage:>8.3f}",
            "-" * 34,
            f"{'total_queries':<24} {self.total_queries:>8d}",
        ]
        return "\n".join(lines)


def recall_at_k(retrieved_sources: list[str], expected_sources: list[str], k: int) -> float:
    """Fraction of expected documents found in top k.

    This is the metric that matters most for RAG. A chunk that was never
    retrieved cannot be cited, so recall puts a ceiling on how good any
    downstream answer can be.

    Args:
        retrieved_sources: Retrieved source identifiers in rank order.
        expected_sources: Identifiers that count as correct.
        k: Cutoff.

    Returns:
        A value from 0.0 to 1.0. Returns 0.0 when nothing is expected.
    """
    if not expected_sources:
        return 0.0
    expected_set = set(expected_sources)
    window = set(retrieved_sources[:k])
    return len(expected_set & window) / len(expected_set)


def precision_at_k(retrieved_sources: list[str], expected_sources: list[str], k: int) -> float:
    """Fraction of the top k results that are correct.

    Low precision means the LLM is being handed irrelevant chunks alongside
    the useful ones, which costs tokens and invites distraction.

    Args:
        retrieved_sources: Retrieved source identifiers in rank order.
        expected_sources: Identifiers that count as correct.
        k: Cutoff.

    Returns:
        A value from 0.0 to 1.0. Returns 0.0 when nothing was retrieved.
    """
    window = retrieved_sources[:k]
    if not window:
        return 0.0
    expected_set = set(expected_sources)
    hits = sum(1 for item in window if item in expected_set)
    return hits / len(window)


def hit_rate_at_k(retrieved_sources: list[str], expected_sources: list[str], k: int) -> float:
    """1.0 if any expected doc found in top k, 0.0 otherwise.

    The bluntest useful measure: did retrieval find anything at all? A hit
    rate well below 1.0 means some queries are unanswerable no matter how
    good the generation step is.

    Args:
        retrieved_sources: Retrieved source identifiers in rank order.
        expected_sources: Identifiers that count as correct.
        k: Cutoff.

    Returns:
        1.0 if at least one expected source appears in top k, else 0.0.
    """
    window = set(retrieved_sources[:k])
    expected_set = set(expected_sources)
    return 1.0 if window & expected_set else 0.0


def reciprocal_rank(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """1/rank of the first correct result, or 0.0 if there is none.

    Unlike recall, this is sensitive to ordering, which is exactly what a
    reranker changes. If reranking helps but recall stays flat, MRR is where
    the improvement shows up.

    Args:
        retrieved_sources: Retrieved source identifiers in rank order.
        expected_sources: Identifiers that count as correct.

    Returns:
        1.0 / rank of the first match, or 0.0.
    """
    expected_set = set(expected_sources)
    for position, item in enumerate(retrieved_sources, start=1):
        if item in expected_set:
            return 1.0 / position
    return 0.0


def keyword_coverage(retrieved_content: str, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords found in retrieved content.

    Args:
        retrieved_content: Concatenated content from retrieved chunks.
        expected_keywords: Keywords that should appear.

    Returns:
        A value from 0.0 to 1.0. Returns 1.0 when no keywords are expected.
    """
    if not expected_keywords:
        return 1.0
    content_lower = retrieved_content.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in content_lower)
    return found / len(expected_keywords)
