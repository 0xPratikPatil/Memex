"""Evaluation framework — golden-set testing and metrics."""

from .golden import GoldenQuery, GoldenSet, match_source
from .metrics import (
    EvalResult,
    QueryMetrics,
    hit_rate_at_k,
    keyword_coverage,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from .runner import (
    BenchmarkResult,
    EvalDataset,
    EvalQuery,
    EvalRunner,
    PerformanceBenchmark,
    compute_all_metrics,
    hit_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
)
from .runner import (
    keyword_coverage as old_keyword_coverage,
)
from .runner import (
    precision_at_k as old_precision_at_k,
)
from .runner import (
    recall_at_k as old_recall_at_k,
)
from .sweep import SweepResult, sweep

__all__ = [
    "BenchmarkResult",
    "EvalDataset",
    "EvalQuery",
    "EvalResult",
    "EvalRunner",
    "GoldenQuery",
    "GoldenSet",
    "PerformanceBenchmark",
    "QueryMetrics",
    "SweepResult",
    "compute_all_metrics",
    "hit_at_k",
    "hit_rate_at_k",
    "keyword_coverage",
    "match_source",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "old_keyword_coverage",
    "old_precision_at_k",
    "old_recall_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "sweep",
]
