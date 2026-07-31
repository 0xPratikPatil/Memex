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
from .sweep import SweepResult, sweep

__all__ = [
    "EvalResult",
    "GoldenQuery",
    "GoldenSet",
    "QueryMetrics",
    "SweepResult",
    "hit_rate_at_k",
    "keyword_coverage",
    "match_source",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "sweep",
]
