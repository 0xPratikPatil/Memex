"""Eval sweep — compare multiple retrieval configs side by side."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .metrics import EvalResult

log = logging.getLogger(__name__)


@dataclass
class SweepResult:
    """The outcome of comparing multiple evaluation variants.

    Attributes:
        variant_name: A name for the variant.
        eval_result: The aggregated evaluation result.
    """

    variant_name: str
    eval_result: EvalResult


def sweep(eval_results: list[tuple[str, EvalResult]]) -> str:
    """Compare multiple eval results side by side.

    Produces a plain-text table with each variant as a row and deltas
    relative to the first variant (baseline).

    Args:
        eval_results: List of (variant_name, EvalResult) tuples.
            The first entry is treated as the baseline.

    Returns:
        A formatted comparison table string.
    """
    if not eval_results:
        return "No variants were evaluated."

    metric_keys = ["avg_recall", "avg_precision", "avg_hit_rate", "avg_mrr", "avg_keyword_coverage"]
    col_headers = ["recall", "precision", "hit_rate", "mrr", "kw_cov"]

    name_width = max(len(name) for name, _ in eval_results)
    name_width = max(name_width, len("variant"))
    cell = 11

    header = f"{'variant':<{name_width}}  " + "  ".join(f"{h:>{cell}}" for h in col_headers)
    lines = [header, "-" * len(header)]

    baseline_metrics = {k: getattr(eval_results[0][1], k, 0.0) for k in metric_keys}

    for name, result in eval_results:
        cells = []
        for mk in metric_keys:
            value = getattr(result, mk, 0.0)
            if (name, result) == eval_results[0]:
                cells.append(f"{value:.3f}".rjust(cell))
            else:
                delta = value - baseline_metrics.get(mk, 0.0)
                cells.append(f"{value:.3f}{delta:+.2f}".rjust(cell))
        lines.append(f"{name:<{name_width}}  " + "  ".join(cells))

    best_name, best_result = max(eval_results, key=lambda x: x[1].avg_recall)
    if len(eval_results) > 1:
        lines.append("")
        lines.append(f"Best recall: {best_name} ({best_result.avg_recall:.3f})")

    return "\n".join(lines)
