# Evaluation Framework Design

**Date**: 2026-07-26
**Status**: Draft
**Author**: Opencode

---

## Problem Statement

The RAG pipeline has no systematic way to measure quality:

1. **No baseline metrics**: We don't know current retrieval quality (precision, recall, MRR).
2. **No regression detection**: Changes to chunking, embedding, or reranking could silently degrade quality.
3. **No A/B comparison**: Can't compare "with HyDE" vs. "without HyDE" objectively.
4. **No ground truth**: No labeled dataset of query → expected document mappings.
5. **No monitoring**: No visibility into retrieval quality over time.

Without evaluation, improvements are guesswork.

---

## Solution Overview

Build an evaluation framework with:

1. **RAGAS Integration**: Use RAGAS metrics (faithfulness, answer relevancy, context precision/recall) for automated evaluation.
2. **Custom Metrics**: Domain-specific metrics for personal RAG use cases.
3. **Test Dataset**: Create and maintain a labeled query-document evaluation set.
4. **Evaluation Pipeline**: Automated evaluation runs that produce metric reports.
5. **A/B Testing Support**: Compare pipeline configurations side-by-side.

---

## Architecture

```
┌──────────────────────┐
│  Evaluation Dataset   │
│  (queries + expected  │
│   results)            │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Evaluation Pipeline  │
│                       │
│  1. Run queries       │
│  2. Collect results   │
│  3. Compute metrics   │
│  4. Generate report   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Metrics Store        │
│  (JSON reports +      │
│   time-series)        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Comparison View      │
│  (A vs B configs)     │
└──────────────────────┘
```

### Metrics

| Metric | Source | Description |
|--------|--------|-------------|
| **Hit@K** | Custom | Fraction of queries where at least 1 relevant doc in top-K |
| **MRR** | Custom | Mean Reciprocal Rank of first relevant result |
| **NDCG@K** | Custom | Normalized Discounted Cumulative Gain |
| **Context Precision** | RAGAS | Precision of retrieved context for answer generation |
| **Context Recall** | RAGAS | Recall of retrieved context vs. ground truth |
| **Faithfulness** | RAGAS | How faithful generated answer is to retrieved context |
| **Answer Relevancy** | RAGAS | How relevant generated answer is to the query |

---

## Files to Modify/Create

| File | Action | Purpose |
|------|--------|---------|
| `src/evaluation/` | **Create** | Evaluation framework module |
| `src/evaluation/__init__.py` | **Create** | Package init |
| `src/evaluation/metrics.py` | **Create** | Custom metric implementations |
| `src/evaluation/ragas_eval.py` | **Create** | RAGAS integration |
| `src/evaluation/dataset.py` | **Create** | Test dataset management |
| `src/evaluation/runner.py` | **Create** | Evaluation pipeline runner |
| `src/evaluation/report.py` | **Create** | Report generation |
| `src/config.py` | **Modify** | Add evaluation config |
| `pyproject.toml` | **Modify** | Add RAGAS + evaluation deps |
| `evaluation.xml` | **Modify** | Update evaluation dataset format |
| `tests/unit/test_evaluation.py` | **Create** | Unit tests for metrics |
| `scripts/evaluate.py` | **Create** | CLI script for running evaluations |

---

## Implementation Details

### 1. Configuration (`config.py` additions)

```python
# ── Evaluation ──────────────────────────────────────────────────────────────
EVAL_DATASET_PATH: str = _env("EVAL_DATASET_PATH", "evaluation.jsonl")
EVAL_OUTPUT_DIR: str = _env("EVAL_OUTPUT_DIR", "eval_reports")
EVAL_TOP_K: int = _env_int("EVAL_TOP_K", 10)
EVAL_RUN_RAGAS: bool = _env_bool("EVAL_RUN_RAGAS", False)
```

### 2. Evaluation Dataset Format (`evaluation.jsonl`)

Each line is a JSON object:

```json
{
  "query": "What was the Q3 revenue?",
  "expected_sources": ["/docs/financial_report_q3.pdf"],
  "expected_content_keywords": ["revenue", "Q3", "financial"],
  "metadata": {
    "difficulty": "easy",
    "category": "financial",
    "expected_doc_type": "report"
  }
}
```

### 3. Metrics Module (`src/evaluation/metrics.py`)

```python
"""Custom RAG evaluation metrics."""

from __future__ import annotations
import math
from typing import Any


def hit_at_k(retrieved_sources: list[str], expected_sources: list[str], k: int = 5) -> float:
    """1.0 if any expected source in top-K results, 0.0 otherwise."""
    top_k = retrieved_sources[:k]
    return 1.0 if any(s in top_k for s in expected_sources) else 0.0


def mean_reciprocal_rank(
    retrieved_sources: list[str], expected_sources: list[str]
) -> float:
    """Reciprocal rank of first relevant result, averaged over queries."""
    for i, source in enumerate(retrieved_sources):
        if source in expected_sources:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    retrieved_sources: list[str],
    expected_sources: list[str],
    k: int = 10,
) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    def dcg(scores: list[float]) -> float:
        return sum(s / math.log2(i + 2) for i, s in enumerate(scores))

    # Binary relevance
    relevance = [1.0 if s in expected_sources else 0.0 for s in retrieved_sources[:k]]
    actual_dcg = dcg(relevance)

    # Ideal DCG
    ideal = sorted(relevance, reverse=True)
    ideal_dcg = dcg(ideal)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def precision_at_k(retrieved_sources: list[str], expected_sources: list[str], k: int = 5) -> float:
    """Fraction of top-K results that are relevant."""
    top_k = retrieved_sources[:k]
    relevant = sum(1 for s in top_k if s in expected_sources)
    return relevant / k if k > 0 else 0.0


def recall_at_k(retrieved_sources: list[str], expected_sources: list[str], k: int = 10) -> float:
    """Fraction of expected sources found in top-K results."""
    top_k = retrieved_sources[:k]
    found = sum(1 for s in expected_sources if s in top_k)
    return found / len(expected_sources) if expected_sources else 0.0


def keyword_coverage(
    retrieved_content: str, expected_keywords: list[str]
) -> float:
    """Fraction of expected keywords found in retrieved content."""
    if not expected_keywords:
        return 1.0
    content_lower = retrieved_content.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in content_lower)
    return found / len(expected_keywords)


def compute_all_metrics(
    retrieved: list[dict[str, Any]],
    expected_sources: list[str],
    expected_keywords: list[str] | None = None,
    k: int = 5,
) -> dict[str, float]:
    """Compute all custom metrics for a single query."""
    sources = [r.get("source", "") for r in retrieved]
    content = " ".join(r.get("content", "") for r in retrieved)

    metrics = {
        f"hit@{k}": hit_at_k(sources, expected_sources, k),
        "mrr": mean_reciprocal_rank(sources, expected_sources),
        f"ndcg@{k}": ndcg_at_k(sources, expected_sources, k),
        f"precision@{k}": precision_at_k(sources, expected_sources, k),
        f"recall@{k}": recall_at_k(sources, expected_sources, k),
    }

    if expected_keywords:
        metrics["keyword_coverage"] = keyword_coverage(content, expected_keywords)

    return metrics
```

### 4. RAGAS Integration (`src/evaluation/ragas_eval.py`)

```python
"""RAGAS evaluation integration."""

from __future__ import annotations
import logging
from typing import Any
from . import config

logger = logging.getLogger("ragas-eval")


def evaluate_with_ragas(
    queries: list[str],
    contexts: list[list[str]],
    answers: list[str],
    ground_truths: list[list[str]],
) -> dict[str, float]:
    """Run RAGAS evaluation on a set of queries.

    Args:
        queries: List of user queries
        contexts: List of retrieved context lists per query
        answers: List of generated answers per query
        ground_truths: List of ground truth answer lists per query

    Returns:
        Dict of metric_name -> score
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from datasets import Dataset

        data = {
            "question": queries,
            "contexts": contexts,
            "answer": answers,
            "ground_truth": ground_truths,
        }
        dataset = Dataset.from_dict(data)

        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )

        return {
            "faithfulness": result["faithfulness"],
            "answer_relevancy": result["answer_relevancy"],
            "context_precision": result["context_precision"],
            "context_recall": result["context_recall"],
        }
    except ImportError:
        logger.warning("RAGAS not installed. Install with: pip install ragas")
        return {}
    except Exception as exc:
        logger.error("RAGAS evaluation failed: %s", exc)
        return {}
```

### 5. Evaluation Runner (`src/evaluation/runner.py`)

```python
"""Evaluation pipeline runner."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import metrics
from .metrics import compute_all_metrics
from .. import config

logger = logging.getLogger("eval-runner")


class EvalDataset:
    """Load and manage evaluation dataset."""

    def __init__(self, path: str = ""):
        self.path = path or config.EVAL_DATASET_PATH
        self.queries: list[dict[str, Any]] = []

    def load(self) -> None:
        """Load evaluation dataset from JSONL file."""
        path = Path(self.path)
        if not path.exists():
            logger.warning("Eval dataset not found: %s", self.path)
            return

        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.queries.append(json.loads(line))

        logger.info("Loaded %d evaluation queries", len(self.queries))

    def add_query(
        self,
        query: str,
        expected_sources: list[str],
        expected_keywords: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a query to the dataset."""
        self.queries.append({
            "query": query,
            "expected_sources": expected_sources,
            "expected_content_keywords": expected_keywords or [],
            "metadata": metadata or {},
        })

    def save(self) -> None:
        """Save dataset to JSONL."""
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for q in self.queries:
                f.write(json.dumps(q) + "\n")


class EvalRunner:
    """Run evaluations against the RAG pipeline."""

    def __init__(self, engine: Any):
        self.engine = engine
        self.dataset = EvalDataset()
        self.dataset.load()

    def run(self, top_k: int = 0) -> dict[str, Any]:
        """Run full evaluation and return results."""
        k = top_k or config.EVAL_TOP_K
        all_metrics: list[dict[str, float]] = []

        for i, query_data in enumerate(self.dataset.queries):
            query = query_data["query"]
            expected_sources = query_data["expected_sources"]
            expected_keywords = query_data.get("expected_content_keywords", [])

            logger.info("Evaluating query %d/%d: %s", i + 1, len(self.dataset.queries), query[:50])

            # Run search
            results = self.engine.hybrid_search(
                query=query,
                top_k=k,
                rerank=config.RERANK_ENABLED,
            )

            # Compute metrics
            query_metrics = compute_all_metrics(
                retrieved=results,
                expected_sources=expected_sources,
                expected_keywords=expected_keywords,
                k=k,
            )
            query_metrics["query"] = query
            all_metrics.append(query_metrics)

        # Aggregate
        aggregated = self._aggregate(all_metrics)
        aggregated["timestamp"] = datetime.now(UTC).isoformat()
        aggregated["num_queries"] = len(all_metrics)
        aggregated["config"] = {
            "top_k": k,
            "rerank": config.RERANK_ENABLED,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_strategy": config.CHUNK_STRATEGY,
        }

        # Save report
        self._save_report(aggregated)

        return aggregated

    def _aggregate(self, all_metrics: list[dict[str, float]]) -> dict[str, float]:
        """Aggregate per-query metrics into averages."""
        if not all_metrics:
            return {}

        metric_keys = [k for k in all_metrics[0].keys() if k != "query"]
        aggregated = {}
        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            aggregated[f"avg_{key}"] = sum(values) / len(values) if values else 0.0
        return aggregated

    def _save_report(self, report: dict[str, Any]) -> None:
        """Save evaluation report to file."""
        output_dir = Path(config.EVAL_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"eval_{timestamp}.json"

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("Evaluation report saved to %s", report_path)
```

### 6. CLI Script (`scripts/evaluate.py`)

```python
"""CLI script for running RAG evaluations."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RAGEngine
from src.evaluation.runner import EvalRunner
from src.evaluation.dataset import EvalDataset


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--dataset", default="evaluation.jsonl", help="Eval dataset path")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K results")
    parser.add_argument("--output", default="eval_reports", help="Output directory")
    parser.add_argument("--add-query", action="store_true", help="Interactive query addition")
    args = parser.parse_args()

    if args.add_query:
        dataset = EvalDataset(args.dataset)
        dataset.load()
        print("Add evaluation queries (empty line to finish):")
        while True:
            query = input("Query: ").strip()
            if not query:
                break
            sources = input("Expected sources (comma-separated): ").strip().split(",")
            keywords = input("Expected keywords (comma-separated): ").strip().split(",")
            dataset.add_query(query, [s.strip() for s in sources], [k.strip() for k in keywords])
            dataset.save()
        print(f"Dataset saved with {len(dataset.queries)} queries")
        return

    engine = RAGEngine()
    runner = EvalRunner(engine)
    runner.dataset.path = args.dataset
    runner.dataset.load()

    print(f"Running evaluation with {len(runner.dataset.queries)} queries...")
    results = runner.run(top_k=args.top_k)

    print("\n=== Evaluation Results ===")
    for key, value in sorted(results.items()):
        if key.startswith("avg_"):
            print(f"  {key}: {value:.4f}")

    print(f"\nReport saved to {args.output}/")


if __name__ == "__main__":
    main()
```

### 7. Dependencies (`pyproject.toml` additions)

```toml
[project.optional-dependencies]
eval = [
    "ragas>=0.2,<1",
    "datasets>=3,<4",
    "langdetect>=1,<2",
]
```

---

## Testing Strategy

### Unit Tests (`tests/unit/test_evaluation.py`)

- `test_hit_at_k_found`: Query matches expected source → 1.0.
- `test_hit_at_k_not_found`: Query doesn't match → 0.0.
- `test_mrr_first_rank`: First result relevant → MRR = 1.0.
- `test_mrr_second_rank`: Second result relevant → MRR = 0.5.
- `test_ndcg_perfect`: All relevant at top → NDCG = 1.0.
- `test_precision_at_k`: Correct ratio of relevant in top-K.
- `test_recall_at_k`: Correct ratio of expected found.
- `test_keyword_coverage`: Keyword presence detection works.
- `test_compute_all_metrics`: Full metric computation returns all keys.
- `test_eval_dataset_load`: JSONL loading works.
- `test_eval_dataset_add_and_save`: Add query and persist.

### Integration Tests

- `test_run_evaluation`: Full evaluation run with mock engine.
- `test_evaluation_report_saved`: Verify report file created.
- `test_evaluation_aggregation`: Verify metrics correctly averaged.

### Sample Dataset

Create `evaluation.jsonl` with 10-20 representative queries covering:
- Easy queries (exact keyword match)
- Medium queries (semantic match needed)
- Hard queries (ambiguous, requires context)
- Multi-document queries
- Filtered queries (by document type)

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| RAGAS is heavy dependency | Large install size | Make RAGAS optional; custom metrics work without it |
| Evaluation takes too long | Slow feedback loop | Run subset for quick checks; full eval only before releases |
| Test dataset becomes stale | Misleading metrics | Version control dataset; add new queries as documents change |
| Metrics don't reflect real user quality | Optimizing wrong thing | Include user studies; use metrics as proxy, not absolute |
| LLM-based metrics are non-deterministic | Inconsistent scores | Run multiple times and average; set temperature=0 |
| Evaluation dataset is too small | Unreliable statistics | Start with 20 queries, grow to 50+ over time |

---

## Priority & Effort

- **Priority**: High (essential for measuring improvement of other features)
- **Estimated effort**: 2-3 days
- **Dependencies**: Optional RAGAS for advanced metrics
- **Rollback**: Evaluation is read-only; no impact on production pipeline

---

## Implementation Order

1. **Custom metrics** (day 1): Implement hit@K, MRR, NDCG — no dependencies.
2. **Evaluation runner** (day 1): Runner + dataset management.
3. **CLI script** (day 2): Script for manual evaluation runs.
4. **RAGAS integration** (day 2-3): Add RAGAS metrics as optional enhancement.
5. **Sample dataset** (day 3): Create initial evaluation dataset from existing documents.
