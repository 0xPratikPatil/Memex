"""Evaluation framework for the RAG pipeline.

Provides custom metrics (Hit@K, MRR, NDCG, precision, recall, keyword coverage),
optional RAGAS integration for faithfulness/relevancy/context metrics, evaluation
dataset management, A/B comparison support, and performance benchmarking.

All evaluation is opt-in and has zero overhead when disabled.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag import config

logger = logging.getLogger("evaluation")


# ── Custom metrics ───────────────────────────────────────────────────────────


def hit_at_k(retrieved_sources: list[str], expected_sources: list[str], k: int = 5) -> float:
    """1.0 if any expected source appears in top-K results, 0.0 otherwise."""
    top_k = retrieved_sources[:k]
    return 1.0 if any(s in top_k for s in expected_sources) else 0.0


def mean_reciprocal_rank(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """Reciprocal rank of the first relevant result."""
    for i, source in enumerate(retrieved_sources):
        if source in expected_sources:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    retrieved_sources: list[str],
    expected_sources: list[str],
    k: int = 10,
) -> float:
    """Normalized Discounted Cumulative Gain at K (binary relevance)."""

    def dcg(scores: list[float]) -> float:
        return sum(s / math.log2(i + 2) for i, s in enumerate(scores))

    relevance = [1.0 if s in expected_sources else 0.0 for s in retrieved_sources[:k]]
    actual_dcg = dcg(relevance)
    ideal = sorted(relevance, reverse=True)
    ideal_dcg = dcg(ideal)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


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


def keyword_coverage(retrieved_content: str, expected_keywords: list[str]) -> float:
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
    """Compute all custom retrieval metrics for a single query."""
    sources = [r.get("source", "") for r in retrieved]
    content = " ".join(r.get("content", "") for r in retrieved)

    metrics: dict[str, float] = {
        f"hit@{k}": hit_at_k(sources, expected_sources, k),
        "mrr": mean_reciprocal_rank(sources, expected_sources),
        f"ndcg@{k}": ndcg_at_k(sources, expected_sources, k),
        f"precision@{k}": precision_at_k(sources, expected_sources, k),
        f"recall@{k}": recall_at_k(sources, expected_sources, k),
    }

    if expected_keywords:
        metrics["keyword_coverage"] = keyword_coverage(content, expected_keywords)

    return metrics


# ── RAGAS integration (optional) ────────────────────────────────────────────


def evaluate_with_ragas(
    queries: list[str],
    contexts: list[list[str]],
    answers: list[str],
    ground_truths: list[list[str]],
) -> dict[str, float]:
    """Run RAGAS evaluation on a set of queries.

    Returns an empty dict if RAGAS is not installed or evaluation fails.
    """
    if not config.EVAL_RUN_RAGAS:
        return {}

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

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
            "faithfulness": float(result["faithfulness"]),
            "answer_relevancy": float(result["answer_relevancy"]),
            "context_precision": float(result["context_precision"]),
            "context_recall": float(result["context_recall"]),
        }
    except ImportError:
        logger.warning("RAGAS not installed. Install with: pip install ragas")
        return {}
    except Exception as exc:
        logger.error("RAGAS evaluation failed: %s", exc)
        return {}


# ── Evaluation dataset ──────────────────────────────────────────────────────


@dataclass
class EvalQuery:
    """A single evaluation query with expected results."""

    query: str
    expected_sources: list[str]
    expected_keywords: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class EvalDataset:
    """Load and manage evaluation datasets (JSONL or XML)."""

    def __init__(self, path: str = "") -> None:
        self.path = path or config.EVAL_DATASET_PATH
        self.queries: list[EvalQuery] = []

    def load(self) -> None:
        """Load evaluation dataset from file (JSONL or XML)."""
        path = Path(self.path)
        if not path.exists():
            logger.warning("Eval dataset not found: %s", self.path)
            return

        if path.suffix == ".xml":
            self._load_xml(path)
        else:
            self._load_jsonl(path)

        logger.info("Loaded %d evaluation queries from %s", len(self.queries), self.path)

    def _load_jsonl(self, path: Path) -> None:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    self.queries.append(
                        EvalQuery(
                            query=data["query"],
                            expected_sources=data.get("expected_sources", []),
                            expected_keywords=data.get("expected_content_keywords", []),
                            metadata=data.get("metadata", {}),
                        )
                    )

    def _load_xml(self, path: Path) -> None:
        import xml.etree.ElementTree as ET

        tree = ET.parse(path)
        root = tree.getroot()
        for qa in root.findall("qa_pair"):
            question_el = qa.find("question")
            answer_el = qa.find("answer")
            if question_el is None or answer_el is None:
                continue
            question = (question_el.text or "").strip()
            answer = (answer_el.text or "").strip()
            if not question:
                continue
            self.queries.append(
                EvalQuery(
                    query=question,
                    expected_sources=[],
                    expected_keywords=[answer] if answer else [],
                    metadata={"expected_answer": answer, "source": "xml"},
                )
            )

    def add_query(
        self,
        query: str,
        expected_sources: list[str],
        expected_keywords: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a query to the dataset."""
        self.queries.append(
            EvalQuery(
                query=query,
                expected_sources=expected_sources,
                expected_keywords=expected_keywords or [],
                metadata=metadata or {},
            )
        )

    def save(self) -> None:
        """Save dataset to JSONL format."""
        path = Path(self.path)
        if path.suffix == ".xml":
            logger.warning("Cannot save to XML format; saving as JSONL instead")
            path = path.with_suffix(".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for q in self.queries:
                obj = {
                    "query": q.query,
                    "expected_sources": q.expected_sources,
                    "expected_content_keywords": q.expected_keywords,
                    "metadata": q.metadata,
                }
                f.write(json.dumps(obj) + "\n")
        logger.info("Saved %d queries to %s", len(self.queries), path)

    def filter_by_category(self, category: str) -> list[EvalQuery]:
        """Return queries matching a metadata category."""
        return [q for q in self.queries if q.metadata.get("category") == category]

    def filter_by_difficulty(self, difficulty: str) -> list[EvalQuery]:
        """Return queries matching a metadata difficulty level."""
        return [q for q in self.queries if q.metadata.get("difficulty") == difficulty]


# ── Performance benchmarks ──────────────────────────────────────────────────


@dataclass
class TimingResult:
    """Result of a timed operation."""

    operation: str
    elapsed_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results for a set of operations."""

    latencies: list[float] = field(default_factory=list)
    operation: str = ""

    @property
    def avg_ms(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    @property
    def p50_ms(self) -> float:
        return self._percentile(50)

    @property
    def p95_ms(self) -> float:
        return self._percentile(95)

    @property
    def p99_ms(self) -> float:
        return self._percentile(99)

    @property
    def min_ms(self) -> float:
        return min(self.latencies) if self.latencies else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.latencies) if self.latencies else 0.0

    @property
    def count(self) -> int:
        return len(self.latencies)

    def _percentile(self, pct: int) -> float:
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * pct / 100)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "count": self.count,
            "avg_ms": round(self.avg_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
        }


class PerformanceBenchmark:
    """Collect and aggregate performance timing data."""

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = {}

    def record(self, operation: str, elapsed_ms: float) -> None:
        """Record a timing measurement."""
        if operation not in self._timings:
            self._timings[operation] = []
        self._timings[operation].append(elapsed_ms)

    def get_results(self) -> dict[str, BenchmarkResult]:
        """Return aggregated benchmark results per operation."""
        results: dict[str, BenchmarkResult] = {}
        for op, latencies in self._timings.items():
            results[op] = BenchmarkResult(operation=op, latencies=latencies)
        return results

    def as_dict(self) -> dict[str, Any]:
        """Serialize all benchmark results."""
        return {op: bench.as_dict() for op, bench in self.get_results().items()}

    def reset(self) -> None:
        """Clear all recorded timings."""
        self._timings.clear()


# ── Evaluation runner ───────────────────────────────────────────────────────


class EvalRunner:
    """Run evaluations against the RAG pipeline and produce reports."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.dataset = EvalDataset()
        self.benchmark = PerformanceBenchmark()
        self.dataset.load()

    def run(self, top_k: int = 0) -> dict[str, Any]:
        """Run full evaluation and return aggregated results."""
        k = top_k or config.EVAL_TOP_K
        all_metrics: list[dict[str, float]] = []

        for i, query_data in enumerate(self.dataset.queries):
            logger.info(
                "Evaluating query %d/%d: %s",
                i + 1,
                len(self.dataset.queries),
                query_data.query[:80],
            )

            t0 = time.monotonic()
            results = self.engine.hybrid_search(
                query=query_data.query,
                top_k=k,
                rerank=config.RERANK_ENABLED,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000
            self.benchmark.record("hybrid_search", elapsed_ms)

            query_metrics = compute_all_metrics(
                retrieved=results,
                expected_sources=query_data.expected_sources,
                expected_keywords=query_data.expected_keywords,
                k=k,
            )
            query_metrics["query"] = query_data.query
            query_metrics["latency_ms"] = elapsed_ms
            all_metrics.append(query_metrics)

        aggregated = self._aggregate(all_metrics)
        aggregated["timestamp"] = datetime.now(UTC).isoformat()
        aggregated["num_queries"] = len(all_metrics)
        aggregated["config"] = {
            "top_k": k,
            "rerank": config.RERANK_ENABLED,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_strategy": config.CHUNK_STRATEGY,
        }
        aggregated["benchmarks"] = self.benchmark.as_dict()

        self._save_report(aggregated)
        return aggregated

    def run_single(
        self,
        query: str,
        expected_sources: list[str] | None = None,
        expected_keywords: list[str] | None = None,
        top_k: int = 0,
    ) -> dict[str, Any]:
        """Run evaluation on a single query and return results."""
        k = top_k or config.EVAL_TOP_K

        t0 = time.monotonic()
        results = self.engine.hybrid_search(
            query=query,
            top_k=k,
            rerank=config.RERANK_ENABLED,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000

        query_metrics: dict[str, Any] = {"query": query, "latency_ms": elapsed_ms}
        if expected_sources is not None:
            query_metrics.update(
                compute_all_metrics(
                    retrieved=results,
                    expected_sources=expected_sources,
                    expected_keywords=expected_keywords,
                    k=k,
                )
            )
        query_metrics["results_count"] = len(results)
        query_metrics["results"] = [
            {"source": r.get("source", ""), "rrf_score": r.get("rrf_score", 0.0)} for r in results
        ]
        return query_metrics

    def compare(
        self,
        results_a: dict[str, Any],
        results_b: dict[str, Any],
        label_a: str = "config_a",
        label_b: str = "config_b",
    ) -> dict[str, Any]:
        """Compare two evaluation result sets for A/B testing."""
        comparison: dict[str, Any] = {
            "label_a": label_a,
            "label_b": label_b,
            "timestamp": datetime.now(UTC).isoformat(),
            "metrics": {},
        }

        metric_keys = [k for k in results_a if k.startswith("avg_")]
        for key in metric_keys:
            val_a = results_a.get(key, 0.0)
            val_b = results_b.get(key, 0.0)
            comparison["metrics"][key] = {
                label_a: val_a,
                label_b: val_b,
                "delta": val_b - val_a,
                "delta_pct": ((val_b - val_a) / val_a * 100) if val_a else 0.0,
            }

        return comparison

    def _aggregate(self, all_metrics: list[dict[str, float]]) -> dict[str, float]:
        """Aggregate per-query metrics into averages."""
        if not all_metrics:
            return {}

        numeric_keys = [k for k in all_metrics[0] if k != "query" and isinstance(all_metrics[0].get(k), (int, float))]
        aggregated: dict[str, float] = {}
        for key in numeric_keys:
            values = [m[key] for m in all_metrics if key in m and isinstance(m[key], (int, float))]
            aggregated[f"avg_{key}"] = sum(values) / len(values) if values else 0.0
        return aggregated

    def _save_report(self, report: dict[str, Any]) -> None:
        """Save evaluation report to JSON file."""
        output_dir = Path(config.EVAL_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"eval_{timestamp}.json"

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("Evaluation report saved to %s", report_path)
