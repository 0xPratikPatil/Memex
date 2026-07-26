"""Unit tests for the evaluation framework."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.services.evaluation import (
    BenchmarkResult,
    EvalDataset,
    EvalRunner,
    PerformanceBenchmark,
    compute_all_metrics,
    hit_at_k,
    keyword_coverage,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# ── hit_at_k tests ──────────────────────────────────────────────────────────


class TestHitAtK:
    def test_hit_at_k_found(self) -> None:
        assert hit_at_k(["a.pdf", "b.pdf"], ["b.pdf"], k=5) == 1.0

    def test_hit_at_k_not_found(self) -> None:
        assert hit_at_k(["a.pdf", "b.pdf"], ["c.pdf"], k=5) == 0.0

    def test_hit_at_k_respects_k(self) -> None:
        assert hit_at_k(["a.pdf", "b.pdf"], ["c.pdf"], k=1) == 0.0

    def test_hit_at_k_exact_k_boundary(self) -> None:
        assert hit_at_k(["a.pdf", "b.pdf", "c.pdf"], ["c.pdf"], k=3) == 1.0

    def test_hit_at_k_outside_k(self) -> None:
        assert hit_at_k(["a.pdf", "b.pdf", "c.pdf"], ["c.pdf"], k=2) == 0.0

    def test_hit_at_k_empty_sources(self) -> None:
        assert hit_at_k([], ["a.pdf"], k=5) == 0.0

    def test_hit_at_k_empty_expected(self) -> None:
        assert hit_at_k(["a.pdf"], [], k=5) == 0.0


# ── MRR tests ───────────────────────────────────────────────────────────────


class TestMeanReciprocalRank:
    def test_mrr_first_rank(self) -> None:
        assert mean_reciprocal_rank(["a.pdf", "b.pdf"], ["a.pdf"]) == 1.0

    def test_mrr_second_rank(self) -> None:
        assert mean_reciprocal_rank(["a.pdf", "b.pdf"], ["b.pdf"]) == 0.5

    def test_mrr_third_rank(self) -> None:
        assert mean_reciprocal_rank(["a.pdf", "b.pdf", "c.pdf"], ["c.pdf"]) == pytest.approx(
            1.0 / 3.0
        )

    def test_mrr_no_match(self) -> None:
        assert mean_reciprocal_rank(["a.pdf", "b.pdf"], ["z.pdf"]) == 0.0

    def test_mrr_multiple_expected_first_match(self) -> None:
        assert mean_reciprocal_rank(["a.pdf", "b.pdf"], ["a.pdf", "b.pdf"]) == 1.0

    def test_mrr_empty_retrieved(self) -> None:
        assert mean_reciprocal_rank([], ["a.pdf"]) == 0.0


# ── NDCG tests ──────────────────────────────────────────────────────────────


class TestNDCG:
    def test_ndcg_perfect(self) -> None:
        assert ndcg_at_k(["a.pdf", "b.pdf"], ["a.pdf", "b.pdf"], k=2) == pytest.approx(1.0)

    def test_ndcg_no_relevant(self) -> None:
        assert ndcg_at_k(["a.pdf", "b.pdf"], ["z.pdf"], k=2) == 0.0

    def test_ndcg_partial(self) -> None:
        score = ndcg_at_k(["a.pdf", "b.pdf", "c.pdf"], ["b.pdf"], k=3)
        assert 0.0 < score < 1.0

    def test_ndcg_empty_retrieved(self) -> None:
        assert ndcg_at_k([], ["a.pdf"], k=5) == 0.0

    def test_ndcg_empty_expected(self) -> None:
        assert ndcg_at_k(["a.pdf"], [], k=5) == 0.0


# ── precision_at_k tests ────────────────────────────────────────────────────


class TestPrecisionAtK:
    def test_precision_all_relevant(self) -> None:
        assert precision_at_k(["a.pdf", "b.pdf"], ["a.pdf", "b.pdf"], k=2) == 1.0

    def test_precision_none_relevant(self) -> None:
        assert precision_at_k(["a.pdf", "b.pdf"], ["z.pdf"], k=2) == 0.0

    def test_precision_half(self) -> None:
        assert precision_at_k(["a.pdf", "b.pdf"], ["a.pdf"], k=2) == 0.5

    def test_precision_k_zero(self) -> None:
        assert precision_at_k(["a.pdf"], ["a.pdf"], k=0) == 0.0


# ── recall_at_k tests ───────────────────────────────────────────────────────


class TestRecallAtK:
    def test_recall_all_found(self) -> None:
        assert recall_at_k(["a.pdf", "b.pdf"], ["a.pdf", "b.pdf"], k=2) == 1.0

    def test_recall_partial(self) -> None:
        assert recall_at_k(["a.pdf", "c.pdf"], ["a.pdf", "b.pdf"], k=2) == 0.5

    def test_recall_none_found(self) -> None:
        assert recall_at_k(["a.pdf"], ["z.pdf"], k=5) == 0.0

    def test_recall_empty_expected(self) -> None:
        assert recall_at_k(["a.pdf"], [], k=5) == 0.0

    def test_recall_empty_retrieved(self) -> None:
        assert recall_at_k([], ["a.pdf"], k=5) == 0.0


# ── keyword_coverage tests ──────────────────────────────────────────────────


class TestKeywordCoverage:
    def test_all_found(self) -> None:
        assert keyword_coverage("revenue Q3 financial", ["revenue", "Q3"]) == 1.0

    def test_partial(self) -> None:
        assert keyword_coverage("revenue report", ["revenue", "Q3"]) == 0.5

    def test_none_found(self) -> None:
        assert keyword_coverage("hello world", ["revenue", "Q3"]) == 0.0

    def test_empty_keywords(self) -> None:
        assert keyword_coverage("anything", []) == 1.0

    def test_case_insensitive(self) -> None:
        assert keyword_coverage("Revenue Q3", ["revenue", "q3"]) == 1.0


# ── compute_all_metrics tests ───────────────────────────────────────────────


class TestComputeAllMetrics:
    def test_returns_all_keys(self) -> None:
        retrieved = [{"source": "a.pdf", "content": "revenue data"}, {"source": "b.pdf", "content": "cost data"}]
        result = compute_all_metrics(retrieved, ["a.pdf"], ["revenue"], k=2)
        assert "hit@2" in result
        assert "mrr" in result
        assert "ndcg@2" in result
        assert "precision@2" in result
        assert "recall@2" in result
        assert "keyword_coverage" in result

    def test_without_keywords(self) -> None:
        retrieved = [{"source": "a.pdf", "content": "data"}]
        result = compute_all_metrics(retrieved, ["a.pdf"], k=1)
        assert "keyword_coverage" not in result

    def test_perfect_scores(self) -> None:
        retrieved = [
            {"source": "a.pdf", "content": "revenue Q3"},
            {"source": "b.pdf", "content": "cost Q3"},
        ]
        result = compute_all_metrics(retrieved, ["a.pdf", "b.pdf"], ["revenue", "Q3"], k=2)
        assert result["hit@2"] == 1.0
        assert result["mrr"] == 1.0
        assert result["precision@2"] == 1.0
        assert result["recall@2"] == 1.0


# ── EvalDataset tests ───────────────────────────────────────────────────────


class TestEvalDataset:
    def test_load_jsonl(self, tmp_path: Path) -> None:
        dataset_file = tmp_path / "test.jsonl"
        lines = [
            json.dumps({"query": "q1", "expected_sources": ["a.pdf"], "expected_content_keywords": ["kw1"]}),
            json.dumps({"query": "q2", "expected_sources": ["b.pdf"], "expected_content_keywords": []}),
        ]
        dataset_file.write_text("\n".join(lines))

        dataset = EvalDataset(str(dataset_file))
        dataset.load()
        assert len(dataset.queries) == 2
        assert dataset.queries[0].query == "q1"
        assert dataset.queries[0].expected_sources == ["a.pdf"]

    def test_load_xml(self, tmp_path: Path) -> None:
        xml_file = tmp_path / "test.xml"
        xml_file.write_text(
            '<evaluation><qa_pair><question>What is X?</question><answer>42</answer></qa_pair></evaluation>'
        )
        dataset = EvalDataset(str(xml_file))
        dataset.load()
        assert len(dataset.queries) == 1
        assert dataset.queries[0].query == "What is X?"
        assert dataset.queries[0].expected_keywords == ["42"]

    def test_load_nonexistent(self) -> None:
        dataset = EvalDataset("/nonexistent/path.jsonl")
        dataset.load()
        assert len(dataset.queries) == 0

    def test_add_and_save(self, tmp_path: Path) -> None:
        dataset_file = tmp_path / "test.jsonl"
        dataset = EvalDataset(str(dataset_file))
        dataset.add_query("q1", ["a.pdf"], ["kw1"], {"category": "test"})
        dataset.save()

        loaded = EvalDataset(str(dataset_file))
        loaded.load()
        assert len(loaded.queries) == 1
        assert loaded.queries[0].query == "q1"
        assert loaded.queries[0].expected_sources == ["a.pdf"]

    def test_filter_by_category(self, tmp_path: Path) -> None:
        dataset = EvalDataset(str(tmp_path / "test.jsonl"))
        dataset.add_query("q1", [], [], {"category": "financial"})
        dataset.add_query("q2", [], [], {"category": "technical"})
        dataset.add_query("q3", [], [], {"category": "financial"})

        financial = dataset.filter_by_category("financial")
        assert len(financial) == 2

    def test_filter_by_difficulty(self, tmp_path: Path) -> None:
        dataset = EvalDataset(str(tmp_path / "test.jsonl"))
        dataset.add_query("q1", [], [], {"difficulty": "easy"})
        dataset.add_query("q2", [], [], {"difficulty": "hard"})

        easy = dataset.filter_by_difficulty("easy")
        assert len(easy) == 1


# ── PerformanceBenchmark tests ──────────────────────────────────────────────


class TestPerformanceBenchmark:
    def test_record_and_results(self) -> None:
        bench = PerformanceBenchmark()
        bench.record("search", 10.0)
        bench.record("search", 20.0)
        bench.record("search", 30.0)

        results = bench.get_results()
        assert "search" in results
        assert results["search"].count == 3
        assert results["search"].avg_ms == 20.0

    def test_percentiles(self) -> None:
        bench = PerformanceBenchmark()
        for i in range(1, 101):
            bench.record("op", float(i))

        results = bench.get_results()
        assert results["op"].p50_ms == pytest.approx(50.0, abs=1.0)
        assert results["op"].min_ms == 1.0
        assert results["op"].max_ms == 100.0

    def test_reset(self) -> None:
        bench = PerformanceBenchmark()
        bench.record("op", 10.0)
        bench.reset()
        assert bench.get_results() == {}

    def test_empty(self) -> None:
        bench = PerformanceBenchmark()
        assert bench.get_results() == {}

    def test_as_dict(self) -> None:
        bench = PerformanceBenchmark()
        bench.record("op", 10.0)
        d = bench.as_dict()
        assert "op" in d
        assert d["op"]["count"] == 1


# ── BenchmarkResult tests ───────────────────────────────────────────────────


class TestBenchmarkResult:
    def test_empty(self) -> None:
        result = BenchmarkResult(operation="test")
        assert result.avg_ms == 0.0
        assert result.count == 0

    def test_as_dict(self) -> None:
        result = BenchmarkResult(operation="test", latencies=[10.0, 20.0])
        d = result.as_dict()
        assert d["operation"] == "test"
        assert d["count"] == 2
        assert d["avg_ms"] == 15.0


# ── EvalRunner tests ────────────────────────────────────────────────────────


class TestEvalRunner:
    @patch("rag.services.evaluation.config")
    def test_run_full_evaluation(self, mock_config: MagicMock, tmp_path: Path) -> None:
        mock_config.EVAL_DATASET_PATH = str(tmp_path / "eval.jsonl")
        mock_config.EVAL_TOP_K = 5
        mock_config.EVAL_OUTPUT_DIR = str(tmp_path / "reports")
        mock_config.EVAL_RUN_RAGAS = False
        mock_config.RERANK_ENABLED = False
        mock_config.CHUNK_SIZE = 512
        mock_config.CHUNK_STRATEGY = "recursive"

        # Create dataset
        dataset_file = tmp_path / "eval.jsonl"
        dataset_file.write_text(
            json.dumps({"query": "test query", "expected_sources": ["a.pdf"], "expected_content_keywords": ["test"]})
        )

        mock_engine = MagicMock()
        mock_engine.hybrid_search.return_value = [
            {"source": "a.pdf", "content": "test content", "rrf_score": 0.9}
        ]

        runner = EvalRunner.__new__(EvalRunner)
        runner.engine = mock_engine
        runner.dataset = EvalDataset(str(dataset_file))
        runner.dataset.load()
        runner.benchmark = PerformanceBenchmark()

        results = runner.run(top_k=5)

        assert results["num_queries"] == 1
        assert results["timestamp"]
        assert "config" in results
        assert "benchmarks" in results

    @patch("rag.services.evaluation.config")
    def test_run_single_query(self, mock_config: MagicMock) -> None:
        mock_config.EVAL_TOP_K = 5
        mock_config.EVAL_RUN_RAGAS = False
        mock_config.RERANK_ENABLED = False
        mock_config.EVAL_DATASET_PATH = "/nonexistent"
        mock_config.EVAL_OUTPUT_DIR = "/tmp"

        mock_engine = MagicMock()
        mock_engine.hybrid_search.return_value = [
            {"source": "a.pdf", "content": "data", "rrf_score": 0.9}
        ]

        runner = EvalRunner.__new__(EvalRunner)
        runner.engine = mock_engine
        runner.dataset = EvalDataset("/nonexistent")
        runner.benchmark = PerformanceBenchmark()

        result = runner.run_single("test query", ["a.pdf"], ["data"])

        assert result["query"] == "test query"
        assert result["results_count"] == 1
        assert result["mrr"] == 1.0

    @patch("rag.services.evaluation.config")
    def test_compare_results(self, mock_config: MagicMock) -> None:
        mock_config.EVAL_TOP_K = 5
        mock_config.EVAL_RUN_RAGAS = False
        mock_config.EVAL_DATASET_PATH = "/nonexistent"
        mock_config.EVAL_OUTPUT_DIR = "/tmp"

        runner = EvalRunner.__new__(EvalRunner)
        runner.engine = MagicMock()
        runner.dataset = EvalDataset("/nonexistent")
        runner.benchmark = PerformanceBenchmark()

        results_a = {"avg_hit@5": 0.8, "avg_mrr": 0.6}
        results_b = {"avg_hit@5": 0.9, "avg_mrr": 0.7}

        comparison = runner.compare(results_a, results_b, "baseline", "treatment")
        assert comparison["label_a"] == "baseline"
        assert comparison["label_b"] == "treatment"
        assert "avg_hit@5" in comparison["metrics"]
        assert comparison["metrics"]["avg_hit@5"]["delta"] == pytest.approx(0.1)


# ── Aggregation tests ───────────────────────────────────────────────────────


class TestAggregation:
    def test_aggregate_averages(self) -> None:
        mock_config_patcher = patch("rag.services.evaluation.config")
        mock_config = mock_config_patcher.start()
        mock_config.EVAL_TOP_K = 5
        mock_config.EVAL_RUN_RAGAS = False
        mock_config.EVAL_DATASET_PATH = "/nonexistent"
        mock_config.EVAL_OUTPUT_DIR = "/tmp"

        try:
            runner = EvalRunner.__new__(EvalRunner)
            runner.engine = MagicMock()
            runner.dataset = EvalDataset("/nonexistent")
            runner.benchmark = PerformanceBenchmark()

            all_metrics = [
                {"query": "q1", "hit@5": 1.0, "mrr": 1.0, "latency_ms": 100.0},
                {"query": "q2", "hit@5": 0.0, "mrr": 0.5, "latency_ms": 200.0},
            ]

            result = runner._aggregate(all_metrics)
            assert result["avg_hit@5"] == 0.5
            assert result["avg_mrr"] == 0.75
            assert result["avg_latency_ms"] == 150.0
        finally:
            mock_config_patcher.stop()

    def test_aggregate_empty(self) -> None:
        mock_config_patcher = patch("rag.services.evaluation.config")
        mock_config = mock_config_patcher.start()
        mock_config.EVAL_DATASET_PATH = "/nonexistent"
        mock_config.EVAL_OUTPUT_DIR = "/tmp"

        try:
            runner = EvalRunner.__new__(EvalRunner)
            runner.engine = MagicMock()
            runner.dataset = EvalDataset("/nonexistent")
            runner.benchmark = PerformanceBenchmark()

            result = runner._aggregate([])
            assert result == {}
        finally:
            mock_config_patcher.stop()
