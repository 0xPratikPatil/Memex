"""Unit tests for the evaluation framework."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.evaluation.golden import GoldenQuery, GoldenSet, match_source
from rag.evaluation.metrics import (
    EvalResult,
    QueryMetrics,
    reciprocal_rank,
)
from rag.evaluation.metrics import (
    hit_rate_at_k as new_hit_rate_at_k,
)
from rag.evaluation.metrics import (
    keyword_coverage as new_keyword_coverage,
)
from rag.evaluation.metrics import (
    precision_at_k as new_precision_at_k,
)
from rag.evaluation.metrics import (
    recall_at_k as new_recall_at_k,
)
from rag.evaluation.sweep import sweep
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
        assert mean_reciprocal_rank(["a.pdf", "b.pdf", "c.pdf"], ["c.pdf"]) == pytest.approx(1.0 / 3.0)

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
            "<evaluation><qa_pair><question>What is X?</question><answer>42</answer></qa_pair></evaluation>"
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
        mock_config.ENABLE_RERANKING = False
        mock_config.CHUNK_SIZE = 512
        mock_config.CHUNK_STRATEGY = "recursive"

        # Create dataset
        dataset_file = tmp_path / "eval.jsonl"
        dataset_file.write_text(
            json.dumps({"query": "test query", "expected_sources": ["a.pdf"], "expected_content_keywords": ["test"]})
        )

        mock_engine = MagicMock()
        mock_engine.hybrid_search.return_value = [{"source": "a.pdf", "content": "test content", "rrf_score": 0.9}]

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
        mock_config.ENABLE_RERANKING = False
        mock_config.EVAL_DATASET_PATH = "/nonexistent"
        mock_config.EVAL_OUTPUT_DIR = "/tmp"

        mock_engine = MagicMock()
        mock_engine.hybrid_search.return_value = [{"source": "a.pdf", "content": "data", "rrf_score": 0.9}]

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


# ═══════════════════════════════════════════════════════════════════════════════
# New evaluation framework: rag.evaluation
# ═══════════════════════════════════════════════════════════════════════════════


# ── Golden set loading ───────────────────────────────────────────────────────


class TestGoldenSetYAML:
    def test_load_from_yaml_list(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.yaml"
        path.write_text(
            "- query: what is revenue\n"
            "  expected_sources: [a.pdf]\n"
            "  expected_keywords: [revenue]\n"
            "- query: how did costs change\n"
            "  expected_sources: [b.pdf, c.pdf]\n"
            "  category: financial\n"
            "  difficulty: hard\n",
            encoding="utf-8",
        )

        gs = GoldenSet.from_yaml(str(path))
        assert len(gs.queries) == 2
        assert gs.queries[0].query == "what is revenue"
        assert gs.queries[0].expected_sources == ["a.pdf"]
        assert gs.queries[0].expected_keywords == ["revenue"]
        assert gs.queries[1].category == "financial"
        assert gs.queries[1].difficulty == "hard"

    def test_load_from_yaml_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.yaml"
        path.write_text(
            "queries:\n"
            "  - query: q1\n"
            "    expected_sources: [a.pdf]\n",
            encoding="utf-8",
        )

        gs = GoldenSet.from_yaml(str(path))
        assert len(gs.queries) == 1
        assert gs.queries[0].query == "q1"

    def test_yaml_with_filters(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.yaml"
        path.write_text(
            "- query: q1\n"
            "  expected_sources: [a.pdf]\n"
            "  filters:\n"
            "    company_name: apple\n",
            encoding="utf-8",
        )

        gs = GoldenSet.from_yaml(str(path))
        assert gs.queries[0].filters == {"company_name": "apple"}

    def test_yaml_single_string_expected(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.yaml"
        path.write_text(
            "- query: q1\n"
            "  expected_sources: a.pdf\n",
            encoding="utf-8",
        )

        gs = GoldenSet.from_yaml(str(path))
        assert gs.queries[0].expected_sources == ["a.pdf"]

    def test_yaml_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            GoldenSet.from_yaml("/nonexistent/golden.yaml")


class TestGoldenSetJSON:
    def test_load_from_json_list(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.json"
        data = [
            {"query": "q1", "expected_sources": ["a.pdf"], "category": "tech"},
            {"query": "q2", "expected_sources": ["b.pdf"]},
        ]
        path.write_text(json.dumps(data), encoding="utf-8")

        gs = GoldenSet.from_json(str(path))
        assert len(gs.queries) == 2
        assert gs.queries[0].query == "q1"
        assert gs.queries[0].category == "tech"

    def test_load_from_json_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.json"
        data = {"queries": [{"query": "q1", "expected_sources": ["a.pdf"]}]}
        path.write_text(json.dumps(data), encoding="utf-8")

        gs = GoldenSet.from_json(str(path))
        assert len(gs.queries) == 1

    def test_json_single_string_expected(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.json"
        data = [{"query": "q1", "expected_sources": "a.pdf"}]
        path.write_text(json.dumps(data), encoding="utf-8")

        gs = GoldenSet.from_json(str(path))
        assert gs.queries[0].expected_sources == ["a.pdf"]

    def test_json_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            GoldenSet.from_json("/nonexistent/golden.json")

    def test_json_missing_expected_key(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.json"
        data = [{"query": "q1"}]
        path.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ValueError, match="expected_sources"):
            GoldenSet.from_json(str(path))

    def test_json_mapping_without_queries_key(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.json"
        data = {"bad_key": [{"query": "q1", "expected_sources": ["a.pdf"]}]}
        path.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ValueError, match="queries"):
            GoldenSet.from_json(str(path))


class TestGoldenQuery:
    def test_defaults(self) -> None:
        gq = GoldenQuery(query="q", expected_sources=["a.pdf"])
        assert gq.expected_keywords == []
        assert gq.category == ""
        assert gq.difficulty == ""
        assert gq.filters is None


# ── Source matching ──────────────────────────────────────────────────────────


class TestMatchSource:
    def test_basename_ignores_directory(self) -> None:
        assert match_source("a.pdf", "/data/docs/a.pdf", mode="basename") is True

    def test_basename_case_insensitive(self) -> None:
        assert match_source("A.PDF", "/data/a.pdf", mode="basename") is True

    def test_basename_mismatch(self) -> None:
        assert match_source("a.pdf", "/data/b.pdf", mode="basename") is False

    def test_exact_full_match(self) -> None:
        assert match_source("/data/a.pdf", "/data/a.pdf", mode="exact") is True

    def test_exact_short_vs_long(self) -> None:
        assert match_source("/data/a.pdf", "a.pdf", mode="exact") is False

    def test_contains_substring(self) -> None:
        assert match_source("a.pdf", "/data/a.pdf", mode="contains") is True

    def test_contains_not_substring(self) -> None:
        assert match_source("b.pdf", "/data/a.pdf", mode="contains") is False

    def test_empty_actual_always_false(self) -> None:
        assert match_source("a.pdf", "", mode="basename") is False
        assert match_source("a.pdf", "", mode="exact") is False
        assert match_source("a.pdf", "", mode="contains") is False

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown match mode"):
            match_source("a.pdf", "a.pdf", mode="fuzzy")


# ── New metrics ──────────────────────────────────────────────────────────────


class TestRecallAtKNew:
    def test_all_found(self) -> None:
        assert new_recall_at_k(["a", "b", "c"], ["a", "c"], k=3) == 1.0

    def test_partial(self) -> None:
        assert new_recall_at_k(["a", "b"], ["a", "c"], k=2) == 0.5

    def test_none_found(self) -> None:
        assert new_recall_at_k(["x", "y"], ["a"], k=2) == 0.0

    def test_empty_expected(self) -> None:
        assert new_recall_at_k(["a"], [], k=5) == 0.0

    def test_k_boundary(self) -> None:
        assert new_recall_at_k(["x", "y", "a"], ["a"], k=2) == 0.0
        assert new_recall_at_k(["x", "y", "a"], ["a"], k=3) == 1.0

    def test_duplicate_hits_not_inflated(self) -> None:
        assert new_recall_at_k(["a", "a", "a"], ["a", "b"], k=3) == 0.5


class TestPrecisionAtKNew:
    def test_all_relevant(self) -> None:
        assert new_precision_at_k(["a", "b"], ["a", "b"], k=2) == 1.0

    def test_none_relevant(self) -> None:
        assert new_precision_at_k(["a", "b"], ["z"], k=2) == 0.0

    def test_half(self) -> None:
        assert new_precision_at_k(["a", "b"], ["a"], k=2) == 0.5

    def test_empty_retrieved(self) -> None:
        assert new_precision_at_k([], ["a"], k=5) == 0.0


class TestHitRateAtKNew:
    def test_hit(self) -> None:
        assert new_hit_rate_at_k(["a", "b"], ["b"], k=2) == 1.0

    def test_miss(self) -> None:
        assert new_hit_rate_at_k(["a", "b"], ["z"], k=2) == 0.0

    def test_respects_k(self) -> None:
        assert new_hit_rate_at_k(["x", "a"], ["a"], k=1) == 0.0
        assert new_hit_rate_at_k(["x", "a"], ["a"], k=2) == 1.0


class TestReciprocalRank:
    def test_first_rank(self) -> None:
        assert reciprocal_rank(["right", "wrong"], ["right"]) == 1.0

    def test_second_rank(self) -> None:
        assert reciprocal_rank(["wrong", "right"], ["right"]) == 0.5

    def test_no_match(self) -> None:
        assert reciprocal_rank(["wrong", "wrong"], ["right"]) == 0.0

    def test_empty(self) -> None:
        assert reciprocal_rank([], ["right"]) == 0.0

    def test_first_of_multiple_expected(self) -> None:
        assert reciprocal_rank(["a", "b"], ["a", "b"]) == 1.0


class TestKeywordCoverageNew:
    def test_all_found(self) -> None:
        assert new_keyword_coverage("revenue Q3 financial", ["revenue", "Q3"]) == 1.0

    def test_partial(self) -> None:
        assert new_keyword_coverage("revenue report", ["revenue", "Q3"]) == 0.5

    def test_none_found(self) -> None:
        assert new_keyword_coverage("hello world", ["revenue", "Q3"]) == 0.0

    def test_empty_keywords(self) -> None:
        assert new_keyword_coverage("anything", []) == 1.0

    def test_case_insensitive(self) -> None:
        assert new_keyword_coverage("Revenue Q3", ["revenue", "q3"]) == 1.0


# ── QueryMetrics / EvalResult dataclasses ────────────────────────────────────


class TestQueryMetrics:
    def test_construction(self) -> None:
        qm = QueryMetrics(
            recall_at_k=0.8,
            precision_at_k=0.6,
            hit_rate_at_k=1.0,
            mrr=0.75,
            keyword_coverage=0.9,
        )
        assert qm.recall_at_k == 0.8
        assert qm.keyword_coverage == 0.9


class TestEvalResult:
    def test_summary_table(self) -> None:
        er = EvalResult(
            queries=[],
            avg_recall=0.85,
            avg_precision=0.70,
            avg_hit_rate=0.90,
            avg_mrr=0.75,
            avg_keyword_coverage=0.80,
            total_queries=10,
        )
        table = er.summary_table()
        assert "recall" in table
        assert "precision" in table
        assert "0.850" in table
        assert "10" in table

    def test_defaults(self) -> None:
        er = EvalResult()
        assert er.total_queries == 0
        assert er.avg_recall == 0.0


# ── Sweep ────────────────────────────────────────────────────────────────────


class TestSweep:
    def test_two_variants(self) -> None:
        baseline = EvalResult(
            avg_recall=0.5, avg_precision=0.4, avg_hit_rate=0.6,
            avg_mrr=0.45, avg_keyword_coverage=0.7, total_queries=5,
        )
        improved = EvalResult(
            avg_recall=0.8, avg_precision=0.7, avg_hit_rate=0.9,
            avg_mrr=0.75, avg_keyword_coverage=0.85, total_queries=5,
        )

        result = sweep([("baseline", baseline), ("improved", improved)])
        assert "baseline" in result
        assert "improved" in result
        assert "Best recall: improved" in result
        assert "+0.30" in result

    def test_single_variant(self) -> None:
        v = EvalResult(avg_recall=0.5, total_queries=3)
        result = sweep([("only", v)])
        assert "only" in result
        assert "Best recall" not in result

    def test_empty(self) -> None:
        assert sweep([]) == "No variants were evaluated."

    def test_delta_sign(self) -> None:
        baseline = EvalResult(
            avg_recall=0.8, avg_precision=0.8, avg_hit_rate=0.8,
            avg_mrr=0.8, avg_keyword_coverage=0.8, total_queries=1,
        )
        worse = EvalResult(
            avg_recall=0.5, avg_precision=0.5, avg_hit_rate=0.5,
            avg_mrr=0.5, avg_keyword_coverage=0.5, total_queries=1,
        )

        result = sweep([("baseline", baseline), ("worse", worse)])
        assert "-0.30" in result
