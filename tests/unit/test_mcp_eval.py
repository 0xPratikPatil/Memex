"""Unit tests for rag_eval and rag_eval_sweep MCP tools."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from memex.mcp.schemas import EvalInput, EvalSweepInput
from memex.mcp.server import rag_eval, rag_eval_sweep


def _make_result(source: str, content: str = "some content") -> dict:
    """Build a minimal search result dict matching RAGEngine output format."""
    return {
        "id": f"chunk-{source}",
        "source": source,
        "content": content,
        "section_header": "",
        "context_prefix": "",
        "rrf_score": 0.0167,
        "rerank_score": 0.9,
        "doc_type": "report",
        "topics": [],
        "language": "en",
        "keywords": [],
        "entities": {},
        "dates": [],
    }


def _write_golden_yaml(path: Path, entries: list[dict]) -> None:
    """Write a golden set as YAML."""
    import yaml

    path.write_text(yaml.dump(entries), encoding="utf-8")


def _write_golden_json(path: Path, entries: list[dict]) -> None:
    """Write a golden set as JSON."""
    path.write_text(json.dumps(entries), encoding="utf-8")


# ── rag_eval tests ──────────────────────────────────────────────────────────


class TestRagEval:
    """Tests for the rag_eval MCP tool."""

    @pytest.mark.asyncio
    async def test_full_eval_with_matching_results(self, tmp_path: Path) -> None:
        """Should compute perfect scores when all expected sources are retrieved."""
        golden_path = tmp_path / "golden.yaml"
        _write_golden_yaml(
            golden_path,
            [
                {
                    "query": "What is revenue?",
                    "expected_sources": ["report.pdf"],
                    "expected_keywords": ["revenue"],
                },
            ],
        )

        mock_engine = MagicMock()
        mock_engine.hybrid_search.return_value = [
            _make_result("report.pdf", "Revenue was $10M."),
        ]

        with patch("memex.mcp.server._get_engine", return_value=mock_engine):
            raw = await rag_eval(EvalInput(golden_set_path=str(golden_path), top_k=1))
            result = json.loads(raw)

        assert result["total_queries"] == 1
        assert result["avg_recall"] == 1.0
        assert result["avg_precision"] == 1.0  # 1 hit / 1 slot = 1.0
        assert result["avg_hit_rate"] == 1.0
        assert result["avg_mrr"] == 1.0
        assert result["avg_keyword_coverage"] == 1.0
        assert result["queries"][0]["query"] == "What is revenue?"
        assert result["queries"][0]["retrieved_sources"] == ["report.pdf"]

    @pytest.mark.asyncio
    async def test_partial_recall(self, tmp_path: Path) -> None:
        """Should compute partial scores when only some expected sources found."""
        golden_path = tmp_path / "golden.yaml"
        _write_golden_yaml(
            golden_path,
            [
                {
                    "query": "Compare reports",
                    "expected_sources": ["report_a.pdf", "report_b.pdf"],
                },
            ],
        )

        mock_engine = MagicMock()
        mock_engine.hybrid_search.return_value = [
            _make_result("report_a.pdf", "Report A content."),
            _make_result("other.pdf", "Other content."),
        ]

        with patch("memex.mcp.server._get_engine", return_value=mock_engine):
            raw = await rag_eval(EvalInput(golden_set_path=str(golden_path), top_k=5))
            result = json.loads(raw)

        assert result["total_queries"] == 1
        assert abs(result["avg_recall"] - 0.5) < 0.001
        assert result["avg_hit_rate"] == 1.0  # at least one expected found
        assert result["avg_mrr"] == 1.0  # first result matched

    @pytest.mark.asyncio
    async def test_no_results(self, tmp_path: Path) -> None:
        """Should compute zero scores when nothing is retrieved."""
        golden_path = tmp_path / "golden.yaml"
        _write_golden_yaml(
            golden_path,
            [
                {"query": "missing", "expected_sources": ["gone.pdf"]},
            ],
        )

        mock_engine = MagicMock()
        mock_engine.hybrid_search.return_value = []

        with patch("memex.mcp.server._get_engine", return_value=mock_engine):
            raw = await rag_eval(EvalInput(golden_set_path=str(golden_path), top_k=5))
            result = json.loads(raw)

        assert result["avg_recall"] == 0.0
        assert result["avg_hit_rate"] == 0.0
        assert result["avg_mrr"] == 0.0

    @pytest.mark.asyncio
    async def test_empty_golden_set(self, tmp_path: Path) -> None:
        """Should return error when golden set has no queries."""
        golden_path = tmp_path / "golden.yaml"
        _write_golden_yaml(golden_path, [])

        with patch("memex.mcp.server._get_engine"):
            result = await rag_eval(EvalInput(golden_set_path=str(golden_path)))

        assert "Error" in result or "empty" in result.lower()

    @pytest.mark.asyncio
    async def test_missing_golden_set_file(self) -> None:
        """Should return error when golden set file does not exist."""
        result = await rag_eval(EvalInput(golden_set_path="/nonexistent/golden.yaml"))
        assert "Error" in result
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_json_golden_set(self, tmp_path: Path) -> None:
        """Should work with JSON format golden set."""
        golden_path = tmp_path / "golden.json"
        _write_golden_json(
            golden_path,
            [
                {
                    "query": "revenue",
                    "expected_sources": ["a.pdf"],
                    "expected_keywords": ["money"],
                },
            ],
        )

        mock_engine = MagicMock()
        mock_engine.hybrid_search.return_value = [
            _make_result("a.pdf", "Money revenue report."),
        ]

        with patch("memex.mcp.server._get_engine", return_value=mock_engine):
            raw = await rag_eval(EvalInput(golden_set_path=str(golden_path), top_k=5))
            result = json.loads(raw)

        assert result["total_queries"] == 1
        assert result["avg_recall"] == 1.0

    @pytest.mark.asyncio
    async def test_multiple_queries(self, tmp_path: Path) -> None:
        """Should aggregate metrics across multiple queries."""
        golden_path = tmp_path / "golden.yaml"
        _write_golden_yaml(
            golden_path,
            [
                {"query": "q1", "expected_sources": ["a.pdf"]},
                {"query": "q2", "expected_sources": ["b.pdf"]},
                {"query": "q3", "expected_sources": ["c.pdf"]},
            ],
        )

        mock_engine = MagicMock()
        # q1: perfect, q2: miss, q3: perfect
        mock_engine.hybrid_search.side_effect = [
            [_make_result("a.pdf")],
            [_make_result("wrong.pdf")],
            [_make_result("c.pdf")],
        ]

        with patch("memex.mcp.server._get_engine", return_value=mock_engine):
            raw = await rag_eval(EvalInput(golden_set_path=str(golden_path), top_k=5))
            result = json.loads(raw)

        assert result["total_queries"] == 3
        assert abs(result["avg_recall"] - 2.0 / 3.0) < 0.001
        assert abs(result["avg_hit_rate"] - 2.0 / 3.0) < 0.001

    @pytest.mark.asyncio
    async def test_search_exception_handled_gracefully(self, tmp_path: Path) -> None:
        """Should handle search failures without crashing."""
        golden_path = tmp_path / "golden.yaml"
        _write_golden_yaml(
            golden_path,
            [
                {"query": "q1", "expected_sources": ["a.pdf"]},
            ],
        )

        mock_engine = MagicMock()
        mock_engine.hybrid_search.side_effect = RuntimeError("Qdrant unreachable")

        with patch("memex.mcp.server._get_engine", return_value=mock_engine):
            raw = await rag_eval(EvalInput(golden_set_path=str(golden_path), top_k=5))
            result = json.loads(raw)

        # Should still return results, just with zero scores
        assert result["total_queries"] == 1
        assert result["avg_recall"] == 0.0

    @pytest.mark.asyncio
    async def test_keyword_coverage(self, tmp_path: Path) -> None:
        """Should compute keyword coverage from retrieved content."""
        golden_path = tmp_path / "golden.yaml"
        _write_golden_yaml(
            golden_path,
            [
                {
                    "query": "q1",
                    "expected_sources": ["a.pdf"],
                    "expected_keywords": ["revenue", "Q3", "growth"],
                },
            ],
        )

        mock_engine = MagicMock()
        mock_engine.hybrid_search.return_value = [
            _make_result("a.pdf", "Revenue and Q3 report showing growth."),
        ]

        with patch("memex.mcp.server._get_engine", return_value=mock_engine):
            raw = await rag_eval(EvalInput(golden_set_path=str(golden_path), top_k=5))
            result = json.loads(raw)

        assert result["avg_keyword_coverage"] == 1.0

    @pytest.mark.asyncio
    async def test_mrr_position(self, tmp_path: Path) -> None:
        """Should compute MRR based on first correct result position."""
        golden_path = tmp_path / "golden.yaml"
        _write_golden_yaml(
            golden_path,
            [
                {"query": "q1", "expected_sources": ["target.pdf"]},
            ],
        )

        mock_engine = MagicMock()
        mock_engine.hybrid_search.return_value = [
            _make_result("wrong.pdf"),
            _make_result("wrong2.pdf"),
            _make_result("target.pdf"),
        ]

        with patch("memex.mcp.server._get_engine", return_value=mock_engine):
            raw = await rag_eval(EvalInput(golden_set_path=str(golden_path), top_k=5))
            result = json.loads(raw)

        assert abs(result["avg_mrr"] - 1.0 / 3.0) < 0.001


# ── rag_eval_sweep tests ────────────────────────────────────────────────────


class TestRagEvalSweep:
    """Tests for the rag_eval_sweep MCP tool."""

    @pytest.mark.asyncio
    async def test_sweep_two_variants(self, tmp_path: Path) -> None:
        """Should compare two variants and produce a delta table."""
        golden_path = tmp_path / "golden.yaml"
        _write_golden_yaml(
            golden_path,
            [
                {"query": "q1", "expected_sources": ["a.pdf"]},
            ],
        )

        mock_engine = MagicMock()
        # variant 1: perfect retrieval; variant 2: miss
        mock_engine.hybrid_search.side_effect = [
            [_make_result("a.pdf")],  # baseline
            [_make_result("wrong.pdf")],  # improved (worse)
        ]

        with patch("memex.mcp.server._get_engine", return_value=mock_engine):
            raw = await rag_eval_sweep(
                EvalSweepInput(
                    golden_set_path=str(golden_path),
                    variants=[
                        {"name": "baseline", "rerank": False},
                        {"name": "no-rerank", "rerank": True},
                    ],
                    top_k=5,
                )
            )
            result = json.loads(raw)

        assert len(result["variants"]) == 2
        assert result["variants"][0]["avg_recall"] == 1.0
        assert result["variants"][1]["avg_recall"] == 0.0
        assert "delta_table" in result
        assert "baseline" in result["delta_table"]
        assert "no-rerank" in result["delta_table"]

    @pytest.mark.asyncio
    async def test_sweep_with_top_k_override(self, tmp_path: Path) -> None:
        """Should respect per-variant top_k override."""
        golden_path = tmp_path / "golden.yaml"
        _write_golden_yaml(
            golden_path,
            [
                {"query": "q1", "expected_sources": ["a.pdf"]},
            ],
        )

        mock_engine = MagicMock()
        mock_engine.hybrid_search.return_value = [_make_result("a.pdf")]

        with patch("memex.mcp.server._get_engine", return_value=mock_engine):
            raw = await rag_eval_sweep(
                EvalSweepInput(
                    golden_set_path=str(golden_path),
                    variants=[
                        {"name": "k3", "top_k": 3},
                    ],
                    top_k=5,
                )
            )
            result = json.loads(raw)

        assert result["variants"][0]["total_queries"] == 1
        # Verify hybrid_search was called with top_k=3 from variant
        call_kwargs = mock_engine.hybrid_search.call_args[1]
        assert call_kwargs["top_k"] == 3

    @pytest.mark.asyncio
    async def test_sweep_empty_golden_set(self, tmp_path: Path) -> None:
        """Should return error when golden set is empty."""
        golden_path = tmp_path / "golden.yaml"
        _write_golden_yaml(golden_path, [])

        with patch("memex.mcp.server._get_engine"):
            result = await rag_eval_sweep(
                EvalSweepInput(
                    golden_set_path=str(golden_path),
                    variants=[{"name": "v1"}],
                )
            )

        assert "Error" in result or "empty" in result.lower()

    @pytest.mark.asyncio
    async def test_sweep_missing_golden_set(self) -> None:
        """Should return error when golden set file does not exist."""
        result = await rag_eval_sweep(
            EvalSweepInput(
                golden_set_path="/nonexistent/golden.yaml",
                variants=[{"name": "v1"}],
            )
        )
        assert "Error" in result
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_sweep_best_recall_highlighted(self, tmp_path: Path) -> None:
        """Should identify the best variant by recall."""
        golden_path = tmp_path / "golden.yaml"
        _write_golden_yaml(
            golden_path,
            [
                {"query": "q1", "expected_sources": ["a.pdf"]},
            ],
        )

        mock_engine = MagicMock()
        mock_engine.hybrid_search.side_effect = [
            [_make_result("wrong.pdf")],  # baseline (0.0 recall)
            [_make_result("a.pdf")],  # better variant (1.0 recall)
        ]

        with patch("memex.mcp.server._get_engine", return_value=mock_engine):
            raw = await rag_eval_sweep(
                EvalSweepInput(
                    golden_set_path=str(golden_path),
                    variants=[
                        {"name": "weak"},
                        {"name": "strong"},
                    ],
                    top_k=5,
                )
            )
            result = json.loads(raw)

        assert "Best recall: strong" in result["delta_table"]

    @pytest.mark.asyncio
    async def test_sweep_search_exception_handled(self, tmp_path: Path) -> None:
        """Should handle search exceptions gracefully in sweep."""
        golden_path = tmp_path / "golden.yaml"
        _write_golden_yaml(
            golden_path,
            [
                {"query": "q1", "expected_sources": ["a.pdf"]},
            ],
        )

        mock_engine = MagicMock()
        mock_engine.hybrid_search.side_effect = RuntimeError("Backend down")

        with patch("memex.mcp.server._get_engine", return_value=mock_engine):
            raw = await rag_eval_sweep(
                EvalSweepInput(
                    golden_set_path=str(golden_path),
                    variants=[{"name": "v1"}],
                )
            )
            result = json.loads(raw)

        assert result["variants"][0]["avg_recall"] == 0.0
