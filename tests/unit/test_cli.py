"""Unit tests for memex.cli — Typer CLI commands."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from memex.cli import app

runner = CliRunner()


class TestVersionFlag:
    def test_version_shows_version(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "memex" in result.output

    def test_help_shows_usage(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Memex RAG" in result.output
        assert "ingest" in result.output
        assert "sync" in result.output
        assert "eval" in result.output


class TestIngestCommand:
    @pytest.fixture(autouse=True)
    def _mock_status_store(self) -> MagicMock:
        """Replace FileStatusStore with a no-op mock for CLI ingest tests."""
        store = MagicMock()
        with patch("memex.engine.ingestion.status.FileStatusStore", return_value=store) as mock_cls:
            yield mock_cls

    def _configure_engine(self, mock_engine: MagicMock) -> None:
        """Configure dedup pre-checks to report 'not already ingested'."""
        mock_engine.check_unmodified_local.return_value = (False, 0)
        mock_engine.is_already_ingested.return_value = (False, 0)

    def test_ingest_missing_path_shows_error(self) -> None:
        result = runner.invoke(app, ["ingest", "/nonexistent/path"])
        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_ingest_empty_directory_shows_error(self, tmp_path) -> None:
        result = runner.invoke(app, ["ingest", str(tmp_path)])
        assert result.exit_code == 1
        assert "No files found" in result.output

    @patch("memex.engine.core.pipeline.RAGEngine")
    @patch("memex.engine.ingestion.loader.parse_file")
    def test_ingest_single_file(self, mock_parse_file, mock_engine_cls, tmp_path) -> None:
        test_file = tmp_path / "doc.md"
        test_file.write_text("# Hello")

        mock_result = MagicMock()
        mock_result.ok = True
        mock_result.markdown = "# Hello"
        mock_parse_file.return_value = mock_result

        mock_engine = MagicMock()
        mock_engine.compute_file_hash.return_value = "hash1"
        mock_engine.ingest_text.return_value = 3
        mock_engine_cls.return_value = mock_engine
        self._configure_engine(mock_engine)

        result = runner.invoke(app, ["ingest", str(test_file)])
        assert result.exit_code == 0
        assert "ingest complete" in result.output
        assert "1 ingested" in result.output

    @patch("memex.engine.core.pipeline.RAGEngine")
    @patch("memex.engine.ingestion.loader.parse_file")
    def test_ingest_with_recursive(self, mock_parse_file, mock_engine_cls, tmp_path) -> None:
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "a.md").write_text("# A")
        (sub / "b.md").write_text("# B")

        mock_result = MagicMock()
        mock_result.ok = True
        mock_result.markdown = "# content"
        mock_parse_file.return_value = mock_result

        mock_engine = MagicMock()
        mock_engine.compute_file_hash.return_value = "hash"
        mock_engine.ingest_text.return_value = 1
        mock_engine_cls.return_value = mock_engine
        self._configure_engine(mock_engine)

        result = runner.invoke(app, ["ingest", str(tmp_path), "--recursive"])
        assert result.exit_code == 0
        assert "ingest complete" in result.output
        assert "2 ingested" in result.output

    @patch("memex.engine.core.pipeline.RAGEngine")
    @patch("memex.engine.ingestion.loader.parse_file")
    def test_ingest_with_source_name(self, mock_parse_file, mock_engine_cls, tmp_path) -> None:
        test_file = tmp_path / "doc.md"
        test_file.write_text("# Hello")

        mock_result = MagicMock()
        mock_result.ok = True
        mock_result.markdown = "# Hello"
        mock_parse_file.return_value = mock_result

        mock_engine = MagicMock()
        mock_engine.compute_file_hash.return_value = "hash1"
        mock_engine.ingest_text.return_value = 2
        mock_engine_cls.return_value = mock_engine
        self._configure_engine(mock_engine)

        result = runner.invoke(app, ["ingest", str(test_file), "-s", "my-source"])
        assert result.exit_code == 0
        call_kwargs = mock_engine.ingest_text.call_args[1]
        assert call_kwargs["metadata"]["source_name"] == "my-source"

    @patch("memex.engine.core.pipeline.RAGEngine")
    @patch("memex.engine.ingestion.loader.parse_file")
    def test_ingest_verbose_enables_debug(self, mock_parse_file, mock_engine_cls, tmp_path) -> None:
        test_file = tmp_path / "doc.md"
        test_file.write_text("# Hello")

        mock_result = MagicMock()
        mock_result.ok = True
        mock_result.markdown = "# Hello"
        mock_parse_file.return_value = mock_result

        mock_engine = MagicMock()
        mock_engine.compute_file_hash.return_value = "hash1"
        mock_engine.ingest_text.return_value = 1
        mock_engine_cls.return_value = mock_engine
        self._configure_engine(mock_engine)

        result = runner.invoke(app, ["ingest", str(test_file), "--verbose"])
        assert result.exit_code == 0

    @patch("memex.engine.core.pipeline.RAGEngine")
    @patch("memex.engine.ingestion.loader.parse_file")
    def test_ingest_reports_docling_errors(self, mock_parse_file, mock_engine_cls, tmp_path) -> None:
        test_file = tmp_path / "bad.pdf"
        test_file.write_bytes(b"bad")

        mock_result = MagicMock()
        mock_result.ok = False
        mock_result.status = "failure"
        mock_result.errors = ["conversion error"]
        mock_parse_file.return_value = mock_result

        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(app, ["ingest", str(test_file)])
        assert result.exit_code == 1
        assert "errors" in result.output

    @patch("memex.engine.core.pipeline.RAGEngine")
    @patch("memex.engine.ingestion.loader.parse_file")
    def test_ingest_with_config_path(self, mock_parse_file, mock_engine_cls, tmp_path) -> None:
        test_file = tmp_path / "doc.md"
        test_file.write_text("# Hello")

        mock_result = MagicMock()
        mock_result.ok = True
        mock_result.markdown = "# Hello"
        mock_parse_file.return_value = mock_result

        mock_engine = MagicMock()
        mock_engine.compute_file_hash.return_value = "hash1"
        mock_engine.ingest_text.return_value = 1
        mock_engine_cls.return_value = mock_engine
        self._configure_engine(mock_engine)

        result = runner.invoke(app, ["ingest", str(test_file), "-c", "custom.yaml"])
        assert result.exit_code == 0


class TestIngestDedup:
    """CLI ingest should skip files already ingested."""

    @pytest.fixture(autouse=True)
    def _mock_status_store(self) -> MagicMock:
        """Replace FileStatusStore with a no-op mock for CLI ingest tests."""
        store = MagicMock()
        with patch("memex.engine.ingestion.status.FileStatusStore", return_value=store) as mock_cls:
            yield mock_cls

    @patch("memex.engine.core.pipeline.RAGEngine")
    @patch("memex.engine.ingestion.loader.parse_file")
    def test_skips_unchanged_local_file(self, mock_parse_file, mock_engine_cls, tmp_path) -> None:
        test_file = tmp_path / "doc.md"
        test_file.write_text("# Hello")

        mock_engine = MagicMock()
        mock_engine.check_unmodified_local.return_value = (True, 4)
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(app, ["ingest", str(test_file)])
        assert result.exit_code == 0
        mock_engine.ingest_text.assert_not_called()

    @patch("memex.engine.core.pipeline.RAGEngine")
    @patch("memex.engine.ingestion.loader.parse_file")
    def test_skips_same_content_hash(self, mock_parse_file, mock_engine_cls, tmp_path) -> None:
        test_file = tmp_path / "doc.md"
        test_file.write_text("# Hello")

        mock_result = MagicMock()
        mock_result.ok = True
        mock_result.markdown = "# Hello"
        mock_parse_file.return_value = mock_result

        mock_engine = MagicMock()
        mock_engine.check_unmodified_local.return_value = (False, 0)
        mock_engine.compute_file_hash.return_value = "hash1"
        mock_engine.is_already_ingested.return_value = (True, 7)
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(app, ["ingest", str(test_file)])
        assert result.exit_code == 0
        mock_engine.ingest_text.assert_not_called()


class TestSyncCommand:
    @patch("memex.engine.sources.sync.sync", new_callable=AsyncMock)
    def test_sync_default_options(self, mock_sync_fn) -> None:
        mock_stats = MagicMock()
        mock_stats.added = 0
        mock_stats.changed = 0
        mock_stats.deleted = 0
        mock_stats.unchanged = 5
        mock_stats.errors = []
        mock_sync_fn.return_value = mock_stats

        result = runner.invoke(app, ["sync"])
        assert result.exit_code == 0
        assert "5 unchanged" in result.output
        assert "5" in result.output

    @patch("memex.engine.sources.sync.sync", new_callable=AsyncMock)
    def test_sync_dry_run(self, mock_sync_fn) -> None:
        mock_stats = MagicMock()
        mock_stats.added = 2
        mock_stats.changed = 1
        mock_stats.deleted = 0
        mock_stats.unchanged = 0
        mock_stats.errors = []
        mock_sync_fn.return_value = mock_stats

        result = runner.invoke(app, ["sync", "--dry-run"])
        assert result.exit_code == 0
        assert "1 would change" in result.output
        assert "1" in result.output

    @patch("memex.engine.sources.sync.sync", new_callable=AsyncMock)
    def test_sync_with_source_name(self, mock_sync_fn) -> None:
        mock_stats = MagicMock()
        mock_stats.added = 0
        mock_stats.changed = 0
        mock_stats.deleted = 0
        mock_stats.unchanged = 0
        mock_stats.errors = []
        mock_sync_fn.return_value = mock_stats

        result = runner.invoke(app, ["sync", "-s", "docs"])
        assert result.exit_code == 0

    @patch("memex.engine.sources.sync.sync", new_callable=AsyncMock)
    def test_sync_reports_errors(self, mock_sync_fn) -> None:
        mock_stats = MagicMock()
        mock_stats.added = 0
        mock_stats.changed = 0
        mock_stats.deleted = 0
        mock_stats.unchanged = 0
        mock_stats.errors = ["source 'x' listing failed"]
        mock_sync_fn.return_value = mock_stats

        result = runner.invoke(app, ["sync"])
        assert result.exit_code == 1
        assert "listing failed" in result.output


class TestEvalCommand:
    def test_eval_missing_file(self) -> None:
        result = runner.invoke(app, ["eval", "/nonexistent/golden.yaml"])
        assert result.exit_code == 1
        assert "not found" in result.output

    @patch("memex.engine.core.pipeline.RAGEngine")
    @patch("memex.engine.core.yaml_config.YamlConfig")
    @patch("memex.engine.evaluation.golden.GoldenSet")
    def test_eval_with_file(self, mock_golden_cls, mock_config_cls, mock_engine_cls, tmp_path) -> None:
        golden = tmp_path / "golden.yaml"
        golden.write_text("queries: []")

        mock_golden = MagicMock()
        mock_golden.queries = []
        mock_golden_cls.from_yaml.return_value = mock_golden

        result = runner.invoke(app, ["eval", str(golden)])
        assert result.exit_code == 1
        assert "No queries" in result.output

    @patch("memex.engine.core.pipeline.RAGEngine")
    @patch("memex.engine.core.yaml_config.YamlConfig")
    @patch("memex.engine.evaluation.golden.GoldenSet")
    def test_eval_top_k_option(self, mock_golden_cls, mock_config_cls, mock_engine_cls, tmp_path) -> None:
        golden = tmp_path / "golden.yaml"
        golden.write_text("queries: []")

        mock_golden = MagicMock()
        mock_golden.queries = []
        mock_golden_cls.from_yaml.return_value = mock_golden

        result = runner.invoke(app, ["eval", str(golden), "--top-k", "10"])
        assert result.exit_code == 1
        assert "No queries" in result.output

    @patch("memex.engine.core.pipeline.RAGEngine")
    @patch("memex.engine.core.yaml_config.YamlConfig")
    @patch("memex.engine.evaluation.golden.GoldenSet")
    def test_eval_compare_rerank(self, mock_golden_cls, mock_config_cls, mock_engine_cls, tmp_path) -> None:
        golden = tmp_path / "golden.yaml"
        golden.write_text("queries: []")

        mock_golden = MagicMock()
        mock_golden.queries = []
        mock_golden_cls.from_yaml.return_value = mock_golden

        result = runner.invoke(app, ["eval", str(golden), "--compare-rerank"])
        assert result.exit_code == 1
        assert "No queries" in result.output

    @patch("memex.engine.core.pipeline.RAGEngine")
    @patch("memex.engine.core.yaml_config.YamlConfig")
    @patch("memex.engine.evaluation.golden.GoldenSet")
    def test_eval_verbose_flag(self, mock_golden_cls, mock_config_cls, mock_engine_cls, tmp_path) -> None:
        golden = tmp_path / "golden.yaml"
        golden.write_text("queries: []")

        mock_golden = MagicMock()
        mock_golden.queries = []
        mock_golden_cls.from_yaml.return_value = mock_golden

        result = runner.invoke(app, ["eval", str(golden), "--verbose"])
        assert result.exit_code == 1
        assert "No queries" in result.output


# ── Status command tests ─────────────────────────────────────────────────────


class TestStatusCommand:
    """memex status should show file processing status."""

    @patch("memex.engine.ingestion.status.FileStatusStore")
    @patch("memex.engine.core.pipeline.RAGEngine")
    @patch("memex.engine.core.yaml_config.YamlConfig")
    def test_status_shows_summary(self, mock_config_cls, mock_engine_cls, mock_store_cls) -> None:
        """Should display status summary table."""
        mock_store = MagicMock()
        mock_store.get_summary.return_value = {
            "pending": 2,
            "processing": 1,
            "done": 10,
            "skipped": 0,
            "retry": 1,
            "failed": 0,
        }
        mock_store.list_records.return_value = []
        mock_store_cls.return_value = mock_store

        mock_engine = MagicMock()
        mock_engine._get_qdrant.return_value = MagicMock()
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "File Processing" in result.output
        assert "10" in result.output

    @patch("memex.engine.ingestion.status.FileStatusStore")
    @patch("memex.engine.core.pipeline.RAGEngine")
    @patch("memex.engine.core.yaml_config.YamlConfig")
    def test_status_shows_failed_count(self, mock_config_cls, mock_engine_cls, mock_store_cls) -> None:
        """Should show failed count in red."""
        mock_store = MagicMock()
        mock_store.get_summary.return_value = {
            "pending": 0,
            "processing": 0,
            "done": 5,
            "skipped": 0,
            "retry": 0,
            "failed": 3,
        }
        mock_store.list_records.return_value = []
        mock_store_cls.return_value = mock_store

        mock_engine = MagicMock()
        mock_engine._get_qdrant.return_value = MagicMock()
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "3" in result.output
