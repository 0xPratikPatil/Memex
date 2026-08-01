"""Unit tests for memex.engine.sources.sync — sync engine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from memex.engine.sources.sync import SyncStats, _get_stored_hashes, _ingest_file, sync

# ── SyncStats tests ─────────────────────────────────────────────────────────


class TestSyncStats:
    def test_default_values(self) -> None:
        stats = SyncStats()
        assert stats.added == 0
        assert stats.changed == 0
        assert stats.deleted == 0
        assert stats.unchanged == 0
        assert stats.errors == []

    def test_summary_format(self) -> None:
        stats = SyncStats(added=2, changed=1, deleted=3, unchanged=5, errors=["e1", "e2"])
        s = stats.summary()
        assert "added=2" in s
        assert "changed=1" in s
        assert "deleted=3" in s
        assert "unchanged=5" in s
        assert "errors=2" in s

    def test_summary_empty(self) -> None:
        stats = SyncStats()
        s = stats.summary()
        assert "added=0" in s
        assert "errors=0" in s


# ── _get_stored_hashes tests ────────────────────────────────────────────────


class TestGetStoredHashes:
    def test_returns_hash_map(self) -> None:
        mock_point_1 = MagicMock()
        mock_point_1.payload = {"source": "/docs/a.pdf", "content_hash": "abc123"}
        mock_point_2 = MagicMock()
        mock_point_2.payload = {"source": "/docs/b.pdf", "content_hash": "def456"}

        mock_engine = MagicMock()
        mock_engine._get_qdrant.return_value.scroll.return_value = (
            [mock_point_1, mock_point_2],
            None,
        )

        with patch("memex.engine.sources.sync.config") as mock_config:
            mock_config.COLLECTION_NAME = "memex"
            result = _get_stored_hashes(mock_engine, "test-source")

        assert result == {"/docs/a.pdf": "abc123", "/docs/b.pdf": "def456"}

    def test_empty_collection(self) -> None:
        mock_engine = MagicMock()
        mock_engine._get_qdrant.return_value.scroll.return_value = ([], None)

        with patch("memex.engine.sources.sync.config") as mock_config:
            mock_config.COLLECTION_NAME = "memex"
            result = _get_stored_hashes(mock_engine, "test-source")

        assert result == {}

    def test_deduplicates_by_source_path(self) -> None:
        mock_point_1 = MagicMock()
        mock_point_1.payload = {"source": "/docs/a.pdf", "content_hash": "abc"}
        mock_point_2 = MagicMock()
        mock_point_2.payload = {"source": "/docs/a.pdf", "content_hash": "abc"}

        mock_engine = MagicMock()
        mock_engine._get_qdrant.return_value.scroll.return_value = (
            [mock_point_1, mock_point_2],
            None,
        )

        with patch("memex.engine.sources.sync.config") as mock_config:
            mock_config.COLLECTION_NAME = "memex"
            result = _get_stored_hashes(mock_engine, "test-source")

        assert len(result) == 1


# ── _ingest_file tests ──────────────────────────────────────────────────────


class TestIngestFile:
    def test_successful_ingest(self, tmp_path: Path) -> None:
        test_file = tmp_path / "doc.pdf"
        test_file.write_bytes(b"test content")

        mock_engine = MagicMock()
        mock_engine.compute_file_hash.return_value = "abc123"
        mock_engine.ingest_text.return_value = 5

        mock_result = MagicMock()
        mock_result.ok = True
        mock_result.markdown = "# Converted"
        mock_result.processing_time = 1.0

        with patch("memex.engine.ingestion.loader.parse_file", return_value=mock_result):
            count = _ingest_file(mock_engine, str(test_file), "my-source")

        assert count == 5
        mock_engine.ingest_text.assert_called_once()
        call_kwargs = mock_engine.ingest_text.call_args[1]
        assert call_kwargs["source_identifier"] == str(test_file)
        assert call_kwargs["content_hash"] == "abc123"
        assert call_kwargs["metadata"]["source_name"] == "my-source"

    def test_raises_on_docling_failure(self, tmp_path: Path) -> None:
        test_file = tmp_path / "bad.pdf"
        test_file.write_bytes(b"bad content")

        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.ok = False
        mock_result.status = "failure"
        mock_result.errors = ["conversion error"]

        with (
            patch("memex.engine.ingestion.loader.parse_file", return_value=mock_result),
            pytest.raises(RuntimeError, match="Docling conversion failed"),
        ):
            _ingest_file(mock_engine, str(test_file), "my-source")


# ── sync() tests ─────────────────────────────────────────────────────────────


class TestSync:
    @pytest.fixture
    def mock_yaml_config(self) -> MagicMock:
        cfg = MagicMock()
        cfg.get_list.return_value = []
        return cfg

    @pytest.fixture
    def mock_engine(self) -> MagicMock:
        engine = MagicMock()
        engine._get_qdrant.return_value = MagicMock()
        return engine

    @pytest.mark.asyncio
    async def test_no_sources_returns_empty_stats(self, mock_yaml_config: MagicMock) -> None:
        mock_yaml_config.get_list.return_value = []
        stats = await sync(mock_yaml_config)
        assert stats.added == 0
        assert stats.changed == 0
        assert stats.deleted == 0

    @pytest.mark.asyncio
    async def test_adds_new_files(self, mock_yaml_config: MagicMock) -> None:
        mock_yaml_config.get_list.return_value = [{"type": "local", "name": "docs", "path": "/tmp/docs"}]

        mock_source = MagicMock()
        mock_file = MagicMock()
        mock_file.path = "/tmp/docs/report.pdf"
        mock_file.name = "report.pdf"
        mock_source.list_files.return_value = [mock_file]
        mock_source.get_content_hash.return_value = "hash1"
        mock_source.download.return_value = Path("/tmp/docs/report.pdf")

        mock_qdrant = MagicMock()
        mock_qdrant.scroll.return_value = ([], None)  # no stored hashes

        mock_engine = MagicMock()
        mock_engine._get_qdrant.return_value = mock_qdrant
        mock_engine.compute_file_hash.return_value = "hash1"
        mock_engine.ingest_text.return_value = 3

        mock_parse_result = MagicMock()
        mock_parse_result.ok = True
        mock_parse_result.markdown = "# Report"
        mock_parse_result.processing_time = 1.0

        with (
            patch("memex.engine.sources.sync.get_source", return_value=mock_source),
            patch("memex.engine.core.pipeline.RAGEngine", return_value=mock_engine),
            patch("memex.engine.ingestion.loader.parse_file", return_value=mock_parse_result),
            patch("memex.engine.sources.sync.config") as mock_config,
        ):
            mock_config.COLLECTION_NAME = "memex"
            stats = await sync(mock_yaml_config)

        assert stats.added == 1
        assert stats.changed == 0
        assert stats.deleted == 0
        mock_engine.ingest_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_detects_changed_files(self, mock_yaml_config: MagicMock) -> None:
        mock_yaml_config.get_list.return_value = [{"type": "local", "name": "docs", "path": "/tmp/docs"}]

        mock_source = MagicMock()
        mock_file = MagicMock()
        mock_file.path = "/tmp/docs/report.pdf"
        mock_file.name = "report.pdf"
        mock_source.list_files.return_value = [mock_file]
        mock_source.get_content_hash.return_value = "new_hash"
        mock_source.download.return_value = Path("/tmp/docs/report.pdf")

        # Stored hash differs -> changed
        mock_point = MagicMock()
        mock_point.payload = {"source": "/tmp/docs/report.pdf", "content_hash": "old_hash"}

        mock_qdrant = MagicMock()
        mock_qdrant.scroll.return_value = ([mock_point], None)

        mock_engine = MagicMock()
        mock_engine._get_qdrant.return_value = mock_qdrant
        mock_engine.compute_file_hash.return_value = "new_hash"
        mock_engine.ingest_text.return_value = 2

        mock_parse_result = MagicMock()
        mock_parse_result.ok = True
        mock_parse_result.markdown = "# Updated"
        mock_parse_result.processing_time = 1.0

        with (
            patch("memex.engine.sources.sync.get_source", return_value=mock_source),
            patch("memex.engine.core.pipeline.RAGEngine", return_value=mock_engine),
            patch("memex.engine.ingestion.loader.parse_file", return_value=mock_parse_result),
            patch("memex.engine.sources.sync.config") as mock_config,
        ):
            mock_config.COLLECTION_NAME = "memex"
            stats = await sync(mock_yaml_config)

        assert stats.changed == 1
        assert stats.added == 0
        mock_engine.delete_by_source.assert_called_with("/tmp/docs/report.pdf")

    @pytest.mark.asyncio
    async def test_unchanged_files_not_modified(self, mock_yaml_config: MagicMock) -> None:
        mock_yaml_config.get_list.return_value = [{"type": "local", "name": "docs", "path": "/tmp/docs"}]

        mock_source = MagicMock()
        mock_file = MagicMock()
        mock_file.path = "/tmp/docs/report.pdf"
        mock_file.name = "report.pdf"
        mock_source.list_files.return_value = [mock_file]
        mock_source.get_content_hash.return_value = "same_hash"

        mock_point = MagicMock()
        mock_point.payload = {"source": "/tmp/docs/report.pdf", "content_hash": "same_hash"}

        mock_qdrant = MagicMock()
        mock_qdrant.scroll.return_value = ([mock_point], None)

        mock_engine = MagicMock()
        mock_engine._get_qdrant.return_value = mock_qdrant

        with (
            patch("memex.engine.sources.sync.get_source", return_value=mock_source),
            patch("memex.engine.core.pipeline.RAGEngine", return_value=mock_engine),
            patch("memex.engine.sources.sync.config") as mock_config,
        ):
            mock_config.COLLECTION_NAME = "memex"
            stats = await sync(mock_yaml_config)

        assert stats.unchanged == 1
        assert stats.added == 0
        mock_engine.ingest_text.assert_not_called()
        mock_engine.delete_by_source.assert_not_called()

    @pytest.mark.asyncio
    async def test_deletes_removed_files(self, mock_yaml_config: MagicMock) -> None:
        mock_yaml_config.get_list.return_value = [{"type": "local", "name": "docs", "path": "/tmp/docs"}]

        mock_source = MagicMock()
        mock_source.list_files.return_value = []  # no files -- all deleted

        # Stored hash for a file that no longer exists
        mock_point = MagicMock()
        mock_point.payload = {"source": "/tmp/docs/old.pdf", "content_hash": "old_hash"}

        mock_qdrant = MagicMock()
        mock_qdrant.scroll.return_value = ([mock_point], None)

        mock_engine = MagicMock()
        mock_engine._get_qdrant.return_value = mock_qdrant

        with (
            patch("memex.engine.sources.sync.get_source", return_value=mock_source),
            patch("memex.engine.core.pipeline.RAGEngine", return_value=mock_engine),
            patch("memex.engine.sources.sync.config") as mock_config,
        ):
            mock_config.COLLECTION_NAME = "memex"
            stats = await sync(mock_yaml_config)

        assert stats.deleted == 1
        mock_engine.delete_by_source.assert_called_with("/tmp/docs/old.pdf")

    @pytest.mark.asyncio
    async def test_dry_run_does_not_write(self, mock_yaml_config: MagicMock) -> None:
        mock_yaml_config.get_list.return_value = [{"type": "local", "name": "docs", "path": "/tmp/docs"}]

        mock_source = MagicMock()
        mock_file = MagicMock()
        mock_file.path = "/tmp/docs/new.pdf"
        mock_file.name = "new.pdf"
        mock_source.list_files.return_value = [mock_file]

        mock_qdrant = MagicMock()
        mock_qdrant.scroll.return_value = ([], None)

        mock_engine = MagicMock()
        mock_engine._get_qdrant.return_value = mock_qdrant

        with (
            patch("memex.engine.sources.sync.get_source", return_value=mock_source),
            patch("memex.engine.core.pipeline.RAGEngine", return_value=mock_engine),
            patch("memex.engine.sources.sync.config") as mock_config,
        ):
            mock_config.COLLECTION_NAME = "memex"
            stats = await sync(mock_yaml_config, dry_run=True)

        assert stats.added == 1
        mock_engine.ingest_text.assert_not_called()
        mock_engine.delete_by_source.assert_not_called()

    @pytest.mark.asyncio
    async def test_source_failure_suppresses_deletions(self, mock_yaml_config: MagicMock) -> None:
        mock_yaml_config.get_list.return_value = [
            {"type": "local", "name": "failing", "path": "/tmp/fail"},
            {"type": "local", "name": "good", "path": "/tmp/good"},
        ]

        failing_source = MagicMock()
        failing_source.list_files.side_effect = RuntimeError("network error")

        good_source = MagicMock()
        good_source.list_files.return_value = []

        # Stored hash for the good source
        mock_point = MagicMock()
        mock_point.payload = {"source": "/tmp/good/old.pdf", "content_hash": "abc"}

        mock_qdrant = MagicMock()
        mock_qdrant.scroll.return_value = ([mock_point], None)

        mock_engine = MagicMock()
        mock_engine._get_qdrant.return_value = mock_qdrant

        def get_source_side_effect(src_type, cfg):
            if cfg.get("name") == "failing":
                return failing_source
            return good_source

        with (
            patch("memex.engine.sources.sync.get_source", side_effect=get_source_side_effect),
            patch("memex.engine.core.pipeline.RAGEngine", return_value=mock_engine),
            patch("memex.engine.sources.sync.config") as mock_config,
        ):
            mock_config.COLLECTION_NAME = "memex"
            stats = await sync(mock_yaml_config)

        assert len(stats.errors) == 1
        assert "network error" in stats.errors[0]
        # Deletions should be suppressed because a source failed to list
        mock_engine.delete_by_source.assert_not_called()

    @pytest.mark.asyncio
    async def test_source_filter(self, mock_yaml_config: MagicMock) -> None:
        mock_yaml_config.get_list.return_value = [
            {"type": "local", "name": "docs", "path": "/tmp/docs"},
            {"type": "local", "name": "other", "path": "/tmp/other"},
        ]

        mock_source = MagicMock()
        mock_source.list_files.return_value = []

        mock_qdrant = MagicMock()
        mock_qdrant.scroll.return_value = ([], None)

        mock_engine = MagicMock()
        mock_engine._get_qdrant.return_value = mock_qdrant

        with (
            patch("memex.engine.sources.sync.get_source", return_value=mock_source),
            patch("memex.engine.core.pipeline.RAGEngine", return_value=mock_engine),
            patch("memex.engine.sources.sync.config") as mock_config,
        ):
            mock_config.COLLECTION_NAME = "memex"
            stats = await sync(mock_yaml_config, source_name="docs")

        # Only "docs" source should have been processed
        assert stats.added == 0
        assert stats.deleted == 0
