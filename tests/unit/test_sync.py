"""Unit tests for memex.engine.sources.sync — sync engine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from memex.engine.core.errors import IngestionError
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

    def test_filters_by_source_name(self) -> None:
        mock_point = MagicMock()
        mock_point.payload = {"source": "/docs/a.pdf", "content_hash": "abc"}

        mock_qdrant = MagicMock()
        mock_qdrant.scroll.return_value = ([mock_point], None)

        mock_engine = MagicMock()
        mock_engine._get_qdrant.return_value = mock_qdrant

        with patch("memex.engine.sources.sync.config") as mock_config:
            mock_config.COLLECTION_NAME = "memex"
            _get_stored_hashes(mock_engine, "my-source")

        # Filter must match the stored source_name payload field, not `source`
        import qdrant_client.models

        call_kwargs = mock_qdrant.scroll.call_args[1]
        flt = call_kwargs["scroll_filter"]
        assert isinstance(flt, qdrant_client.models.Filter)
        assert len(flt.must) == 1
        condition = flt.must[0]
        assert isinstance(condition, qdrant_client.models.FieldCondition)
        assert condition.key == "source_name"
        assert condition.match.value == "my-source"


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

        with (
            patch("memex.engine.ingestion.loader.parse_file", return_value=mock_result),
            patch("memex.engine.sources.sync.config") as mock_config,
        ):
            mock_config.CHUNK_STRATEGY = "recursive"
            count = _ingest_file(mock_engine, str(test_file), str(test_file), "my-source")

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
            patch("memex.engine.sources.sync.config") as mock_config,
            pytest.raises(IngestionError, match="conversion failed"),
        ):
            mock_config.CHUNK_STRATEGY = "recursive"
            _ingest_file(mock_engine, str(test_file), str(test_file), "my-source")


# ── sync() tests ─────────────────────────────────────────────────────────────


class TestSync:
    @pytest.fixture(autouse=True)
    def _mock_status_store(self) -> MagicMock:
        """Replace FileStatusStore with a no-op mock for sync reconciliation tests.

        The store is exercised end-to-end in test_status.py; here we only test
        the sync reconciliation logic against a stateless mock qdrant.
        """
        store = MagicMock()
        store.get_due_retries.return_value = []
        with patch("memex.engine.sources.sync.FileStatusStore", return_value=store) as mock_cls:
            yield mock_cls

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
            mock_config.MAX_CONCURRENT_SYNC = 8
            stats = await sync(mock_yaml_config)

        assert stats.added == 1
        assert stats.changed == 0
        assert stats.deleted == 0
        mock_engine.ingest_text.assert_called_once()
        # source_identifier must be the real source path, not a temp path
        call_kwargs = mock_engine.ingest_text.call_args[1]
        assert call_kwargs["source_identifier"] == "/tmp/docs/report.pdf"

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
            mock_config.MAX_CONCURRENT_SYNC = 8
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
            mock_config.MAX_CONCURRENT_SYNC = 8
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
            mock_config.MAX_CONCURRENT_SYNC = 8
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
            mock_config.MAX_CONCURRENT_SYNC = 8
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
            mock_config.MAX_CONCURRENT_SYNC = 8
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
            mock_config.MAX_CONCURRENT_SYNC = 8
            stats = await sync(mock_yaml_config, source_name="docs")

        # Only "docs" source should have been processed
        assert stats.added == 0
        assert stats.deleted == 0


class TestConversionAhead:
    """Two-stage pipeline: conversions run ahead of the serialized ingest stage."""

    @pytest.fixture(autouse=True)
    def _mock_status_store(self) -> MagicMock:
        store = MagicMock()
        store.get_due_retries.return_value = []
        with patch("memex.engine.sources.sync.FileStatusStore", return_value=store):
            yield store

    @pytest.mark.asyncio
    async def test_conversions_run_ahead_of_ingest(self) -> None:
        """With 2 worker slots, all 3 conversions must start before the first
        ingest finishes — conversions are NOT inlined with the LLM ingest
        stage (which is what left the converter queues idle)."""
        import threading
        import time

        from memex.engine.sources.sync import sync

        started: list[str] = []
        started_lock = threading.Lock()
        ingest_seen: list[int] = []
        ingest_lock = threading.Lock()

        def fake_convert(engine, file_path, source_identifier, progress_cb=None, file_idx=0, total_files=0):
            with started_lock:
                started.append(source_identifier)
            time.sleep(0.05)
            return {"markdown": f"# {source_identifier}", "chunks": None}

        def fake_ingest(engine, markdown, chunks, source_identifier, file_path, source_name, progress_cb=None):
            with started_lock:
                n = len(started)
            with ingest_lock:
                ingest_seen.append(n)
            time.sleep(0.3)
            return 1

        cfg = MagicMock()
        cfg.get_list.return_value = [{"type": "local", "name": "docs", "path": "/tmp/docs"}]

        mock_source = MagicMock()
        files = [MagicMock() for _ in range(5)]
        for i, f in enumerate(files):
            f.path = f"/tmp/docs/doc{i}.pdf"
            f.name = f"doc{i}.pdf"
        mock_source.list_files.return_value = files
        mock_source.download.side_effect = [Path(f"/tmp/docs/doc{i}.pdf") for i in range(5)]

        mock_qdrant = MagicMock()
        mock_qdrant.scroll.return_value = ([], None)

        mock_engine = MagicMock()
        mock_engine._get_qdrant.return_value = mock_qdrant
        mock_engine.compute_file_hash.return_value = "hash"

        with (
            patch("memex.engine.sources.sync.get_source", return_value=mock_source),
            patch("memex.engine.core.pipeline.RAGEngine", return_value=mock_engine),
            patch("memex.engine.sources.sync._convert_file", side_effect=fake_convert),
            patch("memex.engine.sources.sync._ingest_markdown", side_effect=fake_ingest),
            patch("memex.engine.sources.sync.config") as mock_config,
        ):
            mock_config.COLLECTION_NAME = "memex"
            mock_config.MAX_CONCURRENT_SYNC = 2
            stats = await sync(cfg)

        assert stats.added == 5
        # The first ingest must not have started until all conversions were
        # underway — conversions are a separate, parallel stage.
        assert ingest_seen[0] >= 3, f"only {ingest_seen[0]} conversions started before first ingest (inline design)"


class TestBoundedConversionWaves:
    """Conversions run in bounded just-in-time waves, not an upfront burst."""

    @pytest.fixture(autouse=True)
    def _mock_status_store(self) -> MagicMock:
        store = MagicMock()
        store.get_due_retries.return_value = []
        with patch("memex.engine.sources.sync.FileStatusStore", return_value=store):
            yield store

    @pytest.mark.asyncio
    async def test_wave_is_bounded_no_upfront_burst(self) -> None:
        """12 files: at most 8 conversions submitted before the first ingest —
        conversions are paced in waves, not all submitted at once."""
        from memex.engine.sources.sync import sync as sync_fn

        submissions: list[str] = []
        ingest_seen: list[int] = []
        started: list[str] = []
        import threading
        import time
        import concurrent.futures

        start_lock = threading.Lock()
        ingest_lock = threading.Lock()

        def fake_convert(engine, file_path, source_identifier, progress_cb=None, file_idx=0, total_files=0):
            with start_lock:
                started.append(source_identifier)
            time.sleep(0.02)
            return {"markdown": f"# {source_identifier}", "chunks": None}

        max_inflight = [0]

        def fake_ingest(engine, markdown, chunks, source_identifier, file_path, source_name, progress_cb=None):
            with ingest_lock:
                ingest_seen.append(len(submissions))
            time.sleep(0.05)
            return 1

        def fake_convert_wrapped(engine, file_path, source_identifier, progress_cb=None, file_idx=0, total_files=0):
            try:
                return fake_convert(engine, file_path, source_identifier, progress_cb, file_idx, total_files)
            finally:
                with ingest_lock:
                    max_inflight[0] = max(max_inflight[0], len(submissions) - len(started))

        cfg = MagicMock()
        cfg.get_list.return_value = [{"type": "local", "name": "docs", "path": "/tmp/docs"}]
        mock_source = MagicMock()
        files = [MagicMock() for _ in range(12)]
        for i, f in enumerate(files):
            f.path = f"/tmp/docs/doc{i}.pdf"
            f.name = f"doc{i}.pdf"
        mock_source.list_files.return_value = files
        mock_source.download.side_effect = [Path(f"/tmp/docs/doc{i}.pdf") for i in range(12)]
        mock_qdrant = MagicMock()
        mock_qdrant.scroll.return_value = ([], None)
        mock_engine = MagicMock()
        mock_engine._get_qdrant.return_value = mock_qdrant
        mock_engine.compute_file_hash.return_value = "hash"

        orig_executor = concurrent.futures.ThreadPoolExecutor

        class _SpyExecutor(orig_executor):
            def submit(self, fn, *args, **kwargs):
                # conversion submits pass a path str; ingest submits pass a
                # tuple — only count conversions
                if args and isinstance(args[0], str):
                    submissions.append(args[0])
                return super().submit(fn, *args, **kwargs)

        with (
            patch("memex.engine.sources.sync.get_source", return_value=mock_source),
            patch("memex.engine.core.pipeline.RAGEngine", return_value=mock_engine),
            patch("memex.engine.sources.sync._convert_file", side_effect=fake_convert_wrapped),
            patch("memex.engine.sources.sync._ingest_markdown", side_effect=fake_ingest),
            patch("memex.engine.sources.sync.config") as mock_config,
            patch.object(concurrent.futures, "ThreadPoolExecutor", _SpyExecutor),
        ):
            mock_config.COLLECTION_NAME = "memex"
            mock_config.MAX_CONCURRENT_SYNC = 2
            stats = await sync_fn(cfg)

        assert stats.added == 12
        assert ingest_seen, "no ingest ran"
        # Bounded wave: in-flight conversions never exceed the wave size —
        # the upfront burst submitted ALL 12 at once.
        assert max_inflight[0] <= 8, (
            f"max {max_inflight[0]} conversions in flight — unbounded upfront burst"
        )
        assert ingest_seen[0] >= 3, "conversions must run ahead of ingest"

    @pytest.mark.asyncio
    async def test_converted_files_emit_queued_stage(self) -> None:
        """Converted files waiting for the LLM consumer emit 'Queued', not a
        stale 'Converting' — so rows show the truth while queues work ahead."""
        import threading
        import time

        from memex.engine.sources.sync import sync

        stages: list[str] = []

        def cb(p):
            stages.append(p.stage)

        def fake_convert(engine, file_path, source_identifier, progress_cb=None, file_idx=0, total_files=0):
            time.sleep(0.01)
            return {"markdown": f"# {source_identifier}", "chunks": None}

        def fake_ingest(engine, markdown, chunks, source_identifier, file_path, source_name, progress_cb=None):
            return 1

        cfg = MagicMock()
        cfg.get_list.return_value = [{"type": "local", "name": "docs", "path": "/tmp/docs"}]
        mock_source = MagicMock()
        files = [MagicMock() for _ in range(3)]
        for i, f in enumerate(files):
            f.path = f"/tmp/docs/doc{i}.pdf"
            f.name = f"doc{i}.pdf"
        mock_source.list_files.return_value = files
        mock_source.download.side_effect = [Path(f"/tmp/docs/doc{i}.pdf") for i in range(3)]
        mock_qdrant = MagicMock()
        mock_qdrant.scroll.return_value = ([], None)
        mock_engine = MagicMock()
        mock_engine._get_qdrant.return_value = mock_qdrant
        mock_engine.compute_file_hash.return_value = "hash"

        with (
            patch("memex.engine.sources.sync.get_source", return_value=mock_source),
            patch("memex.engine.core.pipeline.RAGEngine", return_value=mock_engine),
            patch("memex.engine.sources.sync._convert_file", side_effect=fake_convert),
            patch("memex.engine.sources.sync._ingest_markdown", side_effect=fake_ingest),
            patch("memex.engine.sources.sync.config") as mock_config,
        ):
            mock_config.COLLECTION_NAME = "memex"
            mock_config.MAX_CONCURRENT_SYNC = 2
            await sync(cfg, progress_cb=cb)

        assert "Queued" in stages, f"no Queued stage emitted; saw {stages}"
        conv_idx = stages.index("Converting")
        queued_idx = stages.index("Queued")
        assert queued_idx > conv_idx


class TestParallelIngestStage:
    """The ingest stage runs concurrently — multiple files through the LLM
    pipeline at once (bounded by workers), so conversion queues never idle."""

    @pytest.fixture(autouse=True)
    def _mock_status_store(self) -> MagicMock:
        store = MagicMock()
        store.get_due_retries.return_value = []
        with patch("memex.engine.sources.sync.FileStatusStore", return_value=store):
            yield store

    @pytest.mark.asyncio
    async def test_ingest_stage_runs_concurrently(self) -> None:
        import threading
        import time

        from memex.engine.sources.sync import sync

        cur = [0]
        max_concurrent = [0]
        state_lock = threading.Lock()

        def fake_convert(engine, file_path, source_identifier, progress_cb=None, file_idx=0, total_files=0):
            time.sleep(0.01)
            return {"markdown": f"# {source_identifier}", "chunks": None}

        def fake_ingest(engine, markdown, chunks, source_identifier, file_path, source_name, progress_cb=None):
            with state_lock:
                cur[0] += 1
                max_concurrent[0] = max(max_concurrent[0], cur[0])
            time.sleep(0.1)
            with state_lock:
                cur[0] -= 1
            return 1

        cfg = MagicMock()
        cfg.get_list.return_value = [{"type": "local", "name": "docs", "path": "/tmp/docs"}]
        mock_source = MagicMock()
        files = [MagicMock() for _ in range(4)]
        for i, f in enumerate(files):
            f.path = f"/tmp/docs/doc{i}.pdf"
            f.name = f"doc{i}.pdf"
        mock_source.list_files.return_value = files
        mock_source.download.side_effect = [Path(f"/tmp/docs/doc{i}.pdf") for i in range(4)]
        mock_qdrant = MagicMock()
        mock_qdrant.scroll.return_value = ([], None)
        mock_engine = MagicMock()
        mock_engine._get_qdrant.return_value = mock_qdrant
        mock_engine.compute_file_hash.return_value = "hash"

        with (
            patch("memex.engine.sources.sync.get_source", return_value=mock_source),
            patch("memex.engine.core.pipeline.RAGEngine", return_value=mock_engine),
            patch("memex.engine.sources.sync._convert_file", side_effect=fake_convert),
            patch("memex.engine.sources.sync._ingest_markdown", side_effect=fake_ingest),
            patch("memex.engine.sources.sync.config") as mock_config,
        ):
            mock_config.COLLECTION_NAME = "memex"
            mock_config.MAX_CONCURRENT_SYNC = 2
            stats = await sync(cfg)

        assert stats.added == 4
        assert max_concurrent[0] >= 2, (
            f"ingest ran with max concurrency {max_concurrent[0]} — "
            "LLM pipeline is serialized (single consumer)"
        )
