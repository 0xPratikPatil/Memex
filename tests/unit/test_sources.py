from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from memex.engine.sources import Source, SourceFile, get_source, list_source_types, register_source
from memex.engine.sources.local import LocalSource


class TestSourceRegistry:
    def test_list_source_types_includes_local(self) -> None:
        assert "local" in list_source_types()

    def test_list_source_types_includes_s3(self) -> None:
        import memex.engine.sources.s3  # noqa: F401
        assert "s3" in list_source_types()

    def test_get_source_local(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = get_source("local", {"name": "test", "path": td})
            assert isinstance(src, LocalSource)

    def test_get_source_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown source type"):
            get_source("nonexistent", {})

    def test_register_source(self) -> None:
        @register_source
        class DummySource(Source):
            type = "_test_dummy"
            name = "dummy"
            extensions = []

            def list_files(self) -> list[SourceFile]:
                return []

            def get_content_hash(self, file: SourceFile) -> str:
                return ""

            def download(self, file: SourceFile, dest: Path) -> Path:
                return dest

        assert "_test_dummy" in list_source_types()
        src = get_source("_test_dummy", {})
        assert isinstance(src, DummySource)


class TestLocalSource:
    def test_list_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "a.txt").write_text("hello")
            (Path(td) / "b.pdf").write_bytes(b"\x00" * 100)
            (Path(td) / "sub").mkdir()
            (Path(td) / "sub" / "c.txt").write_text("world")

            src = LocalSource(name="test", path=td, extensions=[".txt"])
            files = src.list_files()
            names = sorted(f.name for f in files)
            assert names == ["a.txt", "c.txt"]

    def test_list_files_non_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "a.txt").write_text("hello")
            (Path(td) / "sub").mkdir()
            (Path(td) / "sub" / "b.txt").write_text("world")

            src = LocalSource(name="test", path=td, recursive=False)
            files = src.list_files()
            assert len(files) == 1
            assert files[0].name == "a.txt"

    def test_list_files_empty_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = LocalSource(name="test", path=td)
            assert src.list_files() == []

    def test_list_files_nonexistent_dir(self) -> None:
        src = LocalSource(name="test", path="/nonexistent/path")
        assert src.list_files() == []

    def test_get_content_hash(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "file.txt"
            p.write_text("hello world")

            src = LocalSource(name="test", path=td)
            f = src.list_files()[0]
            h = src.get_content_hash(f)

            expected = hashlib.sha256(b"hello world").hexdigest()
            assert h == expected

    def test_download(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src_dir = Path(td) / "src"
            dest_dir = Path(td) / "dest"
            src_dir.mkdir()
            dest_dir.mkdir()
            (src_dir / "file.txt").write_text("content")

            src = LocalSource(name="test", path=str(src_dir))
            f = src.list_files()[0]
            result = src.download(f, dest_dir)

            assert result.exists()
            assert result.read_text() == "content"

    def test_source_file_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "data.md"
            p.write_text("test")
            st = p.stat()

            src = LocalSource(name="test", path=td)
            files = src.list_files()
            assert len(files) == 1
            f = files[0]
            assert f.name == "data.md"
            assert f.path == str(p)
            assert f.size == st.st_size
            assert f.modified_at == st.st_mtime


class TestS3Source:
    @pytest.fixture
    def mock_boto3(self):
        with patch.dict("sys.modules", {"boto3": MagicMock()}) as modules:
            yield modules["boto3"]

    def _make_s3_source(self, **kwargs):
        with tempfile.TemporaryDirectory() as td:
            defaults = {
                "name": "test",
                "bucket": "my-bucket",
                "prefix": "docs/",
                "cache_dir": td,
            }
            defaults.update(kwargs)
            return LocalSource if False else None

    def test_s3_import_error(self) -> None:
        import sys

        boto3_backup = sys.modules.pop("boto3", None)
        sys.modules["boto3"] = None
        try:
            from memex.engine.sources.s3 import S3Source

            src = S3Source(name="test", bucket="b")
            with pytest.raises(ImportError, match="boto3"):
                src.list_files()
        finally:
            if boto3_backup is not None:
                sys.modules["boto3"] = boto3_backup
            else:
                sys.modules.pop("boto3", None)

    def test_s3_list_files(self, mock_boto3: MagicMock) -> None:
        from memex.engine.sources.s3 import S3Source

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        mock_client.get_paginator.return_value.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": "docs/report.pdf",
                        "Size": 1024,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                    {
                        "Key": "docs/data.csv",
                        "Size": 512,
                        "LastModified": datetime(2025, 1, 2, tzinfo=UTC),
                    },
                    {
                        "Key": "docs/subdir/",
                        "Size": 0,
                        "LastModified": datetime(2025, 1, 1, tzinfo=UTC),
                    },
                ]
            }
        ]

        src = S3Source(name="test", bucket="b", extensions=[".pdf"])
        files = src.list_files()
        assert len(files) == 1
        assert files[0].name == "report.pdf"
        assert files[0].path == "docs/report.pdf"
        assert files[0].size == 1024

    def test_s3_download_uses_cache(self, mock_boto3: MagicMock) -> None:
        from memex.engine.sources.s3 import S3Source

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        with tempfile.TemporaryDirectory() as td:
            src = S3Source(name="test", bucket="b", cache_dir=td)
            sf = SourceFile(name="file.txt", path="key/file.txt", size=100, modified_at=1000.0)

            dest = Path(td) / "out"
            dest.mkdir()
            target = dest / "file.txt"
            target.write_bytes(b"\x00" * 100)

            now = os.stat(target).st_mtime
            os.utime(target, (now, sf.modified_at))

            result = src.download(sf, dest)
            mock_client.download_file.assert_not_called()
            assert result == target

    def test_s3_download_fetches(self, mock_boto3: MagicMock) -> None:
        from memex.engine.sources.s3 import S3Source

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        with tempfile.TemporaryDirectory() as td:
            src = S3Source(name="test", bucket="b", cache_dir=td)
            sf = SourceFile(name="file.txt", path="key/file.txt", size=100, modified_at=1000.0)

            dest = Path(td) / "out"
            dest.mkdir()

            result = src.download(sf, dest)
            mock_client.download_file.assert_called_once_with("b", "key/file.txt", str(result))
