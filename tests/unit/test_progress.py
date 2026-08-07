"""Tests for progress tracking data model."""

from memex.engine.core.progress import FileProgress


class TestFileProgress:
    def test_construction(self) -> None:
        p = FileProgress(path="/docs/report.pdf", total=10, current=3, stage="Parsing")
        assert p.path == "/docs/report.pdf"
        assert p.total == 10
        assert p.current == 3
        assert p.stage == "Parsing"
        assert p.chunks == 0
        assert p.error == ""

    def test_with_chunks(self) -> None:
        p = FileProgress(path="/docs/report.pdf", total=10, current=3, stage="Done", chunks=42)
        assert p.chunks == 42
        assert p.error == ""

    def test_with_error(self) -> None:
        p = FileProgress(path="/docs/report.pdf", total=10, current=3, stage="Error", error="parse failed")
        assert p.error == "parse failed"
        assert p.chunks == 0

    def test_callback_type(self) -> None:
        calls: list[FileProgress] = []

        def cb(p: FileProgress) -> None:
            calls.append(p)

        p = FileProgress(path="/test.md", total=1, current=1, stage="Done")
        cb(p)
        assert len(calls) == 1
        assert calls[0].stage == "Done"
