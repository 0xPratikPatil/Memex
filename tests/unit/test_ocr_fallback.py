"""Unit tests for loader.py — OCR fallback quality detection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from memex.engine.core.errors import CorruptedDocumentError, ServiceUnavailableError
from memex.engine.ingestion.loader import ConversionResult, _is_poor_quality, _ocr_to_conversion


class TestIsPoorQuality:
    def test_empty_markdown(self) -> None:
        result = ConversionResult(markdown="", status="success")
        assert _is_poor_quality(result, b"x" * 1000, "scan.pdf") is True

    def test_whitespace_only(self) -> None:
        result = ConversionResult(markdown="   \n\n  ", status="success")
        assert _is_poor_quality(result, b"x" * 1000, "scan.pdf") is True

    def test_short_text_large_file(self) -> None:
        result = ConversionResult(markdown="Short text", status="success")
        assert _is_poor_quality(result, b"x" * 50_000, "scan.pdf") is True

    def test_low_text_to_bytes_ratio(self) -> None:
        # 50 chars of text in 100KB file
        result = ConversionResult(markdown="x" * 50, status="success")
        assert _is_poor_quality(result, b"x" * 100_000, "scan.pdf") is True

    def test_normal_text_ok(self) -> None:
        text = "This is a normal document with enough content to pass the quality check. " * 10
        result = ConversionResult(markdown=text, status="success")
        assert _is_poor_quality(result, b"x" * 10_000, "doc.pdf") is False

    def test_good_ratio_ok(self) -> None:
        # 600 chars in 10KB file = 0.06 ratio (well above 0.005)
        result = ConversionResult(markdown="x" * 600, status="success")
        assert _is_poor_quality(result, b"x" * 10_000, "doc.pdf") is False

    def test_non_ocrable_formats_never_trigger_ocr(self) -> None:
        """DOCX/audio/etc. never trigger OCR even with poor output."""
        for filename in ("doc.docx", "audio.mp3", "sheet.xlsx", "notes.txt", "page.html"):
            result = ConversionResult(markdown="", status="success")
            assert _is_poor_quality(result, b"x" * 50_000, filename) is False

    def test_image_files_are_ocrable(self) -> None:
        result = ConversionResult(markdown="", status="success")
        assert _is_poor_quality(result, b"x" * 50_000, "photo.png") is True
        assert _is_poor_quality(result, b"x" * 50_000, "scan.jpeg") is True
        assert _is_poor_quality(result, b"x" * 50_000, "scan.tiff") is True


class TestOcrToConversion:
    def test_successful_ocr(self) -> None:
        ocr_result = MagicMock()
        ocr_result.markdown = "Extracted text"
        ocr_result.ok = True
        ocr_result.processing_time = 2.5

        conv = _ocr_to_conversion(ocr_result)
        assert conv.markdown == "Extracted text"
        assert conv.status == "success"
        assert conv.processing_time == 2.5
        assert conv.errors == []

    def test_failed_ocr(self) -> None:
        ocr_result = MagicMock()
        ocr_result.markdown = ""
        ocr_result.ok = False
        ocr_result.processing_time = 1.0

        conv = _ocr_to_conversion(ocr_result)
        assert conv.status == "error"
        assert len(conv.errors) == 1


class TestOcrInlineFlow:
    """OCR runs inline when MarkItDown produces poor quality output."""

    def _setup(self, tmp_path: Path):
        f = tmp_path / "scanned.pdf"
        f.write_bytes(b"%PDF-1.4\n" + b"x" * 100_000)
        return str(f)

    def test_ocr_runs_inline_for_scanned_pdf(self, tmp_path: Path) -> None:
        from memex.engine.ingestion.loader import parse_local_file

        f = self._setup(tmp_path)
        fake_ocr = MagicMock()
        fake_ocr.markdown = "OCR text"
        fake_ocr.ok = True
        fake_ocr.processing_time = 1.0

        with (
            patch("memex.engine.ingestion.loader.config.CONVERTER_ENGINE", "markitdown"),
            patch(
                "memex.engine.ingestion.markitdown_client.convert_markdown",
                side_effect=CorruptedDocumentError("empty"),
            ),
            patch("memex.engine.utils.cache.get_cached_parse_result", return_value=None),
            patch("memex.engine.utils.cache.cache_parse_result"),
            patch("memex.engine.ingestion.ocr_client.is_ocr_available", return_value=True),
            patch("memex.engine.ingestion.ocr_client.convert_with_ocr", return_value=fake_ocr),
        ):
            result = parse_local_file(f)

        assert result.ok is True
        assert result.markdown == "OCR text"

    def test_markitdown_outage_raises_for_retry(self, tmp_path: Path) -> None:
        """MarkItDown unreachable (container down) → raises for auto-retry, NOT OCR."""
        import pytest

        from memex.engine.ingestion.loader import parse_local_file

        f = self._setup(tmp_path)
        with (
            patch("memex.engine.ingestion.loader.config.CONVERTER_ENGINE", "markitdown"),
            patch(
                "memex.engine.ingestion.markitdown_client.convert_markdown",
                side_effect=ServiceUnavailableError("MarkItDown", "cannot reach http://localhost:5003"),
            ),
            patch("memex.engine.utils.cache.get_cached_parse_result", return_value=None),
            pytest.raises(ServiceUnavailableError),
        ):
            parse_local_file(f)
