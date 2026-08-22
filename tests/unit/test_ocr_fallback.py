"""Unit tests for loader.py — OCR fallback quality detection."""

from __future__ import annotations

from unittest.mock import MagicMock

from memex.engine.ingestion.loader import ConversionResult, _is_poor_quality, _ocr_to_conversion


class TestIsPoorQuality:
    def test_empty_markdown(self) -> None:
        result = ConversionResult(markdown="", status="success")
        assert _is_poor_quality(result, b"x" * 1000) is True

    def test_whitespace_only(self) -> None:
        result = ConversionResult(markdown="   \n\n  ", status="success")
        assert _is_poor_quality(result, b"x" * 1000) is True

    def test_short_text_large_file(self) -> None:
        result = ConversionResult(markdown="Short text", status="success")
        assert _is_poor_quality(result, b"x" * 50_000) is True

    def test_low_text_to_bytes_ratio(self) -> None:
        # 50 chars of text in 100KB file
        result = ConversionResult(markdown="x" * 50, status="success")
        assert _is_poor_quality(result, b"x" * 100_000) is True

    def test_normal_text_ok(self) -> None:
        text = "This is a normal document with enough content to pass the quality check. " * 10
        result = ConversionResult(markdown=text, status="success")
        assert _is_poor_quality(result, b"x" * 10_000) is False

    def test_good_ratio_ok(self) -> None:
        # 600 chars in 10KB file = 0.06 ratio (well above 0.005)
        result = ConversionResult(markdown="x" * 600, status="success")
        assert _is_poor_quality(result, b"x" * 10_000) is False


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
