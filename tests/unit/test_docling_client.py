"""Unit tests for the legacy Docling path of the loader module."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from memex.engine.ingestion.loader import ConversionResult, _build_options, parse_local_file


@pytest.fixture(autouse=True)
def _force_docling_engine() -> None:
    """These tests exercise the legacy Docling conversion path — force the
    engine flag so the marker branch is not taken."""
    with patch("memex.engine.ingestion.loader.config.CONVERTER_ENGINE", "docling"):
        yield


# ── parse_local_file tests ──────────────────────────────────────────────────


class TestParseLocalFile:
    """Test direct file reading without file server."""

    def test_reads_file_directly_from_filesystem(self, tmp_path: Path) -> None:
        """Should read file bytes directly using pathlib, not HTTP."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)

        with patch("memex.engine.ingestion.loader._post") as mock_post:
            mock_post.return_value = {
                "document": {"md_content": "# Converted"},
                "processing_time": 1.0,
                "status": "success",
                "errors": [],
            }

            with patch("memex.engine.utils.cache.get_cached_parse_result", return_value=None):
                parse_local_file(str(test_file))

            # Verify _post was called with base64-encoded file content
            call_args = mock_post.call_args[0][0]
            sources = call_args["sources"]
            assert len(sources) == 1
            assert sources[0]["kind"] == "file"
            assert sources[0]["filename"] == "test.txt"

            # Verify base64 encoding is correct
            expected_b64 = base64.b64encode(test_content).decode("ascii")
            assert sources[0]["base64_string"] == expected_b64

    def test_raises_file_not_found_for_missing_file(self) -> None:
        """Should raise FileNotFoundError when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            parse_local_file("/nonexistent/path/file.txt")

    def test_returns_conversion_result(self, tmp_path: Path) -> None:
        """Should return ConversionResult with markdown content."""
        test_file = tmp_path / "test.md"
        test_file.write_bytes(b"# Test Document")

        with patch("memex.engine.ingestion.loader._post") as mock_post:
            mock_post.return_value = {
                "document": {"md_content": "# Converted Document"},
                "processing_time": 2.5,
                "status": "success",
                "errors": [],
            }

            with patch("memex.engine.utils.cache.get_cached_parse_result", return_value=None):
                result = parse_local_file(str(test_file))

            assert isinstance(result, ConversionResult)
            assert result.markdown == "# Converted Document"
            assert result.processing_time == 2.5
            assert result.status == "success"
            assert result.errors == []

    def test_caches_result_after_conversion(self, tmp_path: Path) -> None:
        """Should cache the conversion result for future requests."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"Test content")

        with patch("memex.engine.ingestion.loader._post") as mock_post:
            mock_post.return_value = {
                "document": {"md_content": "Converted"},
                "processing_time": 1.0,
                "status": "success",
                "errors": [],
            }

            with (
                patch("memex.engine.utils.cache.get_cached_parse_result", return_value=None),
                patch("memex.engine.utils.cache.cache_parse_result") as mock_cache,
            ):
                parse_local_file(str(test_file))

                # Verify cache was called with correct hash
                expected_hash = hashlib.sha256(str(test_file).encode()).hexdigest()[:16]
                mock_cache.assert_called_once()
                cache_key = mock_cache.call_args[0][0]
                assert cache_key == expected_hash

    def test_returns_cached_result_when_available(self, tmp_path: Path) -> None:
        """Should return cached result without calling Docling."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"Test content")

        cached_result = {
            "markdown": "Cached content",
            "status": "success",
            "processing_time": 0.5,
            "errors": [],
        }

        with patch("memex.engine.utils.cache.get_cached_parse_result", return_value=cached_result):
            result = parse_local_file(str(test_file))

            assert result.markdown == "Cached content"
            assert result.processing_time == 0.5

    @patch("memex.engine.ingestion.loader.config.DOCLING_PICTURE_CLASSIFY", True)
    def test_uses_correct_docling_options(self, tmp_path: Path) -> None:
        """Should include proper conversion options in payload."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"PDF content")

        with patch("memex.engine.ingestion.loader._post") as mock_post:
            mock_post.return_value = {
                "document": {"md_content": "Converted"},
                "processing_time": 1.0,
                "status": "success",
                "errors": [],
            }

            with patch("memex.engine.utils.cache.get_cached_parse_result", return_value=None):
                parse_local_file(str(test_file))

            call_args = mock_post.call_args[0][0]
            options = call_args["options"]

            assert "from_formats" in options
            assert "to_formats" in options
            assert "md" in options["to_formats"]
            assert "json" in options["to_formats"]
            assert "html" in options["to_formats"]
            assert "text" in options["to_formats"]
            assert "do_ocr" in options
            assert "table_mode" in options
            assert "do_table_structure" in options
            assert "image_export_mode" in options
            assert "do_picture_classification" in options
            assert "do_code_enrichment" not in options
            assert "do_formula_enrichment" not in options
            assert "do_chart_extraction" not in options
            assert "include_images" not in options


class TestBuildOptionsCodeEnrichment:
    def test_enrich_code_enabled_by_default(self):
        with patch("memex.engine.ingestion.loader.config.DOCLING_ENRICH_CODE", True):
            opts = _build_options()
        assert opts["do_code_enrichment"] is True

    def test_enrich_code_disabled_via_config(self):
        with patch("memex.engine.ingestion.loader.config.DOCLING_ENRICH_CODE", False):
            opts = _build_options()
        assert "do_code_enrichment" not in opts


class TestBuildOptionsFormulaEnrichment:
    def test_enrich_formula_enabled_by_default(self):
        with patch("memex.engine.ingestion.loader.config.DOCLING_ENRICH_FORMULA", True):
            opts = _build_options()
        assert opts["do_formula_enrichment"] is True

    def test_enrich_formula_disabled_via_config(self):
        with patch("memex.engine.ingestion.loader.config.DOCLING_ENRICH_FORMULA", False):
            opts = _build_options()
        assert "do_formula_enrichment" not in opts


class TestBuildOptionsPictureClassification:
    def test_picture_classify_enabled_by_default(self):
        with patch("memex.engine.ingestion.loader.config.DOCLING_PICTURE_CLASSIFY", True):
            opts = _build_options()
        assert opts["do_picture_classification"] is True

    def test_picture_classify_disabled_via_config(self):
        with patch("memex.engine.ingestion.loader.config.DOCLING_PICTURE_CLASSIFY", False):
            opts = _build_options()
        assert "do_picture_classification" not in opts


class TestBuildOptionsChartExtraction:
    def test_chart_extraction_enabled_by_default(self):
        with patch("memex.engine.ingestion.loader.config.DOCLING_CHART_EXTRACT", True):
            opts = _build_options()
        assert opts["do_chart_extraction"] is True


class TestBuildOptionsImageExport:
    def test_image_export_mode_embedded_by_default(self):
        with patch("memex.engine.ingestion.loader.config.DOCLING_IMAGE_EXPORT", "embedded"):
            opts = _build_options()
        assert opts["image_export_mode"] == "embedded"


class TestBuildOptionsPdfBackend:
    def test_no_pdf_backend_when_empty(self):
        with patch("memex.engine.ingestion.loader.config.DOCLING_PDF_BACKEND", ""):
            opts = _build_options()
        assert "pdf_backend" not in opts

    def test_pdf_backend_included_when_set(self):
        with patch("memex.engine.ingestion.loader.config.DOCLING_PDF_BACKEND", "DLPARSE_V4"):
            opts = _build_options()
        assert opts["pdf_backend"] == "dlparse_v4"


class TestBuildOptionsToFormats:
    def test_requests_all_output_formats(self):
        opts = _build_options()
        formats = opts["to_formats"]
        assert "md" in formats
        assert "json" in formats
        assert "html" in formats
        assert "text" in formats
