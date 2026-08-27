"""Unit tests for memex.engine.converter — fast-mode model/processor filtering.

TDD: these tests define the expected behavior of build_converter_args().
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ── Fake marker classes (real classes so __module__/__qualname__ work) ──────


class _FakeOrderProc:
    pass


class _FakeTableProc:
    pass


class _FakeTextProc:
    pass


_FakeOrderProc.__module__ = "marker.processors.order"
_FakeOrderProc.__qualname__ = "OrderProcessor"
_FakeTableProc.__module__ = "marker.processors.table"
_FakeTableProc.__qualname__ = "TableProcessor"
_FakeTextProc.__module__ = "marker.processors.text"
_FakeTextProc.__qualname__ = "TextProcessor"

_TABLE_PROC_NAME = f"{_FakeTableProc.__module__}.{_FakeTableProc.__qualname__}"


def _build_fake_modules():
    """Create fake marker + torch modules for sys.modules patching."""
    marker_mod = ModuleType("marker")
    marker_converters_mod = ModuleType("marker.converters")
    marker_converters_pdf_mod = ModuleType("marker.converters.pdf")
    marker_processors_mod = ModuleType("marker.processors")
    marker_processors_table_mod = ModuleType("marker.processors.table")

    fake_converter_cls = MagicMock()
    fake_converter_cls.default_processors = (_FakeOrderProc, _FakeTableProc, _FakeTextProc)
    marker_converters_pdf_mod.PdfConverter = fake_converter_cls
    marker_processors_table_mod.TableProcessor = _FakeTableProc

    # Fake torch — skip CUDA (not available in test env)
    fake_torch = ModuleType("torch")
    fake_torch.cuda = MagicMock()
    fake_torch.cuda.is_available.return_value = False

    return {
        "marker": marker_mod,
        "marker.converters": marker_converters_mod,
        "marker.converters.pdf": marker_converters_pdf_mod,
        "marker.processors": marker_processors_mod,
        "marker.processors.table": marker_processors_table_mod,
        "torch": fake_torch,
    }, fake_converter_cls


# Build once — reused across all tests (numpy can't be reloaded per-process)
_FAKE_MODULES, _FAKE_CONVERTER_CLS = _build_fake_modules()


@pytest.fixture(autouse=True)
def _mock_marker():
    """Patch sys.modules with fake marker + torch for every test."""
    with patch.dict(sys.modules, _FAKE_MODULES):
        yield


class TestBuildConverterArgs:
    """build_converter_args() must filter models and processors by mode."""

    def test_fast_mode_removes_table_rec_model(self):
        """In fast mode, table_rec_model must be deleted from artifact_dict."""
        from servers.marker.converter_helpers import build_converter_args

        config_parser = MagicMock()
        artifact_dict = {
            "layout_model": MagicMock(),
            "recognition_model": MagicMock(),
            "table_rec_model": MagicMock(),
            "detection_model": MagicMock(),
            "ocr_error_model": MagicMock(),
        }

        result_artifact, _, _ = build_converter_args(
            mode="fast",
            config_parser=config_parser,
            artifact_dict=artifact_dict,
        )

        assert "table_rec_model" not in result_artifact
        assert "layout_model" in result_artifact
        assert "recognition_model" in result_artifact

    def test_balanced_mode_keeps_table_rec_model(self):
        """In balanced mode, table_rec_model must remain in artifact_dict."""
        from servers.marker.converter_helpers import build_converter_args

        config_parser = MagicMock()
        artifact_dict = {
            "layout_model": MagicMock(),
            "recognition_model": MagicMock(),
            "table_rec_model": MagicMock(),
            "detection_model": MagicMock(),
            "ocr_error_model": MagicMock(),
        }

        result_artifact, _, _ = build_converter_args(
            mode="balanced",
            config_parser=config_parser,
            artifact_dict=artifact_dict,
        )

        assert "table_rec_model" in result_artifact

    def test_fast_mode_excludes_table_processor(self):
        """In fast mode, TableProcessor must be excluded from processor list."""
        from servers.marker.converter_helpers import build_converter_args

        config_parser = MagicMock()
        artifact_dict = {"layout_model": MagicMock(), "table_rec_model": MagicMock()}

        _, result_processors, _ = build_converter_args(
            mode="fast",
            config_parser=config_parser,
            artifact_dict=artifact_dict,
        )

        assert _TABLE_PROC_NAME not in result_processors

    def test_balanced_mode_includes_all_processors(self):
        """In balanced mode, processor_list passes through from config_parser."""
        from servers.marker.converter_helpers import build_converter_args

        config_parser = MagicMock()
        config_parser.get_processors.return_value = ["a.processor.ClassA"]
        artifact_dict = {"layout_model": MagicMock(), "table_rec_model": MagicMock()}

        _, result_processors, _ = build_converter_args(
            mode="balanced",
            config_parser=config_parser,
            artifact_dict=artifact_dict,
        )

        # Balanced mode returns whatever config_parser.get_processors() returns
        assert result_processors == ["a.processor.ClassA"]
        # table_rec_model is NOT deleted
        assert "table_rec_model" in artifact_dict

    def test_fast_mode_returns_all_other_models(self):
        """Fast mode only removes table_rec_model, not other models."""
        from servers.marker.converter_helpers import build_converter_args

        config_parser = MagicMock()
        artifact_dict = {
            "layout_model": MagicMock(),
            "recognition_model": MagicMock(),
            "table_rec_model": MagicMock(),
            "detection_model": MagicMock(),
            "ocr_error_model": MagicMock(),
        }

        result_artifact, _, _ = build_converter_args(
            mode="fast",
            config_parser=config_parser,
            artifact_dict=artifact_dict,
        )

        assert len(result_artifact) == 4
        assert set(result_artifact.keys()) == {
            "layout_model",
            "recognition_model",
            "detection_model",
            "ocr_error_model",
        }
