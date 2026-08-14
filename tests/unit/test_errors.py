"""Unit tests for memex.engine.core.errors — typed exception hierarchy."""

from __future__ import annotations

from memex.engine.core.errors import (
    ConfigError,
    ConversionError,
    ConversionTimeoutError,
    IngestionError,
    MemexError,
    ServiceUnavailableError,
    error_context,
)


class TestMemexError:
    def test_is_runtime_error_subclass(self) -> None:
        assert issubclass(MemexError, RuntimeError)

    def test_hint_appended_to_message(self) -> None:
        err = MemexError("boom", hint="fix it")
        assert str(err) == "boom (fix it)"
        assert "boom" in str(err)

    def test_no_hint_no_suffix(self) -> None:
        err = MemexError("boom")
        assert str(err) == "boom"

    def test_component_default_and_override(self) -> None:
        assert MemexError("x").component == "memex"
        assert MemexError("x", component="storage").component == "storage"

    def test_cause_preserved(self) -> None:
        cause = ValueError("inner")
        err = MemexError("boom", cause=cause)
        assert err.__cause__ is cause


class TestTypedErrors:
    def test_conversion_timeout_specialization(self) -> None:
        err = ConversionTimeoutError("/tmp/a.pdf", timeout_s=120)
        assert isinstance(err, ConversionError)
        assert isinstance(err, MemexError)
        assert err.source == "/tmp/a.pdf"
        assert "timed out" in str(err)
        assert err.hint  # actionable remediation present

    def test_service_unavailable(self) -> None:
        err = ServiceUnavailableError("Qdrant", "connection refused")
        assert err.service == "Qdrant"
        assert "Qdrant" in str(err)
        assert "docker compose" in err.hint

    def test_ingestion_error_fields(self) -> None:
        err = IngestionError("/tmp/f.pdf", "parse failed", stage="Converting")
        assert err.source == "/tmp/f.pdf"
        assert err.stage == "Converting"
        assert "(Converting)" in str(err)

    def test_config_error_hint(self) -> None:
        err = ConfigError("bad config", hint="check config.yaml")
        assert err.hint == "check config.yaml"


class TestErrorContext:
    def test_memex_error_extracts_fields(self) -> None:
        err = IngestionError("/tmp/f.pdf", "parse failed", stage="Converting", hint="retry")
        ctx = error_context(err)
        assert ctx["error_type"] == "IngestionError"
        assert ctx["component"] == "ingestion"
        assert ctx["source"] == "/tmp/f.pdf"
        assert ctx["stage"] == "Converting"
        assert ctx["hint"] == "retry"

    def test_plain_exception_has_type_and_message(self) -> None:
        ctx = error_context(ValueError("nope"))
        assert ctx["error_type"] == "ValueError"
        assert ctx["error"] == "nope"
        assert "component" not in ctx
