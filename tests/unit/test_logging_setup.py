"""Unit tests for memex.engine.core.logging_setup — colored structured logging."""

from __future__ import annotations

import logging

import pytest

from memex.engine.core.logging_setup import get_logger, set_level, setup_logging


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMEX_LOG_JSON", raising=False)
    monkeypatch.delenv("MEMEX_LOG_PLAIN", raising=False)


class TestSetupLogging:
    def test_installs_root_handler(self) -> None:
        setup_logging()
        root = logging.getLogger()
        assert any(h is not None for h in root.handlers)

    def test_verbose_sets_debug(self) -> None:
        setup_logging(verbose=True)
        assert logging.getLogger().level <= logging.DEBUG

    def test_default_sets_info(self) -> None:
        setup_logging(verbose=False)
        assert logging.getLogger().level == logging.INFO

    def test_no_duplicate_handlers(self) -> None:
        setup_logging()
        setup_logging()
        setup_logging()
        assert len(logging.getLogger().handlers) == 1

    def test_json_mode_installs_stream_handler(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMEX_LOG_JSON", "1")
        setup_logging()
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert root.handlers[0].formatter.__class__.__name__ == "_JsonFormatter"

    def test_quiet_third_party_loggers(self) -> None:
        setup_logging()
        assert logging.getLogger("httpx").level >= logging.WARNING

    def test_plain_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEMEX_LOG_PLAIN", "1")
        setup_logging()
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert "Formatter" in type(root.handlers[0].formatter).__name__


class TestGetLogger:
    def test_returns_logger_with_name(self) -> None:
        logger = get_logger("memex.test")
        assert logger.name == "memex.test"
        assert isinstance(logger, logging.Logger)


class TestSetLevel:
    def test_changes_root_level(self) -> None:
        set_level(logging.DEBUG)
        assert logging.getLogger().level == logging.DEBUG
        set_level(logging.INFO)
        assert logging.getLogger().level == logging.INFO
