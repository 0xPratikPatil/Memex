"""Colored, structured logging for Memex.

Production-grade logging setup built on ``rich.logging.RichHandler``:

- Human-friendly colored logs on the terminal (level colors, timestamps).
- Structured ``extra`` attributes rendered as ``key=value`` on every record
  so log lines are greppable and parseable.
- Optional JSON mode (``MEMEX_LOG_JSON=1``) for machine-readable output
  (Logstash / OpenTelemetry / Datadog ingestion).
- Optional plain-text mode (``MEMEX_LOG_PLAIN=1``) when colors are not
  available (CI, piped output, non-TTY).

Entry points:
    setup_logging(verbose=False)   # configure once at process start
    get_logger(name)               # configured child logger
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Any

from rich.logging import RichHandler

# Format of extra attrs appended to a line, e.g. " source=/tmp/a.pdf stage=chunking"
_EXTRA_FORMAT = " %(source)s %(stage)s %(service)s %(hint)s"

_LEVEL_COLORS = {
    "DEBUG": "dim cyan",
    "INFO": "cyan",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold red",
}


def _resolved_extra(record: logging.LogRecord) -> str:
    """Render extra attributes as space-separated key=value pairs."""
    parts: list[str] = []
    for key in ("source", "stage", "service", "hint"):
        val = getattr(record, key, None)
        if val is not None and val != "":
            parts.append(f"{key}={val}")
    return " ".join(parts)


class _HumanFormatter(logging.Formatter):
    """Text formatter that appends structured extras as key=value pairs."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        extras = _resolved_extra(record)
        return f"{message}  {extras}" if extras else message


class _JsonFormatter(logging.Formatter):
    """JSON-lines formatter: one self-describing object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("source", "stage", "service", "hint"):
            val = getattr(record, key, None)
            if val is not None and val != "":
                payload[key] = val
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _build_handler(verbose: bool, *, use_json: bool, use_plain: bool) -> logging.Handler:
    level = logging.DEBUG if verbose else logging.INFO

    if use_json:
        handler: logging.Handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
    elif use_plain:
        handler = logging.StreamHandler(sys.stderr)
        fmt = "%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s" + _EXTRA_FORMAT
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    else:
        handler = RichHandler(
            level=level,
            console=__import__("rich.console", fromlist=["Console"]).Console(stderr=True),
            show_time=True,
            show_level=True,
            show_path=False,
            markup=False,
            rich_tracebacks=True,
            tracebacks_show_locals=verbose,
        )
        handler.setFormatter(_HumanFormatter())

    handler.setLevel(level)
    return handler


def setup_logging(verbose: bool = False, level: int | None = None) -> None:
    """Configure the root logger once.

    Args:
        verbose: Enable DEBUG level. When False, INFO and above.
        level: Explicit level override (e.g. logging.WARNING during live
            progress displays so log lines don't corrupt terminal redraws).
    """
    use_json = os.environ.get("MEMEX_LOG_JSON", "0") == "1"
    use_plain = os.environ.get("MEMEX_LOG_PLAIN", "0") == "1"

    root = logging.getLogger()
    # Avoid duplicate handlers when called more than once (server + import).
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = _build_handler(verbose, use_json=use_json, use_plain=use_plain)
    root.addHandler(handler)
    if level is not None:
        root.setLevel(level)
        handler.setLevel(level)
    else:
        root.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Quiet noisy third-party loggers at INFO.
    for noisy in ("httpx", "httpcore", "urllib3", "qdrant_client", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger wired to the Memex handler."""
    return logging.getLogger(name)


def set_level(level: int | str) -> None:
    """Dynamically change the root log level (e.g. for --verbose)."""
    logging.getLogger().setLevel(level)


__all__ = ["get_logger", "set_level", "setup_logging"]
