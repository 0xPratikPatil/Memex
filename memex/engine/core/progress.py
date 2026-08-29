"""Progress tracking for CLI operations."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum


class PipelineStage(StrEnum):
    """Typed pipeline stages surfaced through progress callbacks.

    Kept as ``str`` values so existing callers comparing against plain
    strings (sync.py, cli.py) keep working.
    """

    SCANNING = "Scanning"
    RECONCILING = "Reconciling"
    HASHING = "Hashing"
    PARSING = "Parsing"
    CONVERTING = "Converting"
    QUEUED = "Queued"
    WAITING_GPU = "Waiting GPU"
    OCR = "OCR"
    CHUNKING = "Chunking"
    CONTEXT = "Context"
    METADATA = "Metadata"
    EMBEDDING = "Embedding"
    STORING = "Storing"
    DELETING = "Deleting"
    DONE = "Done"
    SKIPPED = "Skipped"
    ERROR = "Error"


@dataclass
class FileProgress:
    """Progress state for a single file operation."""

    path: str
    total: int
    current: int
    stage: str
    chunks: int = 0
    error: str = ""
    started_at: float | None = field(default=None, repr=False)


ProgressCallback = Callable[[FileProgress], None]


def stage_is_terminal(stage: str) -> bool:
    """Return True for stages that end a file's lifecycle."""
    return stage in (PipelineStage.DONE, PipelineStage.SKIPPED, PipelineStage.ERROR)


def stage_is_error(stage: str) -> bool:
    """Return True for error terminal stages."""
    return stage == PipelineStage.ERROR


class ActivityRegistry:
    """Thread-safe registry of live LLM-phase activity, for the TUI.

    The pipeline reports ``(source, phase)`` on every phase transition and
    removes the source on terminal events. The CLI's LLM activity row polls
    ``snapshot()`` to render which files are in context / metadata /
    embedding right now.

    Entries are pruned when they have not been touched for ``STALE_S`` so a
    crashed pipeline never leaves ghost activity.
    """

    STALE_S = 60.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[str, float]] = {}

    def report(self, source: str, phase: str) -> None:
        with self._lock:
            self._entries[source] = (phase, time.monotonic())

    def remove(self, source: str) -> None:
        with self._lock:
            self._entries.pop(source, None)

    def snapshot(self) -> dict[str, str]:
        """Return {source: phase} for non-stale entries, pruning stale ones."""
        now = time.monotonic()
        with self._lock:
            stale = [
                src for src, (_p, ts) in self._entries.items() if now - ts > self.STALE_S
            ]
            for src in stale:
                del self._entries[src]
            return {src: phase for src, (phase, _ts) in self._entries.items()}


activity_registry = ActivityRegistry()
