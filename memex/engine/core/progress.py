"""Progress tracking for CLI operations."""

from __future__ import annotations

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
