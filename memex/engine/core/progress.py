"""Progress tracking for CLI operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class FileProgress:
    """Progress state for a single file operation."""

    path: str
    total: int
    current: int
    stage: str
    chunks: int = 0
    error: str = ""


ProgressCallback = Callable[[FileProgress], None]
