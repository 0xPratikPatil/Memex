from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SourceFile:
    """Represents a file discovered from a source."""

    name: str
    path: str
    size: int
    modified_at: float


class Source(ABC):
    """Abstract base class for document sources."""

    name: str
    type: str
    extensions: list[str]

    @abstractmethod
    def list_files(self) -> list[SourceFile]: ...

    @abstractmethod
    def get_content_hash(self, file: SourceFile) -> str: ...

    @abstractmethod
    def download(self, file: SourceFile, dest: Path) -> Path: ...


_SOURCES: dict[str, type[Source]] = {}


def register_source(cls: type[Source]) -> type[Source]:
    """Register a source type by its ``type`` class attribute."""
    _SOURCES[cls.type] = cls
    return cls


def get_source(type_name: str, config: dict[str, Any]) -> Source:
    """Instantiate and return a source by type name.

    Raises:
        ValueError: If ``type_name`` is not registered.
    """
    if type_name not in _SOURCES:
        raise ValueError(f"Unknown source type: {type_name}")
    return _SOURCES[type_name](**config)


def list_source_types() -> list[str]:
    """Return the names of all registered source types."""
    return list(_SOURCES.keys())
