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


class _LazyRegistry:
    """Lazy registration registry for pluggable document sources.

    Supports ``register()`` for third-party source types.
    Acts as a dict-like container for source type lookup.
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[Source]] = {}

    def register(self, cls: type[Source]) -> type[Source]:
        self._registry[cls.type] = cls
        return cls

    def get(self, type_name: str) -> type[Source] | None:
        return self._registry.get(type_name)

    def keys(self) -> list[str]:
        return list(self._registry.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __len__(self) -> int:
        return len(self._registry)


_SOURCES = _LazyRegistry()


def register_source(cls: type[Source]) -> type[Source]:
    """Register a source type by its ``type`` class attribute."""
    return _SOURCES.register(cls)


def get_source(type_name: str, config: dict[str, Any]) -> Source:
    """Instantiate and return a source by type name.

    Raises:
        ValueError: If ``type_name`` is not registered.
    """
    cls = _SOURCES.get(type_name)
    if cls is None:
        raise ValueError(f"Unknown source type: {type_name}")
    kwargs = {k: v for k, v in config.items() if k != "type"}
    return cls(**kwargs)


def list_source_types() -> list[str]:
    """Return the names of all registered source types."""
    return _SOURCES.keys()


# Ensure source implementations are imported so @register_source runs
from memex.engine.sources import (  # noqa: E402
    local,  # noqa: F401
    s3,  # noqa: F401
)
