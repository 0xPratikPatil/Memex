"""Document converter abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Converter(ABC):
    """Base class for document converters."""

    @abstractmethod
    async def convert(self, file_path: str) -> str:
        """Convert a document to markdown. Returns markdown content."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the converter service is available."""
        ...


def get_converter(config_module) -> Converter:
    """Factory: return the configured converter."""
    engine = config_module.get_str("converter.engine", "docling")
    if engine == "markitdown":
        from rag.converters.markitdown import MarkItDownConverter

        return MarkItDownConverter(config_module.get_str("converter.markitdown_url", "http://localhost:5003"))
    else:
        from rag.converters.docling import DoclingConverter

        return DoclingConverter(config_module.get_str("converter.docling_url", "http://localhost:5001"))
