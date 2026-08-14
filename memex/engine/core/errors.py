"""Typed exception hierarchy for Memex RAG.

Every failure in the pipeline should raise a subclass of ``MemexError`` so
callers (CLI, MCP server, sync engine) can handle specific failure classes
without string-matching. Each error carries:

- ``hint``: a short, actionable remediation string for humans/logs.
- ``component``: which pipeline stage/service failed (pipeline.py stages).

Do not raise bare ``RuntimeError`` for domain failures — raise these.
"""

from __future__ import annotations

from typing import Any


class MemexError(RuntimeError):
    """Base class for all Memex errors.

    Extends ``RuntimeError`` so existing ``except RuntimeError`` handlers
    keep working while callers get a typed, catchable hierarchy.
    """

    component: str = "memex"

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        component: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.hint = hint
        if component is not None:
            self.component = component
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        msg = super().__str__()
        if self.hint:
            return f"{msg} ({self.hint})"
        return msg


# ── Configuration & startup ──────────────────────────────────────────────────


class ConfigError(MemexError):
    """Invalid or missing configuration."""

    component = "config"


class ServiceUnavailableError(MemexError):
    """A backing service (Qdrant, Ollama, Docling, Redis, ML) is unreachable."""

    def __init__(
        self,
        service: str,
        message: str,
        *,
        hint: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.service = service
        super().__init__(
            f"{service} unavailable: {message}",
            hint=hint or f"Check that {service} is running (docker compose ps)",
            component="service",
            cause=cause,
        )


# ── Document conversion & chunking ───────────────────────────────────────────


class ConversionError(MemexError):
    """Document parse/convert failure (Docling)."""

    component = "conversion"

    def __init__(
        self,
        source: str,
        detail: str,
        *,
        hint: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.source = source
        super().__init__(f"conversion failed for {source}: {detail}", hint=hint, cause=cause)


class ConversionTimeoutError(ConversionError):
    """A document exceeded the conversion timeout (Docling 504 / max wait)."""

    component = "conversion"

    def __init__(
        self,
        source: str,
        *,
        timeout_s: float | int | None = None,
        hint: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.timeout_s = timeout_s
        timeout_str = f" after {timeout_s}s" if timeout_s else ""
        super().__init__(
            source,
            f"timed out{timeout_str}",
            hint=hint
            or "The document is too large/slow for the conversion window. "
            "Reduce concurrent conversions (converter.docling_max_concurrent), disable OCR "
            "(converter.docling_ocr=false) for digital files, or increase the server window "
            "(DOCLING_SERVE_MAX_SYNC_WAIT).",
            cause=cause,
        )


class ChunkingError(MemexError):
    """Chunking failure (Docling HybridChunker or local splitter)."""

    component = "chunking"


# ── Embedding & storage ──────────────────────────────────────────────────────


class EmbeddingError(MemexError):
    """Embedding generation failure."""

    component = "embedding"


class StorageError(MemexError):
    """Vector store (Qdrant) read/write failure."""

    component = "storage"


# ── Retrieval & generation ───────────────────────────────────────────────────


class RetrievalError(MemexError):
    """Search/retrieval failure."""

    component = "retrieval"


class AnswerError(MemexError):
    """Answer generation failure."""

    component = "answer"


# ── Ingestion & sync ─────────────────────────────────────────────────────────


class IngestionError(MemexError):
    """A single-file ingestion failure (conversion → embed → store)."""

    component = "ingestion"

    def __init__(
        self,
        source: str,
        detail: str,
        *,
        stage: str | None = None,
        hint: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.source = source
        self.stage = stage
        prefix = f" ({stage})" if stage else ""
        super().__init__(f"ingestion failed for {source}{prefix}: {detail}", hint=hint, cause=cause)


class SourceError(MemexError):
    """Source (local dir / S3 bucket) listing or download failure."""

    component = "source"


class SyncError(MemexError):
    """Collection-wide sync failure."""

    component = "sync"


# ── Data integrity ───────────────────────────────────────────────────────────


class CorruptedDocumentError(MemexError):
    """A document converted but produced unusable output (empty/truncated)."""

    component = "conversion"


class DuplicateDocumentError(MemexError):
    """A document with identical content hash already exists."""

    component = "ingestion"


# ── Error context helpers ────────────────────────────────────────────────────


def error_context(exc: BaseException) -> dict[str, Any]:
    """Extract structured context from an exception for logging/observability."""
    ctx: dict[str, Any] = {
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    if isinstance(exc, MemexError):
        ctx["component"] = exc.component
        if exc.hint:
            ctx["hint"] = exc.hint
        for attr in ("source", "service", "stage", "timeout_s"):
            if hasattr(exc, attr) and getattr(exc, attr) is not None:
                ctx[attr] = getattr(exc, attr)
    return ctx
