"""Memex startup banner and service health checks."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field

from memex.engine.core import config
from memex.mcp.status import create_service_checker


@dataclass
class StartupBanner:
    """Startup banner with config overview and service status."""

    config_text: str = ""
    services_text: str = ""
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        parts = [self.config_text]
        if self.services_text:
            parts.append(self.services_text)
        if self.warnings:
            parts.append("WARNING: " + " ".join(self.warnings))
        return "\n".join(parts)


def check_services() -> dict[str, object]:
    """Check all backend services and return their status."""
    checker = create_service_checker()
    result = checker.check_all()
    if inspect.iscoroutine(result):
        return asyncio.run(result)
    return result


def build_startup_banner() -> str:
    """Build the full startup banner with config and service status."""
    from memex import __version__

    lines: list[str] = []
    lines.append("─" * 52)
    lines.append(f"  Memex v{__version__}  ·  Personal RAG MCP Server")
    lines.append("─" * 52)
    lines.append("")
    lines.append("  Configuration:")
    lines.append(f"    embed model : {config.EMBED_MODEL}")
    lines.append(f"    chat model  : {config.CHAT_MODEL}")
    lines.append(f"    chunking    : {config.CHUNK_STRATEGY} @ {config.CHUNK_SIZE} tokens")

    from memex.engine.ingestion.splitter import is_hybrid_chunker_available as _hybrid_ok

    try:
        _ok = _hybrid_ok()
    except Exception:
        _ok = False
    chunker_name = "Docling HybridChunker" if _ok else "legacy recursive"
    lines.append(f"    chunker     : {chunker_name}")

    cache_state = "enabled" if config.ENABLE_CACHE else "disabled"
    lines.append(f"    cache       : {cache_state}")

    if config.ENABLE_QUERY_EXPANSION:
        techniques = []
        if config.ENABLE_HYDE:
            techniques.append("HyDE")
        if config.ENABLE_MULTI_QUERY:
            techniques.append("Multi-Query")
        if config.ENABLE_QUERY_REWRITE:
            techniques.append("Rewrite")
        expansion_str = ", ".join(techniques).lower() if techniques else "on"
        lines.append(f"    expansion   : {expansion_str} (enabled)")
    else:
        lines.append("    expansion   : off")

    lines.append("")

    # Service health
    statuses = check_services()
    lines.append("  Services:")

    unhealthy: list[str] = []
    for name in ("qdrant", "ollama", "docling"):
        s = statuses.get(name)
        if s is None:
            unhealthy.append(name)
            lines.append(f"    {name:12s} missing")
        elif s.healthy:
            lat = f" {s.latency_ms:.1f}ms" if s.latency_ms is not None else ""
            lines.append(f"    {name:12s} healthy{lat}")
        else:
            err = f" — {s.error}" if s.error else ""
            unhealthy.append(name)
            lines.append(f"    {name:12s} unhealthy{err}")

    lines.append("─" * 52)

    return "\n".join(lines)
