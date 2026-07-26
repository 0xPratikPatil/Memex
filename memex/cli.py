"""Memex CLI - Command line interface for the MCP server."""

from __future__ import annotations


def main() -> None:
    """Main entry point for the memex CLI."""
    from memex.server import mcp
    from memex.startup import build_startup_banner

    print(build_startup_banner(), flush=True)
    mcp.run()
