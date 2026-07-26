"""Memex CLI - Command line interface for the MCP server."""

from __future__ import annotations


def main() -> None:
    """Main entry point for the memex CLI."""
    from memex.server import mcp

    mcp.run()


if __name__ == "__main__":
    main()
