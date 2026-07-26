#!/usr/bin/env python3
"""Launch the MCP server in stdio (default) or streamable HTTP mode.

Environment variables:
    MCP_HOST   - bind address for HTTP mode (default 0.0.0.0)
    MCP_PORT   - port for HTTP mode         (default 8080)
    MCP_TRANSPORT - "stdio" | "http"        (default stdio)
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure project root is on sys.path so ``import config`` works.
sys.path.insert(0, os.path.dirname(__file__))

import config


def main() -> None:
    parser = argparse.ArgumentParser(description="Personal RAG MCP Server")
    parser.add_argument(
        "--http",
        action="store_true",
        default=os.getenv("MCP_TRANSPORT", "").lower() == "http",
        help="Run in streamable HTTP mode (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default=config.MCP_HOST,
        help=f"HTTP bind host (default: {config.MCP_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.MCP_PORT,
        help=f"HTTP bind port (default: {config.MCP_PORT})",
    )
    args = parser.parse_args()

    # Import here so config is already loaded.
    from src.server import mcp

    if args.http:
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        app = mcp.streamable_http_app()

        async def health(request: Request) -> JSONResponse:
            return JSONResponse({
                "status": "ok",
                "service": "personal-rag-mcp",
                "transport": "streamable-http",
            })

        app.routes.insert(0, Route("/health", endpoint=health))

        mcp.settings.host = args.host
        mcp.settings.port = args.port
        print(f"MCP server (streamable HTTP) listening on http://{args.host}:{args.port}/mcp")
        print(f"Health check available at http://{args.host}:{args.port}/health")

        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
