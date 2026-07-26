#!/usr/bin/env python3
"""Lightweight file server for MCP Docker access.

Runs on the host and serves files via HTTP so the MCP server in Docker
can fetch them without volume mounts or base64 encoding.

Usage:
    python file_server.py                    # Default: port 9900, serves /mnt and /home
    python file_server.py --port 9900        # Custom port
    python file_server.py --roots /mnt /data # Custom root directories

Security:
    - Only serves files under specified root directories
    - Read-only access
    - Binds to 0.0.0.0 (adjust for production)
"""

import argparse
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote


class FileHandler(BaseHTTPRequestHandler):
    """Serve files from allowed root directories."""

    allowed_roots: list[Path] = []

    def do_GET(self):
        path = unquote(self.path).lstrip("/")

        if not path:
            self.send_json(
                200,
                {
                    "status": "ok",
                    "message": "File server running",
                    "roots": [str(r) for r in self.allowed_roots],
                },
            )
            return

        if path == "health":
            self.send_json(200, {"status": "ok"})
            return

        # Find file in allowed roots
        file_path = self._resolve_path(path)
        if file_path is None:
            self.send_json(404, {"error": f"File not found: {path}"})
            return

        if not file_path.is_file():
            self.send_json(400, {"error": f"Not a file: {path}"})
            return

        # Serve file
        try:
            data = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("X-File-Path", str(file_path))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_json(500, {"error": str(e)})

    def do_HEAD(self):
        path = unquote(self.path).lstrip("/")

        file_path = self._resolve_path(path)
        if file_path is None or not file_path.is_file():
            self.send_response(404)
            self.end_headers()
            return

        try:
            stat = file_path.stat()
            self.send_response(200)
            self.send_header("Content-Length", str(stat.st_size))
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("X-File-Path", str(file_path))
            self.end_headers()
        except Exception:
            self.send_response(500)
            self.end_headers()

    def _resolve_path(self, path: str) -> Path | None:
        """Resolve a path against allowed roots."""
        # Normalize path - remove leading slashes and decode
        path = path.lstrip("/")

        # Try direct path first (relative to root)
        for root in self.allowed_roots:
            candidate = root / path
            try:
                resolved = candidate.resolve()
                root_resolved = root.resolve()
                if resolved.is_relative_to(root_resolved) and resolved.exists():
                    return candidate
            except (ValueError, OSError):
                continue

        # Try absolute path (if path starts with /)
        if path.startswith("/"):
            abs_path = Path(path)
            for root in self.allowed_roots:
                try:
                    resolved = abs_path.resolve()
                    root_resolved = root.resolve()
                    if resolved.is_relative_to(root_resolved) and resolved.exists():
                        return abs_path
                except (ValueError, OSError):
                    continue

        # Try without root prefix (if path includes root like "mnt/...")
        for root in self.allowed_roots:
            # Check if path starts with root name
            root_name = root.name
            if path.startswith(root_name + "/"):
                relative_path = path[len(root_name) + 1 :]
                candidate = root / relative_path
                try:
                    resolved = candidate.resolve()
                    root_resolved = root.resolve()
                    if resolved.is_relative_to(root_resolved) and resolved.exists():
                        return candidate
                except (ValueError, OSError):
                    continue

        return None

    def send_json(self, code: int, data: dict):
        import json

        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Quieter logging
        if "/health" not in str(args):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="File server for MCP Docker access")
    parser.add_argument("--port", type=int, default=9900, help="Port to serve on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["/mnt", "/home"],
        help="Root directories to serve (default: /mnt /home)",
    )
    args = parser.parse_args()

    # Validate roots
    roots = []
    for root_str in args.roots:
        root = Path(root_str)
        if not root.exists():
            print(f"Warning: Root directory does not exist: {root}")
            continue
        if not root.is_dir():
            print(f"Warning: Root is not a directory: {root}")
            continue
        roots.append(root.resolve())

    if not roots:
        print("Error: No valid root directories")
        sys.exit(1)

    FileHandler.allowed_roots = roots

    server = HTTPServer((args.host, args.port), FileHandler)
    print(f"File server running on http://{args.host}:{args.port}")
    print(f"Serving roots: {[str(r) for r in roots]}")
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
