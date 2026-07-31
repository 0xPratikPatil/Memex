"""MarkItDown converter — HTTP client for MarkItDown Docker service."""

from __future__ import annotations

import pathlib

import httpx

from rag.converters import Converter


class MarkItDownConverter(Converter):
    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")

    async def convert(self, file_path: str) -> str:
        """Upload file to MarkItDown service and return markdown."""
        path = pathlib.Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(path, "rb") as f:
                resp = await client.post(
                    f"{self._base_url}/convert",
                    files={"file": (path.name, f, "application/octet-stream")},
                )
            resp.raise_for_status()
            return resp.json().get("markdown", "")

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/health")
                return resp.status_code == 200
        except Exception:
            return False
