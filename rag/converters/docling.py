"""Docling converter — wraps existing Docling client."""

from __future__ import annotations

from rag.converters import Converter


class DoclingConverter(Converter):
    def __init__(self, base_url: str):
        self._base_url = base_url

    async def convert(self, file_path: str) -> str:
        from rag.docling_client import parse_file

        result = parse_file(file_path)
        if not result.ok:
            raise RuntimeError(f"Docling conversion failed: {result.errors}")
        return result.markdown

    async def health_check(self) -> bool:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/health")
                return resp.status_code == 200
        except Exception:
            return False
