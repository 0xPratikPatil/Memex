"""DoclingAsyncClient — async Docling Serve client with status polling.

Uses Docling's async API endpoints for non-blocking document conversion.
Submits jobs via ``/v1/convert/source/async``, polls status, fetches results.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from memex.engine.core import config

logger = logging.getLogger("docling-async-client")


class DoclingAsyncClient:
    """Async Docling Serve client with status polling.

    Args:
        base_url: Docling Serve base URL (e.g., http://localhost:5001).
        api_key: Optional API key for authentication.
    """

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(config.DOCLING_TIMEOUT, connect=10.0))

    def _headers(self) -> dict[str, str]:
        """Build request headers with optional API key."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["X-Api-Key"] = self._api_key
        return headers

    async def health_check(self) -> bool:
        """Check if Docling is reachable and healthy.

        Returns True if /health returns 200, False otherwise.
        """
        try:
            resp = await self._client.get(f"{self._base_url}/health")
            return resp.status_code == 200
        except (httpx.TransportError, httpx.HTTPStatusError):
            return False

    async def ready_check(self) -> bool:
        """Check if Docling models are loaded and ready.

        Returns True if /ready returns 200, False otherwise.
        """
        try:
            resp = await self._client.get(f"{self._base_url}/ready")
            return resp.status_code == 200
        except (httpx.TransportError, httpx.HTTPStatusError):
            return False

    async def submit_conversion(self, payload: dict) -> str:
        """Submit async conversion job, return task_id.

        Args:
            payload: Docling conversion payload with options and sources.

        Returns:
            task_id string for polling.
        """
        resp = await self._client.post(
            f"{self._base_url}/v1/convert/source/async",
            json=payload,
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()["task_id"]

    async def poll_status(self, task_id: str) -> dict:
        """Poll task status from Docling.

        Args:
            task_id: The task ID returned by submit_conversion.

        Returns:
            Dict with task_status (pending|started|success|failure) and task_position.
        """
        resp = await self._client.get(
            f"{self._base_url}/v1/status/poll/{task_id}",
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def get_result(self, task_id: str) -> dict:
        """Fetch completed conversion result.

        Args:
            task_id: The task ID of a completed conversion.

        Returns:
            Full conversion result dict.
        """
        resp = await self._client.get(
            f"{self._base_url}/v1/result/{task_id}",
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def wait_for_completion(
        self,
        task_id: str,
        poll_interval: float = 5.0,
        max_wait: float = 600.0,
    ) -> dict:
        """Poll until task completes or times out.

        Args:
            task_id: The task ID to poll.
            poll_interval: Seconds between polls.
            max_wait: Maximum seconds to wait before raising TimeoutError.

        Returns:
            Final status dict.

        Raises:
            TimeoutError: If task doesn't complete within max_wait.
        """
        start = time.monotonic()
        while True:
            status = await self.poll_status(task_id)
            if status["task_status"] in ("success", "failure"):
                return status
            if time.monotonic() - start > max_wait:
                raise TimeoutError(f"Task {task_id} exceeded {max_wait}s")
            await asyncio.sleep(poll_interval)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
