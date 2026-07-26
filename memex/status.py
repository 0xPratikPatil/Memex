"""Service status checker for Memex MCP server."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger("memex.status")


@dataclass
class ServiceStatus:
    """Status of a backend service."""

    name: str
    url: str
    healthy: bool
    error: str | None = None
    latency_ms: float | None = None


class ServiceChecker:
    """Check status of backend services."""

    def __init__(self) -> None:
        self.services: dict[str, str] = {}

    def register_service(self, name: str, url: str) -> None:
        """Register a service to check."""
        self.services[name] = url

    async def check_service(self, name: str, url: str) -> ServiceStatus:
        """Check if a service is healthy."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Extract base URL (scheme + host + port only)
                from urllib.parse import urlparse

                parsed = urlparse(url)
                base = f"{parsed.scheme}://{parsed.netloc}"

                if name == "ollama":
                    health_url = f"{base}/api/tags"
                elif name == "qdrant":
                    health_url = f"{base}/"
                elif name == "redis":
                    try:
                        import redis

                        r = redis.Redis.from_url(url)
                        r.ping()
                        return ServiceStatus(
                            name=name,
                            url=url,
                            healthy=True,
                            latency_ms=0.0,
                        )
                    except Exception as e:
                        return ServiceStatus(
                            name=name,
                            url=url,
                            healthy=False,
                            error=str(e),
                        )
                else:
                    health_url = f"{base}/health"

                response = await client.get(health_url)
                latency_ms = response.elapsed.total_seconds() * 1000

                return ServiceStatus(
                    name=name,
                    url=url,
                    healthy=response.status_code == 200,
                    latency_ms=latency_ms,
                )
        except httpx.RequestError as e:
            return ServiceStatus(
                name=name,
                url=url,
                healthy=False,
                error=str(e),
            )

    async def check_all(self) -> dict[str, ServiceStatus]:
        """Check all registered services."""
        results = {}
        for name, url in self.services.items():
            results[name] = await self.check_service(name, url)
        return results

    def get_status_summary(self, statuses: dict[str, ServiceStatus]) -> str:
        """Get a human-readable status summary."""
        lines = ["Service Status:"]
        for name, status in statuses.items():
            if status.healthy:
                latency = f" ({status.latency_ms:.0f}ms)" if status.latency_ms else ""
                lines.append(f"  ✓ {name}: healthy{latency}")
            else:
                error = f" - {status.error}" if status.error else ""
                lines.append(f"  ✗ {name}: unhealthy{error}")
        return "\n".join(lines)


def create_service_checker() -> ServiceChecker:
    """Create a service checker with default services."""
    from rag import config

    checker = ServiceChecker()
    checker.register_service("qdrant", config.QDRANT_URL)
    checker.register_service("ollama", config.OLLAMA_EMBED_URL)
    checker.register_service("docling", config.DOCLING_URL)
    checker.register_service("ml-services", config.ML_SERVICES_URL)
    checker.register_service("redis", config.REDIS_URL)
    return checker
