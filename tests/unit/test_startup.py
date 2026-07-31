"""Unit tests for startup module — production startup banner and health checks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from memex.mcp.startup import StartupBanner, build_startup_banner, check_services
from memex.mcp.status import ServiceStatus  # existing class


class TestBuildStartupBanner:
    def test_returns_string_with_version(self):
        banner = build_startup_banner()
        assert isinstance(banner, str)
        assert "Memex" in banner
        assert "v0.5.0" in banner or "0.5.0" in banner

    def test_includes_embed_model(self):
        banner = build_startup_banner()
        assert "embed" in banner.lower()
        assert any(m in banner for m in ("bge-m3", "qwen3-embedding"))

    def test_includes_chat_model(self):
        banner = build_startup_banner()
        assert "chat" in banner.lower()

    def test_includes_chunk_strategy(self):
        banner = build_startup_banner()
        assert "hybrid" in banner.lower()
        assert "1024" in banner
        assert any(name in banner for name in ("Docling HybridChunker", "legacy recursive"))

    def test_includes_cache_status(self):
        with patch("memex.mcp.startup.config.ENABLE_CACHE", True):
            banner = build_startup_banner()
            assert "enabled" in banner.lower() or "cache" in banner.lower()

    def test_includes_expansion_status(self):
        banner = build_startup_banner()
        assert "hyde" in banner.lower() or "expansion" in banner.lower()

    def test_expansion_disabled_shows_off(self):
        with patch.multiple(
            "memex.mcp.startup.config",
            ENABLE_QUERY_EXPANSION=False,
            ENABLE_HYDE=False,
            ENABLE_MULTI_QUERY=False,
            ENABLE_QUERY_REWRITE=False,
        ):
            banner = build_startup_banner()
            assert "expansion" in banner.lower()


class TestCheckServices:
    def test_returns_dict_of_service_statuses(self):
        results = check_services()
        assert isinstance(results, dict)
        assert "qdrant" in results
        assert "ollama" in results
        assert "docling" in results

    def test_uses_service_checker(self):
        mock_checker = MagicMock()
        mock_checker.check_all.return_value = {}
        with patch("memex.mcp.startup.create_service_checker", return_value=mock_checker):
            check_services()
            mock_checker.check_all.assert_called_once()

    def test_formats_healthy_banner_line(self):
        mock_checker = MagicMock()
        mock_checker.check_all.return_value = {
            "qdrant": ServiceStatus(name="qdrant", url="http://localhost:6333", healthy=True, latency_ms=2.5),
        }
        with patch("memex.mcp.startup.create_service_checker", return_value=mock_checker):
            banner = build_startup_banner()
            assert "healthy" in banner
            assert "qdrant" in banner.lower()

    def test_formats_unhealthy_banner_line(self):
        mock_checker = MagicMock()
        mock_checker.check_all.return_value = {
            "qdrant": ServiceStatus(
                name="qdrant",
                url="http://localhost:6333",
                healthy=False,
                error="Connection refused",
            ),
        }
        with patch("memex.mcp.startup.create_service_checker", return_value=mock_checker):
            banner = build_startup_banner()
            lower = banner.lower()
            assert any(w in lower for w in ("unreachable", "unhealthy", "connection refused"))


class TestStartupBannerDataclass:
    def test_default_fields(self):
        sb = StartupBanner()
        assert isinstance(sb.config_text, str)
        assert isinstance(sb.services_text, str)
        assert isinstance(sb.warnings, list)

    def test_str_joins_sections(self):
        sb = StartupBanner(config_text="Config OK", services_text="Services OK", warnings=[])
        output = str(sb)
        assert "Config OK" in output
        assert "Services OK" in output

    def test_str_includes_warnings(self):
        sb = StartupBanner(
            config_text="Config OK",
            services_text="Services OK",
            warnings=["2 services unreachable: qdrant, docling"],
        )
        output = str(sb)
        assert "WARNING" in output
        assert "qdrant" in output
