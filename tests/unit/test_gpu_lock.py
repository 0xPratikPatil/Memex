"""Unit tests for GpuLock — GPU mutual exclusion between Marker and Ollama."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from memex.engine.core import config
from memex.engine.utils.gpu_lock import GpuLock


@pytest.fixture(autouse=True)
def _reset_lock() -> None:
    yield


class TestAcquire:
    def test_disabled_when_gpu_enabled_false(self) -> None:
        lock = GpuLock()
        with (
            patch.object(config, "GPU_ENABLED", False),
            patch("memex.engine.utils.gpu_lock._vram_used_mb", side_effect=AssertionError("must not be called")),
        ):
            lock.acquire("marker")  # must not touch nvidia-smi

    def test_noop_below_threshold(self) -> None:
        lock = GpuLock()
        with (
            patch.object(config, "GPU_ENABLED", True),
            patch.object(config, "GPU_VRAM_THRESHOLD_MB", 6000),
            patch("memex.engine.utils.gpu_lock._vram_used_mb", return_value=1000),
            patch("memex.engine.utils.gpu_lock._unload_ollama_models", side_effect=AssertionError("must not unload")),
        ):
            lock.acquire("marker")  # VRAM below threshold → no unload

    def test_unloads_ollama_over_threshold(self) -> None:
        lock = GpuLock()
        with (
            patch.object(config, "GPU_ENABLED", True),
            patch.object(config, "GPU_VRAM_THRESHOLD_MB", 6000),
            patch.object(config, "GPU_MAX_WAIT_S", 10),
            patch(
                "memex.engine.utils.gpu_lock._vram_used_mb",
                side_effect=[7000, 7000, 5000],  # over, over, then freed
            ),
            patch("memex.engine.utils.gpu_lock._unload_ollama_models") as mock_unload,
        ):
            lock.acquire("marker")
            mock_unload.assert_called_once()

    def test_proceeds_when_nvidia_smi_fails(self) -> None:
        lock = GpuLock()
        with (
            patch.object(config, "GPU_ENABLED", True),
            patch("memex.engine.utils.gpu_lock._vram_used_mb", return_value=None),
            patch("memex.engine.utils.gpu_lock._unload_ollama_models") as mock_unload,
        ):
            lock.acquire("marker")  # must not raise
            mock_unload.assert_not_called()

    def test_warns_and_proceeds_after_max_wait(self) -> None:
        lock = GpuLock()
        with (
            patch.object(config, "GPU_ENABLED", True),
            patch.object(config, "GPU_VRAM_THRESHOLD_MB", 6000),
            patch.object(config, "GPU_MAX_WAIT_S", 0),  # immediate timeout
            patch("memex.engine.utils.gpu_lock._vram_used_mb", return_value=7000),
            patch("memex.engine.utils.gpu_lock._unload_ollama_models"),
        ):
            lock.acquire("marker")  # must not raise, warns + proceeds


class TestRelease:
    def test_release_noop(self) -> None:
        lock = GpuLock()
        lock.acquire("marker")
        lock.release("marker")
        assert lock._owner is None


class TestVramUsed:
    def test_parses_nvidia_smi_output(self) -> None:

        from memex.engine.utils import gpu_lock as gl

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "4321\n"
        with patch("memex.engine.utils.gpu_lock.subprocess.run", return_value=mock_proc):
            assert gl._vram_used_mb() == 4321

    def test_returns_none_on_error(self) -> None:
        import subprocess

        from memex.engine.utils import gpu_lock as gl

        with patch(
            "memex.engine.utils.gpu_lock.subprocess.run",
            side_effect=subprocess.SubprocessError("no gpu"),
        ):
            assert gl._vram_used_mb() is None


class TestUnloadOllama:
    def test_unloads_loaded_models(self) -> None:
        from memex.engine.utils import gpu_lock as gl

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [{"name": "qwen2.5:1.5b"}, {"name": "bge-m3:latest"}]
        }

        with (
            patch.object(config, "OLLAMA_EMBED_URL", "http://localhost:11434/api/embed"),
            patch("memex.engine.utils.gpu_lock.httpx.get", return_value=mock_resp),
            patch("memex.engine.utils.gpu_lock.httpx.Client") as mock_client_cls,
        ):
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            gl._unload_ollama_models()
            assert mock_client.post.call_count == 2
            for call in mock_client.post.call_args_list:
                payload = call[1]["json"]
                assert payload["keep_alive"] == 0

    def test_noop_when_api_unavailable(self) -> None:
        from memex.engine.utils import gpu_lock as gl

        with (
            patch.object(config, "OLLAMA_EMBED_URL", "http://localhost:11434/api/embed"),
            patch("memex.engine.utils.gpu_lock.httpx.get", side_effect=Exception("down")),
        ):
            gl._unload_ollama_models()  # must not raise
