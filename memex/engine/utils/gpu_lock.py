"""GpuLock — mutual exclusion between GPU consumers (Marker vs Ollama).

Marker and Ollama share the GPU. On small cards their combined VRAM footprint
exceeds capacity, which manifests as Ollama stalls (ReadTimeouts) and marker
OOM kills. This lock enforces mutual exclusion when VRAM is tight:

    acquire(owner):
      - if VRAM used < threshold → no-op (both services coexist, e.g. big GPU)
      - else → unload Ollama models (keep_alive=0) and wait for VRAM to free,
               bounded by gpu.max_wait_s; on timeout/failure log + proceed

    release(owner):
      - no-op — Ollama reloads models on demand (container keep_alive=24h)

Best-effort by design: nvidia-smi or Ollama API failures log a warning and
proceed. Coordination must never block ingestion.
"""

from __future__ import annotations

import logging
import subprocess
import time

import httpx

from memex.engine.core import config

logger = logging.getLogger("gpu-lock")


def _vram_used_mb() -> int | None:
    """Return used GPU memory in MB via nvidia-smi, or None on failure."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode != 0:
            return None
        val = out.stdout.strip().splitlines()
        if not val:
            return None
        return int(val[0].strip())
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def _unload_ollama_models() -> None:
    """Evict loaded Ollama models so their VRAM is released.

    Uses the official API: a generate call with keep_alive=0 unloads the model.
    """
    try:
        resp = httpx.get(f"{config.OLLAMA_EMBED_URL.split('/api')[0]}/api/ps", timeout=5.0)
        if resp.status_code != 200:
            return
        models = resp.json().get("models", [])
        base = config.OLLAMA_EMBED_URL.split("/api")[0]
        for m in models:
            name = m.get("name", "")
            if not name:
                continue
            with httpx.Client(timeout=5.0) as client:
                client.post(
                    f"{base}/api/generate",
                    json={"model": name, "keep_alive": 0, "prompt": ""},
                )
        if models:
            logger.info("Unloaded %d Ollama model(s) from GPU", len(models))
    except Exception as exc:  # best-effort
        logger.debug("Ollama model unload failed: %s", exc)


class GpuLock:
    """Coordination lock for GPU-shared services."""

    def __init__(self) -> None:
        self._owner: str | None = None

    def acquire(self, owner: str) -> None:
        """Acquire the GPU for *owner* if VRAM is tight.

        No-op when gpu.enabled=false or VRAM is below the threshold.
        """
        if not config.GPU_ENABLED:
            return
        if self._owner is not None and self._owner != owner:
            logger.debug("GpuLock already held by %s — proceeding concurrently", self._owner)
        self._owner = owner

        used = _vram_used_mb()
        if used is None:
            logger.debug("GpuLock: nvidia-smi unavailable — proceeding without exclusion")
            return
        if used < config.GPU_VRAM_THRESHOLD_MB:
            logger.debug(
                "GpuLock: VRAM %dMB < threshold %dMB — no exclusion needed",
                used,
                config.GPU_VRAM_THRESHOLD_MB,
            )
            return

        logger.info("GpuLock(%s): VRAM %dMB over threshold — unloading Ollama", owner, used)
        _unload_ollama_models()

        # Wait for VRAM to free (bounded).
        deadline = time.monotonic() + config.GPU_MAX_WAIT_S
        while time.monotonic() < deadline:
            used = _vram_used_mb()
            if used is not None and used < config.GPU_VRAM_THRESHOLD_MB:
                logger.info("GpuLock(%s): VRAM freed (%dMB) — proceeding", owner, used)
                return
            time.sleep(2.0)

        logger.warning(
            "GpuLock(%s): VRAM still over threshold after %.0fs — proceeding anyway",
            owner,
            config.GPU_MAX_WAIT_S,
        )

    def release(self, owner: str) -> None:
        """Release the GPU. No-op — Ollama reloads on demand."""
        if self._owner == owner:
            self._owner = None


# Module-level singleton shared by marker client, pipeline, and embedding.
gpu_lock = GpuLock()


__all__ = ["GpuLock", "gpu_lock"]
