"""GpuLock — mutual exclusion between GPU consumers (Marker vs Ollama).

Marker and Ollama share the GPU. On small cards their combined VRAM footprint
exceeds capacity, which manifests as Ollama stalls (ReadTimeouts) and marker
OOM kills. This lock enforces mutual exclusion when VRAM is tight:

    acquire(owner):
      - if VRAM used + owner's footprint < total - safety → no-op (coexist)
      - else → if owner is "marker", evict Ollama models (keep_alive=0) and
               wait for VRAM to free; if owner is an Ollama consumer ("llm"/
               "embed"), wait for marker to release (bounded).
    release(owner):
      - clears the exclusive owner so the other side can proceed.

Best-effort by design: nvidia-smi or Ollama API failures log a warning and
proceed. Coordination must never block ingestion indefinitely.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time

import httpx

from memex.engine.core import config

logger = logging.getLogger("gpu-lock")

# Safety margin: never try to fill the card to 100%.
_SAFETY_MARGIN_MB = 512

# Estimated VRAM footprint per owner (models resident during a job).
# marker: 5 models (layout 1.4G + recognition 1.4G + table 0.2G +
#         detection 0.07G + ocr_error 0.26G) ≈ 3.4GB, plus inference buffers.
# llm/embed: Ollama chat + embed models (qwen2.5 1.6G + bge-m3 0.66G) ≈ 2.3GB.
_OWNER_FOOTPRINT_MB = {
    "marker": 4096,
    "ocr": 4200,  # lightonocr-2-1b fp16 on CUDA
    "ocr-small": 700,  # pp-ocrv6-small / granite-docling-258m
    "llm": 2560,
    "embed": 2560,
}


def _total_vram_mb() -> int | None:
    """Return total GPU memory in MB via nvidia-smi, or None on failure."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
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
        base = config.OLLAMA_EMBED_URL.split("/api")[0]
        resp = httpx.get(f"{base}/api/ps", timeout=5.0)
        if resp.status_code != 200:
            return
        models = resp.json().get("models", [])
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
    """Coordination lock for GPU-shared services.

    Thread-safe: the lock protects the owner tracking so concurrent sync
    workers do not race on acquire/release.
    """

    def __init__(self) -> None:
        self._owner: str | None = None
        self._lock = threading.Lock()

    def acquire(self, owner: str) -> None:
        """Acquire the GPU for *owner* if VRAM is tight.

        No-op when gpu.enabled=false or the requester's footprint fits
        alongside current usage.
        """
        if not config.GPU_ENABLED:
            return

        footprint = _OWNER_FOOTPRINT_MB.get(owner, 2048)
        total = _total_vram_mb()
        if total is None:
            logger.debug("GpuLock: nvidia-smi unavailable — proceeding without exclusion")
            return

        used = _vram_used_mb() or 0
        # Will the requester's models fit alongside current usage?
        if used + footprint + _SAFETY_MARGIN_MB <= total:
            logger.debug(
                "GpuLock(%s): used %dMB + footprint %dMB fits %dMB — no exclusion",
                owner,
                used,
                footprint,
                total,
            )
            return

        with self._lock:
            # Another owner holds the GPU exclusively — wait for release.
            if self._owner is not None and self._owner != owner:
                logger.info(
                    "GpuLock(%s): waiting for %s to release the GPU",
                    owner,
                    self._owner,
                )
                deadline = time.monotonic() + config.GPU_MAX_WAIT_S
                while self._owner is not None and time.monotonic() < deadline:
                    time.sleep(1.0)
                if self._owner is not None:
                    logger.warning(
                        "GpuLock(%s): %s did not release within %.0fs — proceeding anyway",
                        owner,
                        self._owner,
                        config.GPU_MAX_WAIT_S,
                    )
                else:
                    logger.info("GpuLock(%s): GPU released by %s", owner, self._owner)

        # We are the exclusive owner now (or the previous owner released).
        self._owner = owner

        # If we are marker or ocr (VLM), evict Ollama to free VRAM, then wait.
        if owner in ("marker", "ocr"):
            logger.info(
                "GpuLock(%s): used %dMB + footprint %dMB > %dMB — unloading Ollama",
                owner,
                used,
                footprint,
                total,
            )
            _unload_ollama_models()
            deadline = time.monotonic() + config.GPU_MAX_WAIT_S
            while time.monotonic() < deadline:
                freed = _vram_used_mb()
                if freed is not None and freed + footprint + _SAFETY_MARGIN_MB <= total:
                    logger.info("GpuLock(%s): VRAM freed (%dMB) — proceeding", owner, freed)
                    return
                time.sleep(2.0)
            logger.warning(
                "GpuLock(%s): VRAM still tight after %.0fs — proceeding anyway",
                owner,
                config.GPU_MAX_WAIT_S,
            )

    def release(self, owner: str) -> None:
        """Release the GPU. Ollama reloads on demand."""
        with self._lock:
            if self._owner == owner:
                self._owner = None


# Module-level singleton shared by marker client, pipeline, and embedding.
gpu_lock = GpuLock()


__all__ = ["GpuLock", "gpu_lock"]
