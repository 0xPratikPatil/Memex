"""Cross-file embedding batch accumulator.

Small files embed 1-3 texts per Ollama call — GPU time wasted on tiny
batches. The accumulator pools dense-embedding requests from concurrent
ingest workers (multiple files at once) and flushes them as ONE provider
call when ``embedding.batch_size`` texts accumulate or a short deadline
elapses, whichever comes first.

Usage::

    from memex.engine.ingestion.embed_batcher import submit_dense

    vecs = submit_dense(["text a", "text b"])   # may batch with other files
    vecs = submit_dense(["text c"])             # waits for pool flush

Thread-safe. Falls back to a direct per-call embedding on any batch
failure, so a broken batch never loses data — it just degrades to the
old per-file path.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future
from typing import TYPE_CHECKING

from memex.engine.core import config

if TYPE_CHECKING:
    from memex.engine.ingestion.embedding import EmbeddingService

    EmbeddingServiceLike = EmbeddingService
else:
    EmbeddingServiceLike = object

logger = logging.getLogger("embed-batcher")

# Flush when this many texts accumulate, or this long since the first text
# arrived (keeps small corpora from stalling on a never-filling batch).
_FLUSH_TIMEOUT_S = 0.3

_MISSING = object()


class _EmbedBatcher:
    """Pool texts across callers; flush as one batch."""

    def __init__(self, embed_fn) -> None:
        self._embed_fn = embed_fn
        self._lock = threading.Lock()
        self._texts: list[str] = []
        self._futures: list[Future] = []
        self._first_ts: float | None = None
        self._flush_thread = threading.Thread(
            target=self._timeout_flush, daemon=True, name="embed-batch-flush"
        )
        self._flush_thread.start()

    def submit(self, texts: list[str]) -> list[list[float]]:
        """Queue *texts* for embedding; returns vectors in input order.

        Blocks until the batch containing these texts is flushed.
        """
        if not texts:
            return []
        futures: list[Future] = []
        with self._lock:
            for t in texts:
                fut: Future = Future()
                futures.append(fut)
                self._texts.append(t)
                self._futures.append(fut)
            if self._first_ts is None:
                self._first_ts = time.monotonic()
            if len(self._texts) >= config.EMBED_BATCH_SIZE:
                self._flush_locked()
        return [f.result() for f in futures]

    def _flush_locked(self) -> None:
        """Flush pending texts (caller must hold self._lock)."""
        if not self._texts:
            return
        texts, futures = self._texts, self._futures
        self._texts, self._futures = [], []
        self._first_ts = None
        try:
            vecs = self._embed_fn(texts)
            if len(vecs) == len(texts):
                for fut, vec in zip(futures, vecs, strict=True):
                    fut.set_result(vec)
                return
            logger.warning(
                "Batch embedding returned %d vectors for %d texts — filling gaps per-text",
                len(vecs),
                len(texts),
            )
            for i, fut in enumerate(futures):
                if i < len(vecs):
                    fut.set_result(vecs[i])
                else:
                    try:
                        fut.set_result(self._embed_fn([texts[i]])[0])
                    except Exception as per_exc:
                        fut.set_exception(per_exc)
        except Exception as exc:
            logger.warning(
                "Batch embedding failed for %d texts, falling back per-call: %s",
                len(texts),
                exc,
            )
            # Resolve per-text via the direct path so callers get results
            # (or the same error) without losing their futures.
            for text, fut in zip(texts, futures, strict=True):
                try:
                    fut.set_result(self._embed_fn([text])[0])
                except Exception as per_exc:
                    fut.set_exception(per_exc)

    def _timeout_flush(self) -> None:
        """Flush when the pool sits under the batch size for too long."""
        while True:
            time.sleep(0.05)
            with self._lock:
                if (
                    self._first_ts is not None
                    and time.monotonic() - self._first_ts >= _FLUSH_TIMEOUT_S
                ):
                    self._flush_locked()


_service: EmbeddingServiceLike | None = None
_service_lock = threading.Lock()
_batcher: _EmbedBatcher | None = None
_batcher_lock = threading.Lock()


def _dense_embed_direct(texts: list[str]) -> list[list[float]]:
    """Direct embedding via a shared EmbeddingService (no batching)."""
    global _service
    with _service_lock:
        if _service is None:
            from memex.engine.llm import get_embedder

            from memex.engine.ingestion.embedding import EmbeddingService

            _service = EmbeddingService(get_embedder())
    return _service.embed(texts)


def submit_dense(texts: list[str]) -> list[list[float]]:
    """Embed *texts* via the shared cross-file batch accumulator.

    Returns vectors in input order. Falls back to direct embedding if the
    batcher is unavailable.
    """
    global _batcher
    with _batcher_lock:
        if _batcher is None:
            _batcher = _EmbedBatcher(_dense_embed_direct)
    return _batcher.submit(texts)


def flush_dense() -> None:
    """Flush any pending texts (used at process shutdown)."""
    global _batcher
    with _batcher_lock:
        batcher = _batcher
    if batcher is not None:
        with batcher._lock:
            batcher._flush_locked()


__all__ = ["submit_dense", "flush_dense"]
