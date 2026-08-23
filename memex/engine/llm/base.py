"""Abstract base classes for LLM and embedding providers."""

from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Transient failures worth retrying — a stalled/slow LLM call recovers on
# retry; deterministic errors (auth, bad request) do not.
_RETRYABLE_LLM_ERRORS = (httpx.TimeoutException, httpx.TransportError)

# 1 initial attempt + 2 retries, 2s/4s backoff.
_LLM_MAX_ATTEMPTS = 3

# Per-thread event loop for sync LLM calls. asyncio.run() per call creates a
# fresh loop every time; async clients (httpx) bound to a previous loop
# deadlock on the next call (await never resolves). One persistent loop per
# thread keeps clients stable and calls reliable.
_thread_loop_local = threading.local()

# Serializes sync LLM calls across worker threads. One model on one GPU can
# only serve one request at a time anyway, and concurrent async clients
# across threads race on the shared provider instance — a second thread's
# client swap orphans the first thread's in-flight request forever.
_llm_sync_lock = threading.Lock()


def _get_thread_loop() -> asyncio.AbstractEventLoop:
    loop = getattr(_thread_loop_local, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _thread_loop_local.loop = loop
    return loop


def _llm_retry(fn, *args, **kwargs):
    """Call fn with tenacity retry, returning (result, attempts_used).

    Args:
        fn: Callable returning the chat result.
        *args, **kwargs: Passed to fn.

    Returns:
        (result, attempts) tuple. attempts = number of tries before success
        or final failure (1 = no retry needed, 3 = both retries used).
    """
    attempts = {"n": 0}

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_LLM_ERRORS),
        stop=stop_after_attempt(_LLM_MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=2, min=2, max=4),
        reraise=True,
        before_sleep=lambda retry_state: attempts.__setitem__("n", retry_state.attempt_number),
    )
    def _call():
        return fn(*args, **kwargs)

    try:
        result = _call()
        return result, max(attempts["n"] + 1, 1)
    except Exception:
        raise


class LLMProvider(ABC):
    """Async chat completion interface.

    Implementations: OllamaLLM, OpenAILLM, AnthropicLLM, GroqLLM, GoogleLLM.
    """

    @abstractmethod
    async def chat(self, prompt: str, *, model: str | None = None, num_predict: int | None = None) -> str:
        """Generate a chat completion for a single-turn prompt."""

    def chat_sync(self, prompt: str, *, model: str | None = None, num_predict: int | None = None) -> str:
        """Synchronous wrapper for chat() with retry on transient failures."""
        try:
            _ = asyncio.get_running_loop()
        except RuntimeError:
            loop = _get_thread_loop()
            with _llm_sync_lock:
                result, _attempts = _llm_retry(
                    lambda: loop.run_until_complete(self.chat(prompt, model=model, num_predict=num_predict))
                )
            return result
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            result, _attempts = _llm_retry(
                lambda: pool.submit(asyncio.run, self.chat(prompt, model=model, num_predict=num_predict)).result()
            )
            return result

    def chat_sync_with_attempts(
        self, prompt: str, *, model: str | None = None, num_predict: int | None = None
    ) -> tuple[str, int]:
        """Synchronous chat returning (result, attempts_used) for logging."""
        try:
            _ = asyncio.get_running_loop()
        except RuntimeError:
            loop = _get_thread_loop()
            with _llm_sync_lock:
                return _llm_retry(
                    lambda: loop.run_until_complete(self.chat(prompt, model=model, num_predict=num_predict))
                )

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return _llm_retry(
                lambda: pool.submit(asyncio.run, self.chat(prompt, model=model, num_predict=num_predict)).result()
            )


class EmbedProvider(ABC):
    """Synchronous embedding interface.

    Returns one vector per input text. Implementations: OllamaEmbedder,
    OpenAIEmbedder, HuggingFaceEmbedder, FastEmbedEmbedder.
    """

    @abstractmethod
    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        """Generate dense embeddings for a batch of texts."""
