"""FastEmbed embedding provider using the ``fastembed`` library.

Requires ``pip install fastembed``.
"""

from __future__ import annotations

import logging

from memex.engine.llm.base import EmbedProvider

logger = logging.getLogger(__name__)


class FastEmbedEmbedder(EmbedProvider):
    """In-process embeddings via ``fastembed``.

    Model is loaded on first embed() call (lazy init).

    Config keys: ``embedding.model``.
    """

    def __init__(self, model: str = "BAAI/bge-m3") -> None:
        try:
            from fastembed import TextEmbedding  # noqa: F401
        except ImportError as err:
            raise ImportError(
                "FastEmbed embedder requires the 'fastembed' package.\nInstall it with: pip install fastembed"
            ) from err
        self._model_name = model
        self._model = None

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        if self._model is None:
            from fastembed import TextEmbedding

            name = model or self._model_name
            logger.info("Loading FastEmbed model: %s", name)
            self._model = TextEmbedding(model_name=name)

        embeddings = list(self._model.embed(texts))
        return [vec.tolist() for vec in embeddings]
