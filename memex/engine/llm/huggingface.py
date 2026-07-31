"""HuggingFace embedding provider using ``sentence-transformers``.

Requires ``pip install sentence-transformers``.
"""

from __future__ import annotations

import logging
import os

from memex.engine.llm.base import EmbedProvider

logger = logging.getLogger(__name__)

# Tell HF not to phone home during unit tests / CI
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


class HuggingFaceEmbedder(EmbedProvider):
    """In-process embeddings via ``sentence-transformers``.

    Model is loaded on first embed() call (lazy init).

    Config keys: ``embedding.model``.
    """

    def __init__(self, model: str = "BAAI/bge-m3") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except ImportError as err:
            raise ImportError(
                "HuggingFace embedder requires the 'sentence-transformers' package.\n"
                "Install it with: pip install sentence-transformers"
            ) from err
        self._model_name = model
        self._model = None

    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            name = model or self._model_name
            logger.info("Loading HuggingFace model: %s", name)
            self._model = SentenceTransformer(name)

        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [vec.tolist() for vec in embeddings]
