"""Factory functions for LLM and embedding providers.

Reads provider selection from config.yaml:

- ``llm.provider`` → which LLM to use (ollama, openai, anthropic, groq, google, openrouter)
- ``embedding.provider`` → which embedder to use (ollama, openai, huggingface, fastembed)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from memex.engine.llm.base import EmbedProvider, LLMProvider

if TYPE_CHECKING:
    from memex.engine.core.yaml_config import YamlConfig

logger = logging.getLogger(__name__)

# Aliases for easier imports — kept backward-compatible with older code
__all__ = [
    "EmbedProvider",
    "LLMProvider",
    "get_embedder",
    "get_llm",
]


def get_llm(config_obj: YamlConfig | None = None) -> LLMProvider:
    """Create an LLM provider based on ``llm.provider`` in config.

    Args:
        config_obj: Optional YamlConfig instance.  When ``None``, reads from
            the module-level config in ``memex.engine.core.config``.

    Returns:
        A concrete LLMProvider instance ready for chat() calls.
    """
    from memex.engine.core import config as cfg

    provider_name = (
        config_obj.get_str("llm.provider") if config_obj else cfg.LLM_PROVIDER
    ).lower()

    base_url = config_obj.get_str("llm.base_url") if config_obj else cfg.LLM_BASE_URL
    api_key = config_obj.get_str("llm.api_key") if config_obj else cfg.LLM_API_KEY
    model = config_obj.get_str("llm.model") if config_obj else cfg.CHAT_MODEL

    if provider_name == "ollama":
        from memex.engine.llm.ollama import OllamaLLM

        url = base_url or "http://localhost:11434"
        return OllamaLLM(base_url=url, model=model)

    if provider_name == "openai":
        from memex.engine.llm.openai import OpenAILLM

        return OpenAILLM(api_key=api_key, model=model)

    if provider_name == "openrouter":
        from memex.engine.llm.openrouter import OpenRouterLLM

        return OpenRouterLLM(api_key=api_key, model=model)

    if provider_name == "anthropic":
        from memex.engine.llm.anthropic import AnthropicLLM

        return AnthropicLLM(api_key=api_key, model=model)

    if provider_name == "groq":
        from memex.engine.llm.groq import GroqLLM

        return GroqLLM(api_key=api_key, model=model)

    if provider_name == "google":
        from memex.engine.llm.google import GoogleLLM

        return GoogleLLM(api_key=api_key, model=model)

    supported = "ollama, openai, openrouter, anthropic, groq, google"
    logger.warning("Unknown llm.provider=%r, falling back to ollama (supported: %s)", provider_name, supported)
    from memex.engine.llm.ollama import OllamaLLM

    return OllamaLLM(base_url=cfg.OLLAMA_EMBED_URL, model=model)


def get_embedder(config_obj: YamlConfig | None = None) -> EmbedProvider:
    """Create an embedding provider based on ``embedding.provider`` in config.

    Args:
        config_obj: Optional YamlConfig instance.  When ``None``, reads from
            the module-level config in ``memex.engine.core.config``.

    Returns:
        A concrete EmbedProvider instance ready for embed() calls.
    """
    from memex.engine.core import config as cfg

    provider_name = (
        config_obj.get_str("embedding.provider") if config_obj else cfg.EMBED_PROVIDER
    ).lower()

    base_url = config_obj.get_str("embedding.base_url") if config_obj else cfg.OLLAMA_EMBED_URL
    api_key = config_obj.get_str("embedding.api_key") if config_obj else cfg.EMBED_API_KEY
    model = config_obj.get_str("embedding.model") if config_obj else cfg.EMBED_MODEL

    if provider_name == "ollama":
        from memex.engine.llm.ollama import OllamaEmbedder

        url = base_url or "http://localhost:11434"
        return OllamaEmbedder(base_url=url, model=model)

    if provider_name == "openai":
        from memex.engine.llm.openai import OpenAIEmbedder

        return OpenAIEmbedder(api_key=api_key, model=model)

    if provider_name == "huggingface":
        from memex.engine.llm.huggingface import HuggingFaceEmbedder

        return HuggingFaceEmbedder(model=model)

    if provider_name == "fastembed":
        from memex.engine.llm.fastembed import FastEmbedEmbedder

        return FastEmbedEmbedder(model=model)

    supported = "ollama, openai, huggingface, fastembed"
    logger.warning(
        "Unknown embedding.provider=%r, falling back to ollama (supported: %s)",
        provider_name,
        supported,
    )
    from memex.engine.llm.ollama import OllamaEmbedder

    return OllamaEmbedder(base_url=cfg.OLLAMA_EMBED_URL, model=model)
