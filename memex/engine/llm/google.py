"""Google Generative AI LLM provider using ``google-generativeai`` SDK.

Requires ``pip install google-generativeai``.
"""

from __future__ import annotations

import logging

from memex.engine.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class GoogleLLM(LLMProvider):
    """Google chat via the ``google-generativeai`` SDK.

    Config keys: ``llm.api_key``, ``llm.model``.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        try:
            import google.generativeai as genai
        except ImportError as err:
            raise ImportError(
                "Google provider requires the 'google-generativeai' package.\n"
                "Install it with: pip install google-generativeai"
            ) from err
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name=model)
        self._model_name = model

    async def chat(self, prompt: str, *, model: str | None = None, num_predict: int | None = None) -> str:
        import asyncio

        effective = model or self._model_name
        if effective != self._model_name:
            import google.generativeai as genai

            mdl = genai.GenerativeModel(model_name=effective)
        else:
            mdl = self._model

        kwargs: dict = {}
        if num_predict is not None:
            kwargs["generation_config"] = {"max_output_tokens": num_predict}
        response = await asyncio.to_thread(mdl.generate_content, prompt, **kwargs)
        return response.text
