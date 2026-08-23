"""Tests for multi-provider LLM and embedding layer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memex.engine.llm.base import EmbedProvider, LLMProvider
from memex.engine.llm.groq import GroqLLM
from memex.engine.llm.ollama import OllamaEmbedder, OllamaLLM
from memex.engine.llm.openai import OpenAIEmbedder, OpenAILLM
from memex.engine.llm.openrouter import OpenRouterLLM

# ── get_llm factory ──────────────────────────────────────────────────────────


class TestGetLLM:
    def test_ollama_uses_configured_timeout(self) -> None:
        """get_llm should pass LLM_TIMEOUT to OllamaLLM (fixes ReadTimeout)."""
        import memex.engine.core.config as cfg
        from memex.engine.llm import get_llm

        with (
            patch.object(cfg, "LLM_PROVIDER", "ollama"),
            patch.object(cfg, "LLM_BASE_URL", "http://x:11434"),
            patch.object(cfg, "CHAT_MODEL", "qwen2.5:1.5b"),
            patch.object(cfg, "LLM_TIMEOUT", 300.0),
            patch("memex.engine.llm.ollama.OllamaLLM") as mock_llm_cls,
        ):
            get_llm()
            kwargs = mock_llm_cls.call_args[1]
            assert kwargs["timeout"] == 300.0


# ── Ollama LLM ──────────────────────────────────────────────────────────────


class TestOllamaLLM:
    """Tests for OllamaLLM provider."""

    @pytest.mark.asyncio
    async def test_chat_returns_content(self) -> None:
        """Should POST to /api/chat and return message content."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Hello from Ollama"},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.is_closed = False

        with patch.object(OllamaLLM, "_get_client", return_value=mock_client):
            llm = OllamaLLM(base_url="http://localhost:11434", model="test-model")
            result = await llm.chat("Hello")

        assert result == "Hello from Ollama"
        mock_client.post.assert_called_once()
        args = mock_client.post.call_args
        assert args[0][0] == "http://localhost:11434/api/chat"
        payload = args[1]["json"]
        assert payload["model"] == "test-model"
        assert payload["messages"][0]["content"] == "Hello"
        assert payload["stream"] is False

    @pytest.mark.asyncio
    async def test_chat_sends_num_predict(self) -> None:
        """Should forward num_predict into the Ollama options."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "ok"}}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.is_closed = False

        with patch.object(OllamaLLM, "_get_client", return_value=mock_client):
            llm = OllamaLLM(base_url="http://localhost:11434", model="test-model")
            await llm.chat("Hello", num_predict=200)

        payload = mock_client.post.call_args[1]["json"]
        assert payload["options"]["num_predict"] == 200

    @pytest.mark.asyncio
    async def test_chat_omits_num_predict_when_none(self) -> None:
        """Should not send num_predict when not provided."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "ok"}}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.is_closed = False

        with patch.object(OllamaLLM, "_get_client", return_value=mock_client):
            llm = OllamaLLM(base_url="http://localhost:11434", model="test-model")
            await llm.chat("Hello")

        payload = mock_client.post.call_args[1]["json"]
        assert "num_predict" not in payload["options"]

    @pytest.mark.asyncio
    async def test_chat_uses_override_model(self) -> None:
        """Should use the model override when provided."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "ok"}}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.is_closed = False

        with patch.object(OllamaLLM, "_get_client", return_value=mock_client):
            llm = OllamaLLM(base_url="http://localhost:11434", model="default-model")
            await llm.chat("Hi", model="custom-model")

        payload = mock_client.post.call_args[1]["json"]
        assert payload["model"] == "custom-model"

    @pytest.mark.asyncio
    async def test_chat_falls_back_to_thinking_field(self) -> None:
        """Should use thinking field when content is empty."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "", "thinking": "reasoning output"},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.is_closed = False

        with patch.object(OllamaLLM, "_get_client", return_value=mock_client):
            llm = OllamaLLM(base_url="http://localhost:11434")
            result = await llm.chat("Think")

        assert result == "reasoning output"

    def test_chat_sync(self) -> None:
        """chat_sync() should work within a non-async context."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "sync hello"}}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.is_closed = False

        with patch.object(OllamaLLM, "_get_client", return_value=mock_client):
            llm = OllamaLLM(base_url="http://localhost:11434")
            result = llm.chat_sync("sync test")

        assert result == "sync hello"

    @pytest.mark.asyncio
    async def test_chat_after_chat_sync_rebinds_client_to_loop(self) -> None:
        """chat_sync() runs on a per-thread loop with a per-thread client;
        a chat() on another loop/thread must get its own client — never the
        one bound to the other thread's loop."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "ok"}}
        mock_response.raise_for_status = MagicMock()

        clients: list[AsyncMock] = []

        def _make_client(*_args, **_kwargs) -> AsyncMock:
            client = AsyncMock()
            client.post.return_value = mock_response
            client.is_closed = False
            clients.append(client)
            return client

        llm = OllamaLLM(base_url="http://localhost:11434")
        with patch("httpx.AsyncClient", side_effect=_make_client):
            llm.chat_sync("sync test")
            await llm.chat("async test")

        # chat_sync (worker thread) and chat (test thread) each get a
        # dedicated client — never shared across threads/loops.
        assert len(clients) == 2
        assert llm._client_local.client is clients[-1]


# ── Ollama Embedder ────────────────────────────────────────────────────────


class TestOllamaEmbedder:
    """Tests for OllamaEmbedder provider."""

    def test_embed_returns_vectors(self) -> None:
        """Should POST to /api/embed and return embedding vectors."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.is_closed = False

        with patch.object(OllamaEmbedder, "_get_client", return_value=mock_client):
            embedder = OllamaEmbedder(base_url="http://localhost:11434", model="embed-model")
            vectors = embedder.embed(["text a", "text b"])

        assert len(vectors) == 2
        assert vectors[0] == [0.1, 0.2, 0.3]
        assert vectors[1] == [0.4, 0.5, 0.6]
        mock_client.post.assert_called_once()
        payload = mock_client.post.call_args[1]["json"]
        assert payload["model"] == "embed-model"
        assert payload["input"] == ["text a", "text b"]

    def test_embed_strips_api_prefix_from_base_url(self) -> None:
        """Should strip /api/embed or /api/embeddings from base_url."""
        embedder = OllamaEmbedder(base_url="http://local:1234/api/embed")
        assert embedder._base_url == "http://local:1234"

        embedder2 = OllamaEmbedder(base_url="http://local:1234/api/embeddings")
        assert embedder2._base_url == "http://local:1234"


# ── OpenAI LLM ─────────────────────────────────────────────────────────────


class TestOpenAILLM:
    """Tests for OpenAILLM provider."""

    @pytest.mark.asyncio
    async def test_chat_returns_content(self) -> None:
        """Should call /chat/completions with Bearer auth."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "GPT says hi"}}],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.is_closed = False

        with patch("memex.engine.llm.openai._OpenAIBase._get_client", return_value=mock_client):
            llm = OpenAILLM(api_key="sk-test", model="gpt-4o")
            result = await llm.chat("Hello from test")

        assert result == "GPT says hi"
        mock_client.post.assert_called_once_with(
            "/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello from test"}], "temperature": 0},
        )


# ── OpenAI Embedder ────────────────────────────────────────────────────────


class TestOpenAIEmbedder:
    """Tests for OpenAIEmbedder provider."""

    def test_embed_returns_vectors(self) -> None:
        """Should call /embeddings and parse response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2]},
                {"embedding": [0.3, 0.4]},
            ],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client.is_closed = False

        with patch.object(OpenAIEmbedder, "_get_sync_client", return_value=mock_client):
            embedder = OpenAIEmbedder(api_key="sk-test", model="text-embedding-3-small")
            vectors = embedder.embed(["a", "b"])

        assert len(vectors) == 2
        assert vectors[0] == [0.1, 0.2]
        assert vectors[1] == [0.3, 0.4]


# ── OpenRouter LLM ─────────────────────────────────────────────────────────


class TestOpenRouterLLM:
    """Tests for OpenRouterLLM provider."""

    @pytest.mark.asyncio
    async def test_chat_uses_openrouter_base_url(self) -> None:
        """Should call OpenRouter's base URL by default."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "router response"}}],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.is_closed = False

        with patch("memex.engine.llm.openrouter._OpenAIBase._get_client", return_value=mock_client):
            llm = OpenRouterLLM(api_key="sk-test", model="openai/gpt-4o")
            result = await llm.chat("hi")

        assert result == "router response"
        mock_client.post.assert_called_once()


# ── Groq LLM ───────────────────────────────────────────────────────────────


class TestGroqLLM:
    """Tests for GroqLLM provider."""

    @pytest.mark.asyncio
    async def test_chat_returns_content(self) -> None:
        """Should call Groq's OpenAI-compatible endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "groq fast reply"}}],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.is_closed = False

        with patch("memex.engine.llm.groq._OpenAIBase._get_client", return_value=mock_client):
            llm = GroqLLM(api_key="sk-groq", model="llama3-70b-8192")
            result = await llm.chat("quick test")

        assert result == "groq fast reply"
        mock_client.post.assert_called_once()


# ── Anthropic init-time check ──────────────────────────────────────────────


class TestAnthropicLLM:
    """Tests for AnthropicLLM provider."""

    def test_init_raises_import_error(self) -> None:
        """Should raise ImportError with helpful message when SDK missing."""
        with patch.dict("sys.modules", {"anthropic": None}), pytest.raises(ImportError, match="pip install anthropic"):
            from memex.engine.llm.anthropic import AnthropicLLM

            AnthropicLLM(api_key="sk-test")

    def test_init_succeeds_with_sdk(self) -> None:
        """Should initialize when anthropic is importable."""
        mock_anthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from memex.engine.llm.anthropic import AnthropicLLM

            provider = AnthropicLLM(api_key="sk-test", model="claude-3-sonnet")
            assert provider._model == "claude-3-sonnet"


# ── Google init-time check ─────────────────────────────────────────────────


class TestGoogleLLM:
    """Tests for GoogleLLM provider."""

    def test_init_raises_import_error(self) -> None:
        """Should raise ImportError with helpful message when SDK missing."""
        with (
            patch.dict("sys.modules", {"google": None, "google.generativeai": None}),
            pytest.raises(ImportError, match="pip install google-generativeai"),
        ):
            from memex.engine.llm.google import GoogleLLM

            GoogleLLM(api_key="fake")

    def test_init_succeeds_with_sdk(self) -> None:
        """Should initialize when google-generativeai is present."""
        mock_genai = MagicMock()
        mock_genai.GenerativeModel.return_value = MagicMock()
        mock_google = MagicMock()
        mock_google.generativeai = mock_genai
        with patch.dict("sys.modules", {"google": mock_google, "google.generativeai": mock_genai}):
            from memex.engine.llm.google import GoogleLLM

            provider = GoogleLLM(api_key="fake", model="gemini-pro")
            assert provider._model_name == "gemini-pro"


# ── HuggingFace init-time check ────────────────────────────────────────────


class TestHuggingFaceEmbedder:
    """Tests for HuggingFaceEmbedder provider."""

    def test_init_raises_import_error(self) -> None:
        """Should raise ImportError when sentence-transformers missing."""
        with (
            patch.dict("sys.modules", {"sentence_transformers": None}),
            pytest.raises(ImportError, match="pip install sentence-transformers"),
        ):
            from memex.engine.llm.huggingface import HuggingFaceEmbedder

            HuggingFaceEmbedder()

    def test_init_succeeds_with_sdk(self) -> None:
        """Should not load model eagerly, just store the name."""
        with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}):
            from memex.engine.llm.huggingface import HuggingFaceEmbedder

            emb = HuggingFaceEmbedder(model="some/model")
            assert emb._model_name == "some/model"
            assert emb._model is None


# ── FastEmbed init-time check ──────────────────────────────────────────────


class TestFastEmbedEmbedder:
    """Tests for FastEmbedEmbedder provider."""

    def test_init_raises_import_error(self) -> None:
        """Should raise ImportError when fastembed missing."""
        with patch.dict("sys.modules", {"fastembed": None}), pytest.raises(ImportError, match="pip install fastembed"):
            from memex.engine.llm.fastembed import FastEmbedEmbedder

            FastEmbedEmbedder()

    def test_init_succeeds_with_sdk(self) -> None:
        """Should not load model eagerly."""
        with patch.dict("sys.modules", {"fastembed": MagicMock()}):
            from memex.engine.llm.fastembed import FastEmbedEmbedder

            emb = FastEmbedEmbedder(model="BAAI/bge-small-en")
            assert emb._model_name == "BAAI/bge-small-en"
            assert emb._model is None


# ── Factory functions ──────────────────────────────────────────────────────


class TestFactories:
    """Tests for get_llm and get_embedder factory functions."""

    def test_get_llm_defaults_to_llm(self) -> None:
        """get_llm with default config returns OllamaLLM."""
        from unittest.mock import MagicMock

        lookup = {
            "llm.provider": "ollama",
            "llm.base_url": "http://localhost:11437",
            "llm.api_key": "",
            "llm.model": "llama3",
        }
        mock_cfg = MagicMock()
        mock_cfg.get_str.side_effect = lambda k, d="": lookup.get(k, d)

        from memex.engine.llm import get_llm

        provider = get_llm(mock_cfg)
        assert isinstance(provider, OllamaLLM)
        assert provider._model == "llama3"

    def test_get_llm_returns_openai(self) -> None:
        """get_llm should return OpenAILLM when provider=openai."""
        from unittest.mock import MagicMock

        lookup = {
            "llm.provider": "openai",
            "llm.base_url": "",
            "llm.api_key": "sk-abc",
            "llm.model": "gpt-4o-mini",
        }
        mock_cfg = MagicMock()
        mock_cfg.get_str.side_effect = lambda k, d="": lookup.get(k, d)

        from memex.engine.llm import get_llm

        provider = get_llm(mock_cfg)
        assert isinstance(provider, OpenAILLM)
        assert provider._http._model == "gpt-4o-mini"

    def test_get_llm_returns_openrouter(self) -> None:
        """get_llm should return OpenRouterLLM when provider=openrouter."""
        from unittest.mock import MagicMock

        lookup = {
            "llm.provider": "openrouter",
            "llm.base_url": "",
            "llm.api_key": "sk-xyz",
            "llm.model": "anthropic/claude-3-sonnet",
        }
        mock_cfg = MagicMock()
        mock_cfg.get_str.side_effect = lambda k, d="": lookup.get(k, d)

        from memex.engine.llm import get_llm

        provider = get_llm(mock_cfg)
        assert isinstance(provider, OpenRouterLLM)

    def test_get_llm_returns_groq(self) -> None:
        """get_llm should return GroqLLM when provider=groq."""
        from unittest.mock import MagicMock

        lookup = {
            "llm.provider": "groq",
            "llm.base_url": "",
            "llm.api_key": "sk-groq-test",
            "llm.model": "mixtral-8x7b",
        }
        mock_cfg = MagicMock()
        mock_cfg.get_str.side_effect = lambda k, d="": lookup.get(k, d)

        from memex.engine.llm import get_llm

        provider = get_llm(mock_cfg)
        assert isinstance(provider, GroqLLM)

    def test_get_embedder_defaults_to_llm(self) -> None:
        """get_embedder with default config returns OllamaEmbedder."""
        from unittest.mock import MagicMock

        lookup = {
            "embedding.provider": "ollama",
            "embedding.base_url": "http://localhost:11438",
            "embedding.api_key": "",
            "embedding.model": "nomic-embed-text",
        }
        mock_cfg = MagicMock()
        mock_cfg.get_str.side_effect = lambda k, d="": lookup.get(k, d)

        from memex.engine.llm import get_embedder

        provider = get_embedder(mock_cfg)
        assert isinstance(provider, OllamaEmbedder)
        assert provider._model == "nomic-embed-text"

    def test_get_embedder_returns_openai(self) -> None:
        """get_embedder should return OpenAIEmbedder when provider=openai."""
        from unittest.mock import MagicMock

        lookup = {
            "embedding.provider": "openai",
            "embedding.base_url": "",
            "embedding.api_key": "sk-embed",
            "embedding.model": "text-embedding-ada-002",
        }
        mock_cfg = MagicMock()
        mock_cfg.get_str.side_effect = lambda k, d="": lookup.get(k, d)

        from memex.engine.llm import get_embedder

        provider = get_embedder(mock_cfg)
        assert isinstance(provider, OpenAIEmbedder)

    def test_unknown_provider_falls_back_to_llm(self) -> None:
        """Unknown provider names should fall back to Ollama."""
        from unittest.mock import MagicMock

        lookup = {
            "llm.provider": "nonexistent",
            "llm.base_url": "http://localhost:10000",
            "llm.api_key": "",
            "llm.model": "test",
        }
        mock_cfg = MagicMock()
        mock_cfg.get_str.side_effect = lambda k, d="": lookup.get(k, d)

        from memex.engine.llm import get_llm

        provider = get_llm(mock_cfg)
        assert isinstance(provider, OllamaLLM)

    def test_chat_sync_runs_in_event_loop(self) -> None:
        """chat_sync should delegate to chat() and return the result."""

        # Test the base class chat_sync method
        class FakeLLM(LLMProvider):
            async def chat(self, prompt: str, *, model: str | None = None, num_predict: int | None = None) -> str:
                return f"echo: {prompt}"

        provider = FakeLLM()
        result = provider.chat_sync("hello")
        assert result == "echo: hello"

    def test_factory_with_none_config(self) -> None:
        """get_llm(None) should use module-level config from memex.engine.core.config."""
        from unittest.mock import patch as _patch

        with (
            _patch("memex.engine.core.config.LLM_PROVIDER", "ollama"),
            _patch("memex.engine.core.config.LLM_BASE_URL", ""),
            _patch("memex.engine.core.config.LLM_API_KEY", ""),
            _patch("memex.engine.core.config.CHAT_MODEL", "test-model"),
            _patch("memex.engine.core.config.OLLAMA_EMBED_URL", "http://localhost:11434"),
        ):
            from memex.engine.llm import get_llm

            provider = get_llm(None)
            assert isinstance(provider, OllamaLLM)
            assert provider._model == "test-model"


# ── EmbedProvider ABC ──────────────────────────────────────────────────────


class TestEmbedProviderABC:
    """Tests for the EmbedProvider abstract base."""

    def test_subclass_must_implement_embed(self) -> None:
        """Instantiating subclass without embed should fail."""

        class Incomplete(EmbedProvider):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]


class TestLLMProviderABC:
    """Tests for the LLMProvider abstract base."""

    def test_subclass_must_implement_chat(self) -> None:
        """Instantiating subclass without chat should fail."""

        class Incomplete(LLMProvider):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_chat_sync_retries_transient_errors(self) -> None:
        """chat_sync should retry transient failures (2 retries) then succeed."""
        import httpx

        class FlakyLLM(LLMProvider):
            def __init__(self) -> None:
                self.calls = 0

            async def chat(self, prompt: str, *, model: str | None = None, num_predict: int | None = None) -> str:
                self.calls += 1
                if self.calls < 3:
                    raise httpx.ReadTimeout("stalled")
                return "recovered"

        provider = FlakyLLM()
        result, attempts = provider.chat_sync_with_attempts("hello")
        assert result == "recovered"
        assert attempts == 3  # 1 initial + 2 retries
        assert provider.calls == 3

    def test_chat_sync_raises_after_max_retries(self) -> None:
        """chat_sync should raise after both retries are exhausted."""
        import httpx

        class AlwaysFailLLM(LLMProvider):
            def __init__(self) -> None:
                self.calls = 0

            async def chat(self, prompt: str, *, model: str | None = None, num_predict: int | None = None) -> str:
                self.calls += 1
                raise httpx.ConnectError("down")

        provider = AlwaysFailLLM()
        with pytest.raises(httpx.ConnectError):
            provider.chat_sync("hello")
        assert provider.calls == 3
