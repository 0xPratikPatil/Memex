"""Unit tests for filter_tools module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from rag.filter_tools import (
    ExtractedFilters,
    FieldInfo,
    FilterContext,
    _build_extraction_prompt,
    _classify_field,
    _get_known_metadata_fields,
    _parse_llm_response,
    extract_filters,
    get_filter_context,
)

# ── Dataclass tests ─────────────────────────────────────────────────────────


class TestFieldInfo:
    def test_basic_construction(self) -> None:
        f = FieldInfo(name="doc_type", type="string", values=["report", "email"], count=42)
        assert f.name == "doc_type"
        assert f.type == "string"
        assert f.values == ["report", "email"]
        assert f.count == 42

    def test_empty_values(self) -> None:
        f = FieldInfo(name="topics", type="list", values=[], count=0)
        assert f.values == []
        assert f.count == 0


class TestFilterContext:
    def test_defaults(self) -> None:
        ctx = FilterContext(fields=[])
        assert ctx.fields == []
        assert ctx.suggested_filters is None
        assert ctx.sample_query == ""

    def test_with_suggestion(self) -> None:
        ctx = FilterContext(
            fields=[FieldInfo(name="lang", type="string", values=["en"], count=1)],
            suggested_filters={"language": "en"},
            sample_query="find english docs",
        )
        assert ctx.suggested_filters == {"language": "en"}


class TestExtractedFilters:
    def test_basic(self) -> None:
        ef = ExtractedFilters(filters={"doc_type": "report"}, explanation="query mentions report", confidence=0.8)
        assert ef.filters == {"doc_type": "report"}
        assert ef.confidence == 0.8


# ── classify_field ──────────────────────────────────────────────────────────


class TestClassifyField:
    def test_string_values(self) -> None:
        assert _classify_field(["hello", "world"]) == "string"

    def test_integer_values(self) -> None:
        assert _classify_field([2024, 2025]) == "integer"

    def test_list_values(self) -> None:
        assert _classify_field([["a", "b"], ["c"]]) == "list"

    def test_empty(self) -> None:
        assert _classify_field([]) == "string"


# ── Known metadata fields fallback ─────────────────────────────────────────


class TestKnownMetadataFields:
    def test_returns_expected_fields(self) -> None:
        fields = _get_known_metadata_fields()
        names = [f.name for f in fields]
        assert "doc_type" in names
        assert "topics" in names
        assert "language" in names
        assert "keywords" in names
        assert "entities" in names
        assert "dates" in names

    def test_all_have_zero_count(self) -> None:
        fields = _get_known_metadata_fields()
        assert all(f.count == 0 for f in fields)


# ── get_filter_context ──────────────────────────────────────────────────────


class TestGetFilterContext:
    @pytest.mark.asyncio
    async def test_no_qdrant_returns_known_fields(self) -> None:
        ctx = await get_filter_context(config=MagicMock(), qdrant_client=None)
        assert len(ctx.fields) > 0
        assert ctx.suggested_filters is None

    @pytest.mark.asyncio
    async def test_empty_collection_returns_known_fields(self) -> None:
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)

        ctx = await get_filter_context(config=MagicMock(), qdrant_client=mock_client, collection="test")
        assert len(ctx.fields) > 0  # falls back to known fields

    @pytest.mark.asyncio
    async def test_discovers_fields_from_scroll(self) -> None:
        # Create mock points with payloads
        point1 = MagicMock()
        point1.payload = {
            "doc_type": "report",
            "topics": ["finance", "revenue"],
            "language": "en",
            "content": "some content",
            "source": "/doc.pdf",
        }
        point2 = MagicMock()
        point2.payload = {
            "doc_type": "email",
            "topics": ["operations"],
            "language": "en",
            "content": "other content",
            "source": "/doc2.pdf",
        }

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([point1, point2], None)

        ctx = await get_filter_context(config=MagicMock(), qdrant_client=mock_client, collection="test")

        field_names = {f.name for f in ctx.fields}
        assert "doc_type" in field_names
        assert "topics" in field_names
        assert "language" in field_names
        # Internal keys should be excluded
        assert "content" not in field_names
        assert "source" not in field_names

    @pytest.mark.asyncio
    async def test_scroll_exception_falls_back(self) -> None:
        mock_client = MagicMock()
        mock_client.scroll.side_effect = RuntimeError("connection lost")

        ctx = await get_filter_context(config=MagicMock(), qdrant_client=mock_client)
        assert len(ctx.fields) > 0  # fallback

    @pytest.mark.asyncio
    async def test_suggests_filters_with_query(self) -> None:
        point = MagicMock()
        point.payload = {"doc_type": "report", "language": "en", "content": "x", "source": "/a"}

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([point], None)

        mock_llm = AsyncMock(return_value='{"doc_type": "report"}\nQuery mentions report')

        ctx = await get_filter_context(
            config=MagicMock(),
            query="find reports",
            qdrant_client=mock_client,
            collection="test",
            llm_call=mock_llm,
        )
        assert ctx.suggested_filters is not None
        assert "doc_type" in ctx.suggested_filters

    @pytest.mark.asyncio
    async def test_llm_failure_does_not_crash(self) -> None:
        point = MagicMock()
        point.payload = {"doc_type": "report", "content": "x", "source": "/a"}

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([point], None)

        mock_llm = AsyncMock(side_effect=RuntimeError("LLM down"))

        ctx = await get_filter_context(
            config=MagicMock(),
            query="test",
            qdrant_client=mock_client,
            llm_call=mock_llm,
        )
        # Should still return fields, just no suggestion
        assert len(ctx.fields) > 0
        assert ctx.suggested_filters is None


# ── extract_filters ─────────────────────────────────────────────────────────


class TestExtractFilters:
    @pytest.mark.asyncio
    async def test_no_llm_returns_empty(self) -> None:
        result = await extract_filters("test query", [FieldInfo("x", "string", [], 0)], llm_call=None)
        assert result.filters == {}
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_no_fields_returns_empty(self) -> None:
        result = await extract_filters("test", [], llm_call=AsyncMock())
        assert result.filters == {}
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_parses_json_response(self) -> None:
        fields = [FieldInfo("doc_type", "string", ["report", "email"], 10)]
        mock_llm = AsyncMock(return_value='{"doc_type": "report"}\nUser asks about reports')

        result = await extract_filters("find reports", fields, llm_call=mock_llm)
        assert result.filters == {"doc_type": "report"}
        assert result.confidence > 0.5

    @pytest.mark.asyncio
    async def test_parses_json_with_code_fence(self) -> None:
        fields = [FieldInfo("language", "string", ["en", "fr"], 5)]
        mock_llm = AsyncMock(
            return_value='```json\n{"language": "en"}\n```\nQuery is in English'
        )

        result = await extract_filters("english docs", fields, llm_call=mock_llm)
        assert result.filters == {"language": "en"}

    @pytest.mark.asyncio
    async def test_drops_unknown_fields(self) -> None:
        fields = [FieldInfo("doc_type", "string", ["report"], 5)]
        mock_llm = AsyncMock(
            return_value='{"doc_type": "report", "nonexistent": "value"}\nFiltered'
        )

        result = await extract_filters("find reports", fields, llm_call=mock_llm)
        assert "doc_type" in result.filters
        assert "nonexistent" not in result.filters

    @pytest.mark.asyncio
    async def test_handles_invalid_json(self) -> None:
        fields = [FieldInfo("doc_type", "string", ["report"], 5)]
        mock_llm = AsyncMock(return_value="I think you should filter by doc_type report")

        result = await extract_filters("reports", fields, llm_call=mock_llm)
        assert result.filters == {}
        assert result.confidence < 0.5

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty(self) -> None:
        fields = [FieldInfo("doc_type", "string", ["report"], 5)]
        mock_llm = AsyncMock(side_effect=RuntimeError("timeout"))

        result = await extract_filters("test", fields, llm_call=mock_llm)
        assert result.filters == {}
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_list_field_values(self) -> None:
        fields = [FieldInfo("topics", "list", ["finance", "tech"], 10)]
        mock_llm = AsyncMock(
            return_value='{"topics": ["finance"]}\nQuery is about finance'
        )

        result = await extract_filters("finance docs", fields, llm_call=mock_llm)
        assert result.filters == {"topics": ["finance"]}

    @pytest.mark.asyncio
    async def test_empty_validated_filters_low_confidence(self) -> None:
        fields = [FieldInfo("doc_type", "string", ["report"], 5)]
        # LLM returns a non-dict JSON
        mock_llm = AsyncMock(return_value='["not", "a", "dict"]\nNo idea')

        result = await extract_filters("test", fields, llm_call=mock_llm)
        assert result.filters == {}


# ── Prompt builder ──────────────────────────────────────────────────────────


class TestBuildExtractionPrompt:
    def test_contains_query(self) -> None:
        fields = [FieldInfo("doc_type", "string", ["report"], 5)]
        prompt = _build_extraction_prompt("find reports", fields)
        assert "find reports" in prompt

    def test_contains_field_names(self) -> None:
        fields = [
            FieldInfo("doc_type", "string", ["report", "email"], 10),
            FieldInfo("topics", "list", ["finance"], 5),
        ]
        prompt = _build_extraction_prompt("test", fields)
        assert "doc_type" in prompt
        assert "topics" in prompt

    def test_contains_field_values(self) -> None:
        fields = [FieldInfo("language", "string", ["en", "fr", "de"], 10)]
        prompt = _build_extraction_prompt("test", fields)
        assert "en" in prompt
        assert "fr" in prompt

    def test_empty_values_note(self) -> None:
        fields = [FieldInfo("new_field", "string", [], 0)]
        prompt = _build_extraction_prompt("test", fields)
        assert "no values indexed yet" in prompt.lower() or "(no values" in prompt.lower()


# ── Response parser ─────────────────────────────────────────────────────────


class TestParseLlmResponse:
    def test_valid_json(self) -> None:
        response = '{"doc_type": "report"}\nFiltered for reports'
        filters, explanation, confidence = _parse_llm_response(response)
        assert filters == {"doc_type": "report"}
        assert "report" in explanation.lower()
        assert confidence > 0.5

    def test_json_with_code_fence(self) -> None:
        response = '```json\n{"language": "en"}\n```\nEnglish docs'
        filters, _explanation, _confidence = _parse_llm_response(response)
        assert filters == {"language": "en"}

    def test_invalid_json_falls_back(self) -> None:
        response = "Just filter by doc type"
        filters, _explanation, confidence = _parse_llm_response(response)
        assert filters == {}
        assert confidence < 0.5

    def test_empty_response(self) -> None:
        filters, _explanation, confidence = _parse_llm_response("")
        assert filters == {}
        assert confidence == 0.1

    def test_non_dict_json(self) -> None:
        response = '["a", "b"]\nSome explanation'
        filters, _explanation, _confidence = _parse_llm_response(response)
        assert filters == {}

    def test_explanation_prefix_stripped(self) -> None:
        response = '{"x": "y"}\nExplanation: because of the query'
        _, explanation, _ = _parse_llm_response(response)
        assert not explanation.lower().startswith("explanation:")

    def test_multiple_filters(self) -> None:
        response = (
            '{"doc_type": "report", "language": "en", "topics": ["finance"]}'
            "\nThe query mentions reports in English about finance"
        )
        filters, _explanation, confidence = _parse_llm_response(response)
        assert len(filters) == 3
        assert confidence > 0.7
