"""Unit tests for metadata extractor module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from memex.engine.core import config
from memex.engine.metadata.extractor import MetadataExtractor


@pytest.fixture
def mock_llm() -> MagicMock:
    provider = MagicMock()

    def _chat_sync(prompt: str, *, model: str | None = None, num_predict: int | None = None) -> str:
        if "document classifier" in prompt:
            return "report"
        elif "named-entity extractor" in prompt:
            return json.dumps(
                {
                    "people": ["Alice Smith"],
                    "organizations": ["Acme Corp"],
                    "dates": ["2026-01-15"],
                    "locations": ["New York"],
                    "products": ["Widget Pro"],
                }
            )
        elif "topic labels" in prompt:
            return '["finance", "technology"]'
        elif "short summary" in prompt:
            return "This document discusses quarterly revenue."
        elif "keywords" in prompt:
            return json.dumps({"keywords": ["revenue", "growth", "quarterly"]})
        return "mocked LLM response"

    provider.chat_sync = MagicMock(side_effect=_chat_sync)
    return provider


@pytest.fixture
def extractor(mock_llm: MagicMock) -> MetadataExtractor:
    return MetadataExtractor(mock_llm)


# ── Entity extraction ─────────────────────────────────────────────────────


class TestEntityExtraction:
    def test_extract_entities_returns_dict(self, extractor: MetadataExtractor) -> None:
        entities = extractor.extract_entities("Alice Smith from Acme Corp reported revenue of $10M.")
        assert isinstance(entities, dict)

    def test_extract_entities_returns_expected_keys(self, extractor: MetadataExtractor) -> None:
        entities = extractor.extract_entities("Alice Smith from Acme Corp reported revenue of $10M.")
        # All five entity types should be present
        for key in ("people", "organizations", "dates", "locations", "products"):
            assert key in entities
            assert isinstance(entities[key], list)

    def test_extract_entities_limits_count(self, extractor: MetadataExtractor) -> None:
        entities = extractor.extract_entities("Test text with many entities.")
        for _key, values in entities.items():
            assert len(values) <= config.MAX_ENTITIES_PER_CHUNK

    def test_extract_entities_no_client_returns_empty(self) -> None:
        ext = MetadataExtractor(None)
        entities = ext.extract_entities("Test text.")
        assert entities == {}

    def test_extract_entities_handles_llm_error(self) -> None:
        provider = MagicMock()
        provider.chat_sync = MagicMock(side_effect=Exception("timeout"))
        ext = MetadataExtractor(provider)
        entities = ext.extract_entities("Test text.")
        assert entities == {}

    def test_extract_entities_handles_json_error(self) -> None:
        provider = MagicMock()
        provider.chat_sync = MagicMock(return_value="not json at all")
        ext = MetadataExtractor(provider)
        entities = ext.extract_entities("Test text.")
        assert entities == {}


# ── Document classification ────────────────────────────────────────────────


class TestDocumentClassification:
    def test_classify_returns_string(self, extractor: MetadataExtractor) -> None:
        doc_type = extractor.classify_document("This quarterly report presents financial results for Q3 2026.")
        assert isinstance(doc_type, str)

    def test_classify_returns_valid_type(self, extractor: MetadataExtractor) -> None:
        valid = {
            "report",
            "email",
            "article",
            "code",
            "documentation",
            "presentation",
            "resume",
            "contract",
            "invoice",
            "meeting_notes",
            "other",
            "unknown",
        }
        doc_type = extractor.classify_document("This quarterly report presents financial results.")
        assert doc_type in valid

    def test_classify_no_client_returns_unknown(self) -> None:
        ext = MetadataExtractor(None)
        assert ext.classify_document("Test.") == "unknown"

    def test_classify_handles_llm_error(self) -> None:
        provider = MagicMock()
        provider.chat_sync = MagicMock(side_effect=Exception("timeout"))
        ext = MetadataExtractor(provider)
        assert ext.classify_document("Test.") == "unknown"

    def test_classify_normalizes_output(self) -> None:
        provider = MagicMock()
        provider.chat_sync = MagicMock(return_value="  Report  ")
        ext = MetadataExtractor(provider)
        assert ext.classify_document("Test.") == "report"


# ── Topic extraction ───────────────────────────────────────────────────────


class TestTopicExtraction:
    def test_extract_topics_returns_list(self, extractor: MetadataExtractor) -> None:
        topics = extractor.extract_topics("This document covers quarterly financial analysis and revenue forecasting.")
        assert isinstance(topics, list)

    def test_extract_topics_limits_count(self, extractor: MetadataExtractor) -> None:
        topics = extractor.extract_topics("This document covers quarterly financial analysis and revenue forecasting.")
        assert len(topics) <= config.MAX_TOPICS_PER_CHUNK

    def test_extract_topics_no_client_returns_empty(self) -> None:
        ext = MetadataExtractor(None)
        assert ext.extract_topics("Test.") == []

    def test_extract_topics_handles_json_error(self) -> None:
        provider = MagicMock()
        provider.chat_sync = MagicMock(return_value="not json")
        ext = MetadataExtractor(provider)
        assert ext.extract_topics("Test.") == []


# ── Language detection ─────────────────────────────────────────────────────


class TestLanguageDetection:
    def test_detect_language_returns_string(self, extractor: MetadataExtractor) -> None:
        lang = extractor.detect_language("This is a test document in English with enough words for detection.")
        assert isinstance(lang, str)

    def test_detect_language_returns_iso_code(self, extractor: MetadataExtractor) -> None:
        lang = extractor.detect_language("This is a test document in English with enough words for detection.")
        # Should be a 2-letter code or empty
        assert len(lang) <= 3

    def test_detect_language_short_text_returns_empty(self) -> None:
        ext = MetadataExtractor(None)
        assert ext.detect_language("Hi.") == ""

    def test_detect_language_empty_returns_empty(self) -> None:
        ext = MetadataExtractor(None)
        assert ext.detect_language("") == ""

    def test_detect_language_handles_import_error(self) -> None:
        ext = MetadataExtractor(None)
        with patch.dict("sys.modules", {"langdetect": None}):
            lang = ext.detect_language("This is a test document with enough text for detection.")
            assert lang == ""


# ── Keyword extraction ─────────────────────────────────────────────────────


class TestKeywordExtraction:
    def test_extract_keywords_returns_list(self, extractor: MetadataExtractor) -> None:
        keywords = extractor.extract_keywords(
            "Revenue analysis shows significant growth in quarterly financial reports."
        )
        assert isinstance(keywords, list)

    def test_extract_keywords_returns_lowercase(self) -> None:
        ext = MetadataExtractor(None)
        keywords = ext.extract_keywords("Revenue ANALYSIS shows significant growth in QUARTERLY financial reports.")
        for kw in keywords:
            assert kw == kw.lower()

    def test_extract_keywords_finds_relevant_words(self) -> None:
        ext = MetadataExtractor(None)
        keywords = ext.extract_keywords("Revenue analysis shows significant growth in quarterly financial reports.")
        assert "revenue" in keywords or "analysis" in keywords

    def test_extract_keywords_filters_stopwords(self) -> None:
        ext = MetadataExtractor(None)
        keywords = ext.extract_keywords("This is the analysis that shows growth and revenue.")
        # Common stopwords should be filtered
        assert "this" not in keywords
        assert "that" not in keywords

    def test_extract_keywords_limits_to_ten(self) -> None:
        ext = MetadataExtractor(None)
        long_text = " ".join(f"word{i!s} " * 3 for i in range(20))
        keywords = ext.extract_keywords(long_text)
        assert len(keywords) <= 10

    def test_extract_keywords_handles_code_blocks(self) -> None:
        ext = MetadataExtractor(None)
        keywords = ext.extract_keywords("Revenue analysis ```def foo(): pass``` shows growth.")
        assert isinstance(keywords, list)


# ── Structural metadata ────────────────────────────────────────────────────


class TestStructuralMetadata:
    def test_extract_structural_returns_dict(self) -> None:
        ext = MetadataExtractor(None)
        chunk = {
            "content": "Revenue increased by 15%.",
            "section_header": "## Revenue",
            "chunk_index": 0,
            "total_chunks": 5,
        }
        structural = ext.extract_structural(chunk, "/docs/report.pdf")
        assert isinstance(structural, dict)

    def test_extract_structural_heading_level(self) -> None:
        ext = MetadataExtractor(None)
        chunk = {"content": "Test.", "section_header": "### subsection"}
        structural = ext.extract_structural(chunk, "")
        assert structural["heading_level"] == 3

    def test_extract_structural_heading_level_zero(self) -> None:
        ext = MetadataExtractor(None)
        chunk = {"content": "Test.", "section_header": ""}
        structural = ext.extract_structural(chunk, "")
        assert structural["heading_level"] == 0

    def test_extract_structural_is_list(self) -> None:
        ext = MetadataExtractor(None)
        chunk = {"content": "1. First item\n2. Second item", "section_header": ""}
        structural = ext.extract_structural(chunk, "")
        assert structural["is_list"] is True

    def test_extract_structural_is_code(self) -> None:
        ext = MetadataExtractor(None)
        chunk = {"content": "```python\nprint('hello')\n```", "section_header": ""}
        structural = ext.extract_structural(chunk, "")
        assert structural["is_code"] is True

    def test_extract_structural_is_table(self) -> None:
        ext = MetadataExtractor(None)
        chunk = {
            "content": "| Name | Value |\n|------|-------|\n| A | 1 |",
            "section_header": "",
        }
        structural = ext.extract_structural(chunk, "")
        assert structural["is_table"] is True

    def test_extract_structural_counts(self) -> None:
        ext = MetadataExtractor(None)
        chunk = {
            "content": "Visit https://example.com or email user@test.com",
            "section_header": "",
        }
        structural = ext.extract_structural(chunk, "")
        assert structural["link_count"] == 1
        assert structural["email_count"] == 1

    def test_extract_structural_word_count(self) -> None:
        ext = MetadataExtractor(None)
        chunk = {"content": "one two three four", "section_header": ""}
        structural = ext.extract_structural(chunk, "")
        assert structural["word_count"] == 4

    def test_extract_structural_chunk_index(self) -> None:
        ext = MetadataExtractor(None)
        chunk = {"content": "Test.", "section_header": "", "chunk_index": 3, "total_chunks": 10}
        structural = ext.extract_structural(chunk, "")
        assert structural["chunk_index"] == 3
        assert structural["total_chunks"] == 10


# ── extract_all ────────────────────────────────────────────────────────────


class TestExtractAll:
    def test_extract_all_merges_metadata(self, extractor: MetadataExtractor) -> None:
        chunk = {
            "content": "Alice Smith from Acme Corp reported revenue of $10M on 2026-01-15.",
            "section_header": "## Revenue",
            "chunk_index": 0,
            "total_chunks": 5,
        }
        metadata = extractor.extract_all(
            chunk=chunk,
            document_text="Quarterly report from Acme Corp.",
            source_identifier="/docs/report.pdf",
        )
        assert "structural" in metadata
        assert metadata["structural"]["heading_level"] == 2

    def test_extract_all_disabled_features_not_extracted(self) -> None:
        with patch.multiple(
            config,
            ENABLE_ENTITY_EXTRACTION=False,
            ENABLE_DOC_CLASSIFICATION=False,
            ENABLE_TOPIC_TAGGING=False,
            ENABLE_LANGUAGE_DETECTION=False,
        ):
            ext = MetadataExtractor(None)
            chunk = {"content": "Test content with enough words for detection.", "section_header": ""}
            metadata = ext.extract_all(chunk=chunk)
            assert "entities" not in metadata
            assert "doc_type" not in metadata
            assert "topics" not in metadata
            assert "language" not in metadata
            # Structural and keywords always run
            assert "structural" in metadata

    def test_extract_all_first_chunk_gets_doc_type(self, extractor: MetadataExtractor) -> None:
        chunk = {
            "content": "This is a quarterly report about financial results.",
            "section_header": "",
            "chunk_index": 0,
            "total_chunks": 5,
        }
        metadata = extractor.extract_all(
            chunk=chunk,
            document_text="Quarterly report.",
        )
        assert "doc_type" in metadata

    def test_extract_all_non_first_chunk_skips_doc_type(self, extractor: MetadataExtractor) -> None:
        chunk = {
            "content": "This section discusses revenue.",
            "section_header": "## Revenue",
            "chunk_index": 2,
            "total_chunks": 5,
        }
        metadata = extractor.extract_all(chunk=chunk)
        assert "doc_type" not in metadata

    def test_extract_all_keywords_always_present(self) -> None:
        with patch.multiple(
            config,
            ENABLE_ENTITY_EXTRACTION=False,
            ENABLE_DOC_CLASSIFICATION=False,
            ENABLE_TOPIC_TAGGING=False,
            ENABLE_LANGUAGE_DETECTION=False,
        ):
            ext = MetadataExtractor(None)
            chunk = {"content": "Revenue analysis shows significant growth.", "section_header": ""}
            metadata = ext.extract_all(chunk=chunk)
            assert "keywords" in metadata
            assert isinstance(metadata["keywords"], list)

    def test_extract_all_dates_from_entities(self) -> None:
        with patch.multiple(
            config,
            ENABLE_ENTITY_EXTRACTION=True,
            ENABLE_DOC_CLASSIFICATION=False,
            ENABLE_TOPIC_TAGGING=False,
            ENABLE_LANGUAGE_DETECTION=False,
        ):
            ext = MetadataExtractor(None)
            chunk = {"content": "Report dated 2026-01-15.", "section_header": ""}
            metadata = ext.extract_all(chunk=chunk)
            # Without LLM, entities is empty, so no dates
            assert "entities" not in metadata or metadata["entities"] == {}

    def test_extract_all_structural_always_present(self) -> None:
        with patch.multiple(
            config,
            ENABLE_ENTITY_EXTRACTION=False,
            ENABLE_DOC_CLASSIFICATION=False,
            ENABLE_TOPIC_TAGGING=False,
            ENABLE_LANGUAGE_DETECTION=False,
        ):
            ext = MetadataExtractor(None)
            chunk = {"content": "Test.", "section_header": "## Section"}
            metadata = ext.extract_all(chunk=chunk)
            assert "structural" in metadata
            assert metadata["structural"]["heading_level"] == 2


# ── _chat helper ───────────────────────────────────────────────────────────


class TestChatHelper:
    def test_chat_returns_empty_without_client(self) -> None:
        ext = MetadataExtractor(None)
        result = ext._chat("test prompt")
        assert result == ""

    def test_chat_calls_llm(self, extractor: MetadataExtractor) -> None:
        result = extractor._chat("test prompt")
        assert isinstance(result, str)
        extractor._llm.chat_sync.assert_called()

    def test_chat_handles_transport_error(self) -> None:
        provider = MagicMock()
        provider.chat_sync = MagicMock(side_effect=Exception("connection refused"))
        ext = MetadataExtractor(provider)
        with pytest.raises(Exception):  # noqa: B017
            ext._chat("test prompt")
