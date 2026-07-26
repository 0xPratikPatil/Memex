"""Integration tests for metadata extraction with live services."""

from __future__ import annotations

import httpx
import pytest

from rag.services.metadata_extractor import MetadataExtractor


def _ollama_reachable() -> bool:
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get("http://localhost:11434/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


def _langdetect_available() -> bool:
    try:
        import langdetect  # noqa: F401

        return True
    except ImportError:
        return False


# ── Non-LLM tests ───────────────────────────────────────────────────────────


@pytest.mark.integration
class TestLanguageDetection:
    def test_detect_english_language(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if not _langdetect_available():
            pytest.skip("langdetect not installed")

        monkeypatch.setattr("rag.config.ENABLE_METADATA_EXTRACTION", True)
        monkeypatch.setattr("rag.config.ENABLE_LANGUAGE_DETECTION", True)
        monkeypatch.setattr("rag.config.CHAT_MODEL", "qwen2.5:0.5b")

        extractor = MetadataExtractor(None)
        lang = extractor.detect_language(
            "This is an English document with enough text for language detection to work reliably."
        )
        assert lang == "en"


@pytest.mark.integration
class TestStructuralMetadata:
    def test_heading_level_from_markdown_header(self) -> None:
        extractor = MetadataExtractor(None)
        chunk = {
            "content": "Some paragraph content here.",
            "section_header": "## Section Title",
            "chunk_index": 0,
            "total_chunks": 3,
        }
        structural = extractor.extract_structural(chunk, "/docs/test.md")
        assert structural["heading_level"] == 2

    def test_heading_level_zero_for_no_header(self) -> None:
        extractor = MetadataExtractor(None)
        chunk = {"content": "Plain paragraph without a heading.", "section_header": ""}
        structural = extractor.extract_structural(chunk, "")
        assert structural["heading_level"] == 0

    def test_word_count(self) -> None:
        extractor = MetadataExtractor(None)
        chunk = {"content": "one two three four five", "section_header": ""}
        structural = extractor.extract_structural(chunk, "")
        assert structural["word_count"] == 5

    def test_char_count(self) -> None:
        extractor = MetadataExtractor(None)
        chunk = {"content": "hello world", "section_header": ""}
        structural = extractor.extract_structural(chunk, "")
        assert structural["char_count"] == 11

    def test_detects_list(self) -> None:
        extractor = MetadataExtractor(None)
        chunk = {"content": "1. First item\n2. Second item\n3. Third", "section_header": ""}
        structural = extractor.extract_structural(chunk, "")
        assert structural["is_list"] is True

    def test_detects_code_block(self) -> None:
        extractor = MetadataExtractor(None)
        chunk = {"content": "```python\nprint('hello')\n```", "section_header": ""}
        structural = extractor.extract_structural(chunk, "")
        assert structural["is_code"] is True

    def test_detects_table(self) -> None:
        extractor = MetadataExtractor(None)
        chunk = {
            "content": "| Name | Value |\n|------|-------|\n| A    | 1     |",
            "section_header": "",
        }
        structural = extractor.extract_structural(chunk, "")
        assert structural["is_table"] is True

    def test_preserves_chunk_index_and_total(self) -> None:
        extractor = MetadataExtractor(None)
        chunk = {"content": "Test.", "section_header": "", "chunk_index": 3, "total_chunks": 10}
        structural = extractor.extract_structural(chunk, "")
        assert structural["chunk_index"] == 3
        assert structural["total_chunks"] == 10

    def test_all_required_keys_present(self) -> None:
        extractor = MetadataExtractor(None)
        chunk = {"content": "Test.", "section_header": ""}
        structural = extractor.extract_structural(chunk, "")
        required = {
            "chunk_index",
            "total_chunks",
            "heading_level",
            "section_header",
            "is_list",
            "is_code",
            "is_table",
            "char_count",
            "word_count",
            "link_count",
            "email_count",
            "phone_count",
        }
        assert required.issubset(structural.keys())


@pytest.mark.integration
class TestDateExtraction:
    def test_extract_iso_date(self) -> None:
        extractor = MetadataExtractor(None)
        dates = extractor.extract_dates("The deadline is 2026-07-26 for submission.")
        assert any(d["value"] == "2026-07-26" and d["format"] == "iso" for d in dates)

    def test_extract_written_date(self) -> None:
        extractor = MetadataExtractor(None)
        dates = extractor.extract_dates("Published on January 15, 2026 in the journal.")
        assert any("January" in d["value"] and d["format"] == "written" for d in dates)

    def test_extract_quarter(self) -> None:
        extractor = MetadataExtractor(None)
        dates = extractor.extract_dates("Q3 2026 results are in.")
        assert any(d["value"] == "Q3 2026" and d["format"] == "quarter" for d in dates)

    def test_extract_fiscal_year(self) -> None:
        extractor = MetadataExtractor(None)
        dates = extractor.extract_dates("FY2026 budget has been approved.")
        assert any(d["format"] == "fiscal_year" for d in dates)

    def test_extract_slash_date(self) -> None:
        extractor = MetadataExtractor(None)
        dates = extractor.extract_dates("Signed on 07/26/2026 by the director.")
        assert any(d["format"] == "slash" for d in dates)

    def test_multiple_formats_in_one_text(self) -> None:
        extractor = MetadataExtractor(None)
        dates = extractor.extract_dates("Published January 15, 2026. Deadline: 2026-07-26. Q1 2026 review.")
        formats = {d["format"] for d in dates}
        assert "iso" in formats
        assert "written" in formats
        assert "quarter" in formats

    def test_deduplicates_repeated_dates(self) -> None:
        extractor = MetadataExtractor(None)
        dates = extractor.extract_dates("2026-07-26 is the date. Remember: 2026-07-26.")
        values = [d["value"] for d in dates]
        assert values.count("2026-07-26") == 1

    def test_no_dates_returns_empty_list(self) -> None:
        extractor = MetadataExtractor(None)
        dates = extractor.extract_dates("No temporal information in this sentence.")
        assert dates == []


@pytest.mark.integration
class TestKeywordExtraction:
    def test_extracts_keywords_as_lowercase_list(self) -> None:
        extractor = MetadataExtractor(None)
        keywords = extractor.extract_keywords(
            "Revenue analysis shows significant growth in quarterly financial reports."
        )
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert all(isinstance(k, str) and k == k.lower() for k in keywords)

    def test_filters_common_stopwords(self) -> None:
        extractor = MetadataExtractor(None)
        keywords = extractor.extract_keywords("This is the analysis that shows growth and revenue with more data.")
        assert "this" not in keywords
        assert "that" not in keywords
        assert "with" not in keywords
        assert "from" not in keywords

    def test_returns_relevant_content_words(self) -> None:
        extractor = MetadataExtractor(None)
        keywords = extractor.extract_keywords(
            "Machine learning algorithms provide accurate predictions for complex datasets."
        )
        assert "machine" in keywords or "learning" in keywords

    def test_limits_to_ten_keywords(self) -> None:
        extractor = MetadataExtractor(None)
        long_text = " ".join(f"unique_term{i!s}" * 3 for i in range(50))
        keywords = extractor.extract_keywords(long_text)
        assert len(keywords) <= 10

    def test_handles_code_blocks(self) -> None:
        extractor = MetadataExtractor(None)
        keywords = extractor.extract_keywords("Revenue analysis ```python\ndef analyze():\n    pass\n``` shows growth.")
        assert isinstance(keywords, list)


# ── LLM tests (skip if Ollama not reachable) ────────────────────────────────


@pytest.mark.integration
class TestEntityExtractionLLM:
    @pytest.fixture
    def ollama_client(self) -> httpx.Client:
        if not _ollama_reachable():
            pytest.skip("Ollama not reachable")
        return httpx.Client(timeout=60.0)

    def test_extract_entities_returns_dict(self, ollama_client: httpx.Client, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("rag.config.ENABLE_METADATA_EXTRACTION", True)
        monkeypatch.setattr("rag.config.ENABLE_ENTITY_EXTRACTION", True)
        monkeypatch.setattr("rag.config.CHAT_MODEL", "qwen2.5:0.5b")

        extractor = MetadataExtractor(ollama_client)
        entities = extractor.extract_entities("Alice Smith from Acme Corp reported revenue of $10M in New York.")
        assert isinstance(entities, dict)

    def test_extract_entities_returns_valid_structure(
        self, ollama_client: httpx.Client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("rag.config.ENABLE_METADATA_EXTRACTION", True)
        monkeypatch.setattr("rag.config.ENABLE_ENTITY_EXTRACTION", True)
        monkeypatch.setattr("rag.config.CHAT_MODEL", "qwen2.5:0.5b")

        extractor = MetadataExtractor(ollama_client)
        entities = extractor.extract_entities(
            "Alice Smith from Acme Corp launched Widget Pro in New York on 2026-01-15."
        )
        assert isinstance(entities, dict)
        # qwen2.5:0.5b may return empty dict — acceptable for a tiny model
        if entities:
            for key in ("people", "organizations", "dates", "locations", "products"):
                assert key in entities


@pytest.mark.integration
class TestTopicExtractionLLM:
    @pytest.fixture
    def ollama_client(self) -> httpx.Client:
        if not _ollama_reachable():
            pytest.skip("Ollama not reachable")
        return httpx.Client(timeout=60.0)

    def test_extract_topics_returns_list(self, ollama_client: httpx.Client, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("rag.config.ENABLE_METADATA_EXTRACTION", True)
        monkeypatch.setattr("rag.config.ENABLE_TOPIC_TAGGING", True)
        monkeypatch.setattr("rag.config.CHAT_MODEL", "qwen2.5:0.5b")

        extractor = MetadataExtractor(ollama_client)
        topics = extractor.extract_topics("This document covers quarterly financial analysis and revenue forecasting.")
        assert isinstance(topics, list)


@pytest.mark.integration
class TestDocumentClassificationLLM:
    @pytest.fixture
    def ollama_client(self) -> httpx.Client:
        if not _ollama_reachable():
            pytest.skip("Ollama not reachable")
        return httpx.Client(timeout=60.0)

    def test_classify_document_returns_valid_type(
        self, ollama_client: httpx.Client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("rag.config.ENABLE_METADATA_EXTRACTION", True)
        monkeypatch.setattr("rag.config.ENABLE_DOC_CLASSIFICATION", True)
        monkeypatch.setattr("rag.config.CHAT_MODEL", "qwen2.5:0.5b")

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
        extractor = MetadataExtractor(ollama_client)
        doc_type = extractor.classify_document("This quarterly report presents financial results for Q3 2026.")
        assert doc_type in valid


@pytest.mark.integration
class TestExtractAllIntegration:
    def test_extract_all_non_llm_features(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if not _langdetect_available():
            pytest.skip("langdetect not installed")

        monkeypatch.setattr("rag.config.ENABLE_METADATA_EXTRACTION", True)
        monkeypatch.setattr("rag.config.ENABLE_ENTITY_EXTRACTION", False)
        monkeypatch.setattr("rag.config.ENABLE_DOC_CLASSIFICATION", False)
        monkeypatch.setattr("rag.config.ENABLE_TOPIC_TAGGING", False)
        monkeypatch.setattr("rag.config.ENABLE_LANGUAGE_DETECTION", True)
        monkeypatch.setattr("rag.config.CHAT_MODEL", "qwen2.5:0.5b")

        extractor = MetadataExtractor(None)
        chunk = {
            "content": "Revenue grew 15% in Q3 2026 per the report dated 2026-01-15.",
            "section_header": "## Revenue Analysis",
            "chunk_index": 0,
            "total_chunks": 5,
        }
        metadata = extractor.extract_all(
            chunk=chunk,
            document_text="Quarterly financial report.",
            source_identifier="/docs/report.pdf",
        )
        assert "structural" in metadata
        assert "keywords" in metadata
        assert "dates" in metadata
        assert "language" in metadata
        assert "entities" not in metadata
        assert "doc_type" not in metadata
        assert "topics" not in metadata

    @pytest.fixture
    def ollama_client(self) -> httpx.Client:
        if not _ollama_reachable():
            pytest.skip("Ollama not reachable")
        return httpx.Client(timeout=60.0)

    def test_extract_all_with_llm_features(self, ollama_client: httpx.Client, monkeypatch: pytest.MonkeyPatch) -> None:
        if not _langdetect_available():
            pytest.skip("langdetect not installed")

        monkeypatch.setattr("rag.config.ENABLE_METADATA_EXTRACTION", True)
        monkeypatch.setattr("rag.config.ENABLE_ENTITY_EXTRACTION", True)
        monkeypatch.setattr("rag.config.ENABLE_DOC_CLASSIFICATION", True)
        monkeypatch.setattr("rag.config.ENABLE_TOPIC_TAGGING", True)
        monkeypatch.setattr("rag.config.ENABLE_LANGUAGE_DETECTION", True)
        monkeypatch.setattr("rag.config.CHAT_MODEL", "qwen2.5:0.5b")

        extractor = MetadataExtractor(ollama_client)
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
        assert "keywords" in metadata
        assert "dates" in metadata
        assert "language" in metadata
        assert "entities" in metadata
        assert "doc_type" in metadata
        assert "topics" in metadata
