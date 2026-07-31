"""Unit tests for config module defaults."""

from __future__ import annotations

import os
from importlib import reload
from unittest.mock import patch

from memex.engine.core import config


class TestChunkingDefaults:
    def test_chunk_size_default_1024(self):
        reload(config)
        assert config.CHUNK_SIZE == 1024

    def test_chunk_overlap_default_128(self):
        reload(config)
        assert config.CHUNK_OVERLAP == 128

    def test_chunk_strategy_default_hybrid(self):
        reload(config)
        assert config.CHUNK_STRATEGY == "hybrid"

    def test_chunk_merge_peers_default_true(self):
        reload(config)
        assert config.CHUNK_MERGE_PEERS is True


class TestQueryExpansionDefaults:
    def test_enable_query_expansion_true(self):
        reload(config)
        assert config.ENABLE_QUERY_EXPANSION is True

    def test_enable_hyde_true(self):
        reload(config)
        assert config.ENABLE_HYDE is True

    def test_enable_multi_query_true(self):
        reload(config)
        assert config.ENABLE_MULTI_QUERY is True

    def test_enable_query_rewrite_true(self):
        reload(config)
        assert config.ENABLE_QUERY_REWRITE is True


class TestCacheDefaults:
    def test_enable_cache_true(self):
        reload(config)
        assert config.ENABLE_CACHE is True


class TestSearchDefaults:
    def test_search_top_k_default_30(self):
        reload(config)
        assert config.SEARCH_TOP_K == 30


class TestEmbeddingDefaults:
    def test_embed_batch_size_default_64(self):
        reload(config)
        assert config.EMBED_BATCH_SIZE == 64


class TestContextualRetrievalDefaults:
    def test_enabled_by_default(self):
        with patch.dict(os.environ, {"ENABLE_CONTEXTUAL_RETRIEVAL": "true"}, clear=False):
            reload(config)
            assert config.ENABLE_CONTEXTUAL_RETRIEVAL is True

    def test_strategy_default_summary(self):
        reload(config)
        assert config.CONTEXT_STRATEGY == "summary"


class TestMetadataDefaults:
    def test_metadata_extraction_enabled(self):
        with patch.dict(os.environ, {"ENABLE_METADATA_EXTRACTION": "true"}, clear=False):
            reload(config)
            assert config.ENABLE_METADATA_EXTRACTION is True

    def test_entity_extraction_enabled(self):
        with patch.dict(os.environ, {"ENABLE_ENTITY_EXTRACTION": "true"}, clear=False):
            reload(config)
            assert config.ENABLE_ENTITY_EXTRACTION is True

    def test_doc_classification_enabled(self):
        with patch.dict(os.environ, {"ENABLE_DOC_CLASSIFICATION": "true"}, clear=False):
            reload(config)
            assert config.ENABLE_DOC_CLASSIFICATION is True

    def test_topic_tagging_enabled(self):
        with patch.dict(os.environ, {"ENABLE_TOPIC_TAGGING": "true"}, clear=False):
            reload(config)
            assert config.ENABLE_TOPIC_TAGGING is True


class TestDoclingEnrichmentDefaults:
    def test_enrich_code_default_false(self):
        reload(config)
        assert config.DOCLING_ENRICH_CODE is False

    def test_enrich_formula_default_false(self):
        reload(config)
        assert config.DOCLING_ENRICH_FORMULA is False

    def test_picture_classify_default_true(self):
        with patch.dict(os.environ, {"DOCLING_PICTURE_CLASSIFY": "true"}, clear=False):
            reload(config)
            assert config.DOCLING_PICTURE_CLASSIFY is True

    def test_chart_extract_default_false(self):
        reload(config)
        assert config.DOCLING_CHART_EXTRACT is False

    def test_image_export_default_embedded(self):
        reload(config)
        assert config.DOCLING_IMAGE_EXPORT == "embedded"

    def test_pdf_backend_default_empty(self, monkeypatch):
        monkeypatch.setenv("DOCLING_PDF_BACKEND", "")
        reload(config)
        assert config.DOCLING_PDF_BACKEND == ""
