"""Metadata extraction: entities, classification, topics, language, structure.

Extracts rich metadata from document chunks during ingestion. All extractors
are optional and gated behind feature flags in ``config``. Failures are
gracefully handled — metadata is never required for ingestion to succeed.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from memex.engine.core import config
from memex.engine.llm.base import LLMProvider

logger = logging.getLogger("metadata-extractor")

_DOC_TYPES = {
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
}

# Version of the metadata extraction schema/prompts. Bump whenever the
# extraction logic changes — stored chunks with an older (or missing)
# metadata_version are re-ingested on the next sync, so new fields/prompts
# actually reach the collection.
METADATA_VERSION = 3

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_URL_RE = re.compile(r"https?://[^\s<>\"']+")
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}")


class MetadataExtractor:
    """Extracts rich metadata from document chunks.

    Each extraction method is independent. Failures are logged and return
    empty/default values so ingestion is never blocked.
    """

    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        self._llm = llm_provider
        self._lang_detector: Any = None

    # ── Public API ────────────────────────────────────────────────────────

    def extract_all(
        self,
        chunk: dict[str, Any],
        document_text: str = "",
        source_identifier: str = "",
    ) -> dict[str, Any]:
        """Extract all configured metadata from a chunk.

        Returns a flat dict whose keys are stored directly in the Qdrant
        payload. Disabled extractors are skipped silently.
        """
        metadata: dict[str, Any] = {}

        if config.ENABLE_ENTITY_EXTRACTION:
            metadata["entities"] = self.extract_entities(chunk["content"])

        if config.ENABLE_DOC_CLASSIFICATION and chunk.get("chunk_index", 0) == 0:
            metadata["doc_type"] = self.classify_document(
                document_text or chunk["content"],
            )

        if config.ENABLE_TOPIC_TAGGING:
            metadata["topics"] = self.extract_topics(chunk["content"])

        if config.ENABLE_LANGUAGE_DETECTION:
            lang = self.detect_language(chunk["content"])
            if lang:
                metadata["language"] = lang

        keywords = self.extract_keywords(chunk["content"])
        if keywords:
            metadata["keywords"] = keywords

        metadata["structural"] = self.extract_structural(chunk, source_identifier)

        return metadata

    # ── Entity extraction ─────────────────────────────────────────────────

    def extract_entities(self, text: str) -> dict[str, list[str]]:
        """Extract named entities from text using LLM.

        Returns dict with keys: people, organizations, locations, products, dates.
        Each value is a list of unique strings (capped by config).
        """
        if not self._llm:
            logger.debug("No LLM provider, skipping entity extraction")
            return {}

        prompt = (
            "You are a precise named-entity extractor.\n"
            "Return a JSON object with exactly these keys:\n"
            '- "people": full names of people mentioned\n'
            '- "organizations": company, agency, or institution names\n'
            '- "locations": cities, countries, addresses, or place names\n'
            '- "products": product names, model numbers, or brand names\n'
            '- "dates": all dates mentioned, as readable strings (e.g. "January 15, 2026", '
            '"Q3 2025", "2024"). Normalize partial dates (e.g. "Jan" -> "January"). '
            "Deduplicate: if the same date appears multiple ways, keep the most complete form.\n"
            "Each value is a list of unique strings. Use empty arrays when there are no "
            "matches — never omit keys. Raw JSON only, no markdown fences.\n\n"
            f"Text: {text[:1000]}"
        )
        try:
            response = self._chat(prompt)
            entities = json.loads(self._strip_code_fences(response))
            return {k: v[: config.MAX_ENTITIES_PER_CHUNK] for k, v in entities.items() if isinstance(v, list)}
        except (json.JSONDecodeError, Exception) as exc:
            logger.debug("Entity extraction failed: %s", exc)
            return {}

    # ── Document classification ───────────────────────────────────────────

    def classify_document(self, text: str) -> str:
        """Classify document type using LLM.

        Returns one of: report, email, article, code, documentation,
        presentation, resume, contract, invoice, meeting_notes, other, unknown.
        """
        if not self._llm:
            return "unknown"

        prompt = (
            "You are a document classifier.\n"
            "Classify the document type. Choose exactly one of: "
            "report, email, article, code, documentation, presentation, "
            "resume, contract, invoice, meeting_notes, other.\n"
            "Prefer 'contract' for legal agreements, deeds, trusts, and signed "
            "documents with clauses and parties. Prefer 'report' for structured "
            "analyses with findings. Output the type only — no punctuation, no quotes.\n\n"
            f"Text: {text[:2000]}"
        )
        try:
            doc_type = self._chat(prompt).strip().lower()
            if doc_type in _DOC_TYPES:
                return doc_type
            for v in _DOC_TYPES:
                if v in doc_type:
                    return v
            return "other"
        except Exception as exc:
            logger.debug("Document classification failed: %s", exc)
            return "unknown"

    # ── Topic extraction ──────────────────────────────────────────────────

    def extract_topics(self, text: str) -> list[str]:
        """Extract topic labels from text using LLM.

        Returns a list of up to ``MAX_TOPICS_PER_CHUNK`` topic strings.
        """
        if not self._llm:
            return []

        prompt = (
            f"You are a topic tagger. Extract up to {config.MAX_TOPICS_PER_CHUNK} topic labels "
            "from this text. Topics must be short noun phrases (2-4 words) that describe the "
            "main subjects. Avoid generic labels like 'information' or 'details'. "
            "Return a JSON array of strings. Use [] if no clear topics. "
            "Raw JSON only, no markdown fences.\n\n"
            f"Text: {text[:1000]}"
        )
        try:
            response = self._chat(prompt)
            topics = json.loads(self._strip_code_fences(response))
            if isinstance(topics, list):
                return [str(t) for t in topics[: config.MAX_TOPICS_PER_CHUNK]]
            return []
        except (json.JSONDecodeError, Exception) as exc:
            logger.debug("Topic extraction failed: %s", exc)
            return []

    # ── Language detection ────────────────────────────────────────────────

    def detect_language(self, text: str) -> str:
        """Detect text language. Returns ISO 639-1 code or empty string."""
        if len(text.strip()) < 20:
            return ""
        try:
            if self._lang_detector is None:
                from langdetect import detect

                self._lang_detector = detect
            result = self._lang_detector(text[:500])
            return result if isinstance(result, str) else ""
        except Exception:
            return ""

    # ── Keyword extraction ────────────────────────────────────────────────

    def extract_keywords(self, text: str) -> list[str]:
        """Extract keywords using simple TF heuristics.

        Returns top keywords (lowercase, deduplicated, length > 3 chars).
        """
        cleaned = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
        cleaned = _URL_RE.sub(" ", cleaned)
        cleaned = _EMAIL_RE.sub(" ", cleaned)

        words = re.findall(r"\b[a-zA-Z]{4,}\b", cleaned.lower())

        stopwords = {
            "this",
            "that",
            "with",
            "from",
            "have",
            "been",
            "were",
            "they",
            "their",
            "which",
            "about",
            "would",
            "could",
            "should",
            "there",
            "also",
            "more",
            "than",
            "some",
            "only",
            "into",
            "over",
            "such",
            "very",
            "does",
            "will",
            "each",
            "made",
            "when",
            "what",
            "your",
            "then",
            "them",
            "other",
            "most",
            "can",
            "but",
            "not",
            "for",
            "the",
            "and",
            "are",
            "was",
            "one",
            "our",
            "out",
            "all",
            "its",
            "use",
            "may",
            "how",
            "any",
        }
        freq: dict[str, int] = {}
        for w in words:
            if w not in stopwords:
                freq[w] = freq.get(w, 0) + 1

        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:10]]

    # ── Structural metadata ───────────────────────────────────────────────

    def extract_structural(
        self,
        chunk: dict[str, Any],
        source_identifier: str,
    ) -> dict[str, Any]:
        """Extract structural metadata from chunk position and content."""
        content = chunk.get("content", "")
        header = chunk.get("section_header", "")

        heading_level = 0
        if header:
            match = re.match(r"^(#{1,6})\s+", header)
            if match:
                heading_level = len(match.group(1))

        is_list = bool(re.match(r"^\s*[-*\d]+[.)]\s", content))
        is_code = "```" in content or content.startswith("    ")
        is_table = "|" in content and content.count("|") >= 3

        links = _URL_RE.findall(content)
        emails = _EMAIL_RE.findall(content)
        phones = _PHONE_RE.findall(content)

        return {
            "chunk_index": chunk.get("chunk_index", 0),
            "total_chunks": chunk.get("total_chunks", 0),
            "heading_level": heading_level,
            "section_header": header,
            "is_list": is_list,
            "is_code": is_code,
            "is_table": is_table,
            "char_count": len(content),
            "word_count": len(content.split()),
            "link_count": len(links),
            "email_count": len(emails),
            "phone_count": len(phones),
        }

    # ── Batch extraction ──────────────────────────────────────────────────

    def extract_batch(
        self,
        chunks: list[dict[str, Any]],
        document_text: str = "",
        source_identifier: str = "",
        batch_size: int = 4,
    ) -> list[dict[str, Any]]:
        """Extract metadata for multiple chunks in batches to reduce LLM calls.

        Batch size 4: small models (qwen2.5:1.5b) cannot reliably emit a
        JSON array of 10 objects — they truncate after 1-2 and the rest are
        silently lost. 4 objects per call stays well within output limits.

        Combines entity extraction and topic tagging into a single prompt per batch.
        Returns a list of metadata dicts, one per chunk.
        """
        if not chunks:
            return []

        if len(chunks) <= batch_size:
            return self._extract_batch_metadata(
                chunks,
                document_text,
                source_identifier,
                doc_type=True,
                batch_start=0,
                total_chunks=len(chunks),
            )

        results: list[dict[str, Any]] = [{} for _ in range(len(chunks))]
        doc_type_found = False

        batches = []
        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start : batch_start + batch_size]
            batches.append((batch, batch_start))

        for batch, batch_start in batches:
            need_doc_type = not doc_type_found
            batch_meta = self._extract_batch_metadata(
                batch,
                document_text,
                source_identifier,
                doc_type=need_doc_type,
                batch_start=batch_start,
                total_chunks=len(chunks),
            )
            if need_doc_type and batch_meta and batch_meta[0].get("doc_type"):
                doc_type_found = True
            for i, meta in enumerate(batch_meta):
                results[batch_start + i] = meta

        return results

    def _extract_batch_metadata(
        self,
        batch: list[dict[str, Any]],
        document_text: str,
        source_identifier: str,
        doc_type: bool = True,
        batch_start: int = 0,
        total_chunks: int = 0,
    ) -> list[dict[str, Any]]:
        """Extract metadata for a batch of chunks in a single LLM call."""
        if not self._llm:
            return [{} for _ in batch]

        chunks_text = "\n\n".join(f"[Chunk {i}]: {c['content'][:500]}" for i, c in enumerate(batch))

        tasks = []
        if config.ENABLE_ENTITY_EXTRACTION:
            tasks.append(
                "entities (JSON object with keys: people (full names), "
                "organizations (companies/agencies), locations (places), "
                "products (product names), dates (readable date strings like "
                '"January 15, 2026" or "Q3 2025" — normalize and deduplicate). '
                "Each value is a list of unique strings."
            )
        if config.ENABLE_TOPIC_TAGGING:
            tasks.append(f"topics (JSON array of up to {config.MAX_TOPICS_PER_CHUNK} topic labels)")
        if doc_type and config.ENABLE_DOC_CLASSIFICATION:
            tasks.append(
                "doc_type ("
                "report, email, article, code, documentation, presentation, "
                "resume, contract, invoice, meeting_notes, other)"
            )

        if not tasks:
            return [{} for _ in batch]

        prompt = (
            "You are a precise metadata extraction engine for a document corpus.\n"
            "Extract metadata from each chunk below.\n\n"
            "OUTPUT CONTRACT (strict):\n"
            "- Return exactly ONE JSON array with one object per chunk, in the same order.\n"
            f"- Every object must contain ALL keys: {', '.join(tasks)}.\n"
            "- Use empty arrays for fields with no matches — never omit a key.\n"
            "- Never add keys beyond the requested ones.\n"
            "- Output raw JSON only — no markdown fences, no commentary.\n\n"
            "Example for a single chunk:\n"
            '{"entities": {"people": ["Jane Doe"], "organizations": ["Acme Corp"], '
            '"locations": ["Mumbai"], "products": [], "dates": ["January 15, 2026"]}, '
            '"topics": ["corporate governance"], "doc_type": "contract"}\n\n'
            f"Chunks:\n{chunks_text}"
        )

        try:
            # 1200 tokens: a batch of chunks with entities+topics JSON needs
            # far more than 400 — truncation caused JSON parse failures and
            # 10x slower per-chunk fallback calls.
            response = self._chat(prompt, num_predict=1200)
            parsed = json.loads(self._strip_code_fences(response))
            if not isinstance(parsed, list):
                parsed = [parsed]
            if len(parsed) < len(batch):
                # Model truncated the array — padding would silently drop
                # metadata for the missing chunks. Force per-chunk instead.
                raise json.JSONDecodeError(
                    f"short array: {len(parsed)}/{len(batch)} objects", response, 0
                )
            while len(parsed) < len(batch):
                parsed.append({})
            normalized = [self._normalize_metadata(m) for m in parsed[: len(batch)]]
            for i, (chunk, meta) in enumerate(zip(batch, normalized, strict=True)):
                chunk_with_index = {**chunk, "chunk_index": batch_start + i, "total_chunks": total_chunks}
                if config.ENABLE_LANGUAGE_DETECTION:
                    lang = self.detect_language(chunk["content"])
                    if lang:
                        meta["language"] = lang
                meta["keywords"] = self.extract_keywords(chunk["content"])
                meta["structural"] = self.extract_structural(chunk_with_index, source_identifier)
            return normalized
        except (json.JSONDecodeError, Exception) as exc:
            logger.debug("Batch metadata extraction failed, falling back to per-chunk: %s", exc)
            return self._fallback_per_chunk(
                batch, document_text, source_identifier, doc_type, batch_start, total_chunks
            )

    def _fallback_per_chunk(
        self,
        batch: list[dict[str, Any]],
        document_text: str,
        source_identifier: str,
        doc_type: bool,
        batch_start: int = 0,
        total_chunks: int = 0,
    ) -> list[dict[str, Any]]:
        """Fallback to per-chunk extraction when batch fails."""
        results = []
        for i, chunk in enumerate(batch):
            meta: dict[str, Any] = {}
            chunk_with_index = {**chunk, "chunk_index": batch_start + i, "total_chunks": total_chunks}
            if config.ENABLE_ENTITY_EXTRACTION:
                meta["entities"] = self.extract_entities(chunk["content"])
            if doc_type and config.ENABLE_DOC_CLASSIFICATION and i == 0:
                meta["doc_type"] = self.classify_document(document_text or chunk["content"])
            if config.ENABLE_TOPIC_TAGGING:
                meta["topics"] = self.extract_topics(chunk["content"])
            if config.ENABLE_LANGUAGE_DETECTION:
                lang = self.detect_language(chunk["content"])
                if lang:
                    meta["language"] = lang
            meta["keywords"] = self.extract_keywords(chunk["content"])
            meta["structural"] = self.extract_structural(chunk_with_index, source_identifier)
            results.append(meta)
        return results

    def _normalize_metadata(self, meta: dict[str, Any]) -> dict[str, Any]:
        """Normalize LLM output to expected metadata format."""
        result: dict[str, Any] = {}
        if "entities" in meta and isinstance(meta["entities"], dict):
            result["entities"] = {
                k: v[: config.MAX_ENTITIES_PER_CHUNK] for k, v in meta["entities"].items() if isinstance(v, list)
            }
        if "topics" in meta and isinstance(meta["topics"], list):
            result["topics"] = [str(t) for t in meta["topics"][: config.MAX_TOPICS_PER_CHUNK]]
        if "doc_type" in meta:
            doc_type = str(meta["doc_type"]).strip().lower()
            result["doc_type"] = doc_type if doc_type in _DOC_TYPES else "other"
        return result

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove markdown code fences (```json ... ```) from LLM output."""
        return re.sub(r"```(?:json)?\s*\n?(.*?)\n?\s*```", r"\1", text, flags=re.DOTALL).strip()

    def _chat(self, prompt: str, num_predict: int = 200) -> str:
        """Call LLM provider synchronously."""
        if not self._llm:
            return ""
        model = config.METADATA_MODEL or config.CHAT_MODEL
        return self._llm.chat_sync(prompt, model=model, num_predict=num_predict)
