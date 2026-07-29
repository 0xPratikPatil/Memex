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

import httpx

from rag import config

logger = logging.getLogger("metadata-extractor")

# ── Rule-based date patterns ──────────────────────────────────────────────────

_DATE_PATTERNS: list[tuple[str, str]] = [
    (r"\b\d{4}-\d{2}-\d{2}\b", "iso"),
    (r"\b\d{1,2}/\d{1,2}/\d{4}\b", "slash"),
    (r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4}\b", "written"),
    (r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{4}\b", "written_alt"),
    (r"\b(?:Q[1-4])\s+\d{4}\b", "quarter"),
    (r"\b(?:FY|fy)\s*\d{4}\b", "fiscal_year"),
]

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_URL_RE = re.compile(r"https?://[^\s<>\"']+")
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}")


class MetadataExtractor:
    """Extracts rich metadata from document chunks.

    Each extraction method is independent. Failures are logged and return
    empty/default values so ingestion is never blocked.
    """

    def __init__(self, ollama_client: httpx.Client | None = None) -> None:
        self._ollama = ollama_client
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

        # Temporal metadata (always run — cheap regex)
        dates = self.extract_dates(chunk["content"])
        if dates:
            metadata["dates"] = dates

        # Keywords (always run — cheap regex)
        keywords = self.extract_keywords(chunk["content"])
        if keywords:
            metadata["keywords"] = keywords

        # Structural metadata (always run — no external calls)
        metadata["structural"] = self.extract_structural(chunk, source_identifier)

        return metadata

    # ── Entity extraction ─────────────────────────────────────────────────

    def extract_entities(self, text: str) -> dict[str, list[str]]:
        """Extract named entities from text using LLM.

        Returns dict with keys: people, organizations, dates, locations, products.
        Each value is a list of unique strings (capped by config).
        """
        if not self._ollama:
            logger.debug("No Ollama client, skipping entity extraction")
            return {}

        prompt = (
            "Extract named entities from this text. Return JSON with keys: "
            "people, organizations, dates, locations, products. "
            "Each value is a list of unique strings. Only output JSON.\n\n"
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
        if not self._ollama:
            return "unknown"

        prompt = (
            "Classify this document into one of these types: "
            "report, email, article, code, documentation, presentation, "
            "resume, contract, invoice, meeting_notes, other. "
            "Only output the type.\n\n"
            f"Text: {text[:2000]}"
        )
        try:
            doc_type = self._chat(prompt).strip().lower()
            # Normalize: extract first word if LLM returns extra text
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
            }
            if doc_type in valid:
                return doc_type
            # Try partial match
            for v in valid:
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
        if not self._ollama:
            return []

        prompt = (
            f"Extract up to {config.MAX_TOPICS_PER_CHUNK} topic labels from this text. "
            "Return as JSON array of strings. Only output JSON.\n\n"
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

    # ── Temporal extraction ───────────────────────────────────────────────

    def extract_dates(self, text: str) -> list[dict[str, str]]:
        """Extract dates from text using regex patterns.

        Returns list of {"value": "...", "format": "..."} dicts.
        """
        dates: list[dict[str, str]] = []
        seen: set[str] = set()
        for pattern, fmt in _DATE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                val = match.group(0)
                if val not in seen:
                    seen.add(val)
                    dates.append({"value": val, "format": fmt})
        return dates

    # ── Keyword extraction ────────────────────────────────────────────────

    def extract_keywords(self, text: str) -> list[str]:
        """Extract keywords using simple TF heuristics.

        Returns top keywords (lowercase, deduplicated, length > 3 chars).
        """
        # Remove code blocks and URLs for keyword extraction
        cleaned = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
        cleaned = _URL_RE.sub(" ", cleaned)
        cleaned = _EMAIL_RE.sub(" ", cleaned)

        words = re.findall(r"\b[a-zA-Z]{4,}\b", cleaned.lower())

        # Simple frequency filter: skip very common words
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

        # Return top 10 by frequency
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

        # Parse heading level from markdown header
        heading_level = 0
        if header:
            match = re.match(r"^(#{1,6})\s+", header)
            if match:
                heading_level = len(match.group(1))

        # Detect content types via regex
        is_list = bool(re.match(r"^\s*[-*\d]+[.)]\s", content))
        is_code = "```" in content or content.startswith("    ")
        is_table = "|" in content and content.count("|") >= 3

        # Count links and emails
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
        batch_size: int = 10,
    ) -> list[dict[str, Any]]:
        """Extract metadata for multiple chunks in batches to reduce LLM calls.

        Combines entity extraction and topic tagging into a single prompt per batch.
        Returns a list of metadata dicts, one per chunk.
        """
        if not chunks:
            return []

        # For small documents (<=batch_size), single LLM call
        if len(chunks) <= batch_size:
            return self._extract_batch_metadata(
                chunks,
                document_text,
                source_identifier,
                doc_type=True,
                batch_start=0,
            )

        # For larger documents, process in parallel batches
        results: list[dict[str, Any]] = [{} for _ in range(len(chunks))]
        doc_type_found = False

        batches = []
        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start : batch_start + batch_size]
            batches.append((batch, batch_start))

        # Process batches sequentially to maintain doc_type consistency
        # (first batch extracts doc_type, subsequent batches skip it)
        for batch, batch_start in batches:
            need_doc_type = not doc_type_found
            batch_meta = self._extract_batch_metadata(
                batch,
                document_text,
                source_identifier,
                doc_type=need_doc_type,
                batch_start=batch_start,
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
    ) -> list[dict[str, Any]]:
        """Extract metadata for a batch of chunks in a single LLM call."""
        if not self._ollama:
            return [{} for _ in batch]

        chunks_text = "\n\n".join(f"[Chunk {i}]: {c['content'][:500]}" for i, c in enumerate(batch))

        tasks = []
        if config.ENABLE_ENTITY_EXTRACTION:
            tasks.append("entities (JSON with keys: people, orgs, dates, locations, products — each a list)")
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
            f"Extract metadata from each chunk below. For each chunk, return a JSON object "
            f"with keys matching these tasks: {'; '.join(tasks)}.\n\n"
            f"Return a JSON array of objects, one per chunk, in order. "
            f"Only output JSON.\n\n"
            f"Chunks:\n{chunks_text}"
        )

        try:
            response = self._chat(prompt, num_predict=400)
            parsed = json.loads(self._strip_code_fences(response))
            if not isinstance(parsed, list):
                parsed = [parsed]
            # Pad/truncate to match batch size
            while len(parsed) < len(batch):
                parsed.append({})
            normalized = [self._normalize_metadata(m) for m in parsed[: len(batch)]]
            # Add cheap regex-based metadata per chunk
            for i, (chunk, meta) in enumerate(zip(batch, normalized, strict=True)):
                chunk_with_index = {**chunk, "chunk_index": batch_start + i}
                if config.ENABLE_LANGUAGE_DETECTION:
                    lang = self.detect_language(chunk["content"])
                    if lang:
                        meta["language"] = lang
                meta["dates"] = self.extract_dates(chunk["content"])
                meta["keywords"] = self.extract_keywords(chunk["content"])
                meta["structural"] = self.extract_structural(chunk_with_index, source_identifier)
            return normalized
        except (json.JSONDecodeError, Exception) as exc:
            logger.debug("Batch metadata extraction failed, falling back to per-chunk: %s", exc)
            return self._fallback_per_chunk(batch, document_text, source_identifier, doc_type)

    def _fallback_per_chunk(
        self,
        batch: list[dict[str, Any]],
        document_text: str,
        source_identifier: str,
        doc_type: bool,
    ) -> list[dict[str, Any]]:
        """Fallback to per-chunk extraction when batch fails."""
        results = []
        for i, chunk in enumerate(batch):
            meta: dict[str, Any] = {}
            chunk_with_index = {**chunk, "chunk_index": i}
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
            meta["dates"] = self.extract_dates(chunk["content"])
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
            result["doc_type"] = str(meta["doc_type"]).strip().lower()
        return result

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove markdown code fences (```json ... ```) from LLM output."""
        return re.sub(r"```(?:json)?\s*\n?(.*?)\n?\s*```", r"\1", text, flags=re.DOTALL).strip()

    def _chat(self, prompt: str, num_predict: int = 200) -> str:
        """Call Ollama chat API via shared helper."""
        from rag.ollama_chat import ollama_chat

        model = config.METADATA_MODEL or config.CHAT_MODEL
        return ollama_chat(prompt, model=model, num_predict=num_predict, ollama_client=self._ollama)
