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
            logger.warning("No LLM provider — entity extraction skipped")
            return {}

        prompt = (
            "You are a named-entity extractor for a document search index.\n"
            "Extract entities from the text and return a JSON object.\n\n"
            "SCHEMA (exactly these keys, each a list of strings):\n"
            '  "people": full names of people (not titles or pronouns)\n'
            '  "organizations": companies, agencies, institutions, teams\n'
            '  "locations": cities, countries, regions, addresses\n'
            '  "products": product names, model numbers, brands\n'
            '  "dates": dates or date ranges, normalized (e.g. "2026-01-15", "Q3 2025")\n\n'
            "RULES:\n"
            "- Return ONLY the JSON object — no preamble, no markdown fences, no comments\n"
            "- Use empty arrays [] when a category has no matches — never omit a key\n"
            "- Deduplicate: same entity appears once, keep the most complete form\n"
            "- Keep the original case of proper nouns (\"Acme Corp\", not \"acme corp\")\n"
            "- Do not invent entities that are not explicitly in the text\n\n"
            "EXAMPLE:\n"
            'Text: "The contract with Tesla Motors was signed in Berlin by Anna Müller on March 3."\n'
            'Output: {"people": ["Anna Müller"], "organizations": ["Tesla Motors"], '
            '"locations": ["Berlin"], "products": [], "dates": ["March 3"]}\n\n'
            f"Text: {text[:1000]}\n\n"
            "Output:"
        )
        try:
            response = self._chat(prompt)
            entities = self._extract_json(response)
            if not isinstance(entities, dict):
                raise json.JSONDecodeError("not a JSON object", response, 0)
            return {k: v[: config.MAX_ENTITIES_PER_CHUNK] for k, v in entities.items() if isinstance(v, list)}
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Entity extraction failed: %s", exc)
            return {}

    # ── Document classification ───────────────────────────────────────────

    def classify_document(self, text: str) -> str:
        """Classify document type using LLM.

        Returns one of: report, email, article, code, documentation,
        presentation, resume, contract, invoice, meeting_notes, other, unknown.
        """
        if not self._llm:
            logger.warning("No LLM provider — document classification skipped")
            return "unknown"

        prompt = (
            "You are a document classifier for a search index.\n"
            "Classify the text into exactly one of these types:\n"
            "report, email, article, code, documentation, presentation, "
            "resume, contract, invoice, meeting_notes, other\n\n"
            "GUIDANCE:\n"
            '- "contract" — legal agreements, deeds, trusts, signed documents with clauses and parties\n'
            '- "report" — structured analyses with findings, stats, recommendations\n'
            '- "email" — messages with greeting/salutation and sign-off\n'
            '- "invoice" — billing with amounts, line items, due dates\n'
            '- "meeting_notes" — minutes, agenda, action items, attendee list\n'
            '- "code" — source code, configs, scripts\n'
            '- "documentation" — API refs, user guides, READMEs\n'
            '- "presentation" — slide-style content, bullet-heavy with titles\n'
            '- "resume" — CV, skills, work history, education\n'
            '- "article" — prose with headline, byline, or journalistic tone\n'
            '- "other" — anything that does not fit the above\n\n'
            "Return ONLY the type name — one word, lowercase, no quotes, no punctuation, "
            "no explanations.\n\n"
            "EXAMPLE:\n"
            'Text: "This Agreement is entered into by Acme Corp and Jane Doe."\n'
            'Output: contract\n\n'
            f"Text: {text[:2000]}\n\n"
            "Output:"
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
            logger.warning("Document classification failed: %s", exc)
            return "unknown"

    # ── Topic extraction ──────────────────────────────────────────────────

    def extract_topics(self, text: str) -> list[str]:
        """Extract topic labels from text using LLM.

        Returns a list of up to ``MAX_TOPICS_PER_CHUNK`` topic strings.
        """
        if not self._llm:
            logger.warning("No LLM provider — topic extraction skipped")
            return []

        prompt = (
            "You are a topic tagger for a document search index.\n"
            f"Extract up to {config.MAX_TOPICS_PER_CHUNK} topic labels from the text.\n\n"
            "RULES:\n"
            "- Topics must be short noun phrases (2-4 words) describing the main subjects\n"
            "- Broad, generic labels are useless — be specific (\"machine learning\" not \"technology\")\n"
            "- Avoid: information, details, data, content, general, overview, introduction\n"
            "- Return ONLY a JSON array of strings — no markdown fences, no keys, no comments\n"
            "- Use [] when the text has no clear topics\n\n"
            "EXAMPLE:\n"
            'Text: "The 2025 annual report shows revenue grew 20% in the enterprise segment, driven by cloud adoption in Europe."\n'
            'Output: ["annual report", "revenue growth", "cloud adoption", "enterprise segment"]\n\n'
            f"Text: {text[:1000]}\n\n"
            "Output:"
        )
        try:
            response = self._chat(prompt)
            topics = self._extract_json(response)
            if isinstance(topics, list):
                return [str(t) for t in topics[: config.MAX_TOPICS_PER_CHUNK]]
            return []
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Topic extraction failed: %s", exc)
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
            logger.warning(
                "No LLM provider — metadata extraction skipped for %d chunks",
                len(batch),
            )
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
            "You are a metadata extraction engine for a document search index.\n"
            f"Extract metadata from each of the {len(batch)} chunks below.\n\n"
            "TASK PER CHUNK:\n"
            f"- {', '.join(tasks)}\n\n"
            "OUTPUT CONTRACT (strict):\n"
            "- Return ONE JSON array with exactly one object per chunk, in the SAME order as the input\n"
            f"- Every object must contain ALL these keys: {', '.join(tasks)}\n"
            "- Use empty arrays for fields with no matches — never omit a key\n"
            "- Never add keys beyond the requested ones\n"
            "- Output raw JSON only — no markdown fences, no commentary, no numbering\n\n"
            "EXAMPLE (for 2 chunks, entities + topics enabled):\n"
            'Input: [Chunk 0]: "The board approved the merger with GlobalTech." [Chunk 1]: "Revenue grew 15% this quarter."\n'
            'Output: [{"entities": {"people": [], "organizations": ["GlobalTech"], "locations": [], '
            '"products": [], "dates": []}, "topics": ["board approval", "merger"]}, '
            '{"entities": {"people": [], "organizations": [], "locations": [], "products": [], "dates": []}, '
            '"topics": ["revenue growth"]}]\n\n'
            f"Chunks:\n{chunks_text}\n\n"
            "Output:"
        )

        try:
            # 1200 tokens: a batch of chunks with entities+topics JSON needs
            # far more than 400 — truncation caused JSON parse failures and
            # 10x slower per-chunk fallback calls.
            response = self._chat(prompt, num_predict=1200)
            parsed = self._extract_json(response)
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
            logger.warning("Batch metadata extraction failed, falling back to per-chunk: %s", exc)
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
        """Fallback to per-chunk extraction when batch fails.

        Extracts ALL metadata fields for each chunk: entities, doc_type,
        topics, language, keywords, structural. Each extraction is independent
        — failures in one field don't affect others.
        """
        results = []
        for i, chunk in enumerate(batch):
            meta: dict[str, Any] = {}
            chunk_with_index = {**chunk, "chunk_index": batch_start + i, "total_chunks": total_chunks}

            # Entities (LLM-based)
            if config.ENABLE_ENTITY_EXTRACTION:
                meta["entities"] = self.extract_entities(chunk["content"])

            # Document type (LLM-based, only for first chunk in batch)
            if doc_type and config.ENABLE_DOC_CLASSIFICATION and i == 0:
                meta["doc_type"] = self.classify_document(document_text or chunk["content"])

            # Topics (LLM-based)
            if config.ENABLE_TOPIC_TAGGING:
                meta["topics"] = self.extract_topics(chunk["content"])

            # Language detection (langdetect library, no LLM)
            if config.ENABLE_LANGUAGE_DETECTION:
                lang = self.detect_language(chunk["content"])
                if lang:
                    meta["language"] = lang

            # Keywords (TF heuristics, no LLM)
            meta["keywords"] = self.extract_keywords(chunk["content"])

            # Structural metadata (heuristics, no LLM)
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

    @staticmethod
    def _extract_json(text: str) -> Any:
        """Parse JSON out of noisy LLM output.

        Tries full-string parse first, then scans for the first balanced
        ``{...}`` object or ``[...]`` array — models often wrap JSON in
        prose ("Here is the result: ...") or trail with comments.
        """
        import json as _json

        if not text:
            return None
        cleaned = MetadataExtractor._strip_code_fences(text)
        try:
            return _json.loads(cleaned)
        except _json.JSONDecodeError:
            pass
        for i, ch in enumerate(cleaned):
            if ch not in "{[":
                continue
            depth = 0
            in_str = False
            esc = False
            for j in range(i, len(cleaned)):
                c = cleaned[j]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        in_str = False
                    continue
                if c == '"':
                    in_str = True
                elif c in "{[":
                    depth += 1
                elif c in "}]":
                    depth -= 1
                    if depth == 0:
                        try:
                            return _json.loads(cleaned[i : j + 1])
                        except _json.JSONDecodeError:
                            break
        return None

    def _chat(self, prompt: str, num_predict: int = 200) -> str:
        """Call LLM provider synchronously."""
        if not self._llm:
            return ""
        model = config.METADATA_MODEL or config.CHAT_MODEL
        return self._llm.chat_sync(prompt, model=model, num_predict=num_predict)
