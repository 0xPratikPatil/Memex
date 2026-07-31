"""Agent-oriented filter tools — context and extraction.

Provides two async helpers:

* ``get_filter_context`` — discovers metadata fields stored in Qdrant,
  collects their unique values, and optionally suggests filters for a query.
* ``extract_filters`` — uses an LLM to parse a natural language query into
  structured metadata filters given the available fields.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

MAX_VALUES_PER_FIELD: int = 100

# Internal payload keys that should never be surfaced to agents.
_INTERNAL_KEYS = frozenset(
    {
        "source",
        "content_hash",
        "content",
        "context_prefix",
        "chunk_index",
        "total_chunks",
        "ingested_at",
        "file_mtime",
        "file_size",
        "section_header",
    }
)


@dataclass
class FieldInfo:
    """Describes a single metadata field discovered in the vector store."""

    name: str
    type: str  # "string" | "integer" | "list"
    values: list[str]  # stored unique values (capped)
    count: int  # number of chunks carrying this field


@dataclass
class FilterContext:
    """Full picture of what metadata is available for filtering."""

    fields: list[FieldInfo]
    suggested_filters: dict | None = None
    sample_query: str = ""


@dataclass
class ExtractedFilters:
    """Result of LLM-based filter extraction from a natural language query."""

    filters: dict
    explanation: str
    confidence: float


# ── Field discovery ────────────────────────────────────────────────────────


def _classify_field(values: list[object]) -> str:
    """Heuristic field-type classification from a sample of values.

    Checks all values: if all are int → integer, all float → float,
    any list → list, otherwise → keyword (string).
    """
    if not values:
        return "string"
    if any(isinstance(v, list) for v in values):
        return "list"
    nums = [v for v in values if isinstance(v, (int, float))]
    if len(nums) == len(values):
        if all(isinstance(v, int) for v in nums):
            return "integer"
        return "float"
    return "keyword"


def _discover_fields_from_scroll(
    qdrant_client: Any,
    collection: str,
) -> list[FieldInfo]:
    """Walk every payload key via scroll and collect unique values.

    This is the brute-force fallback when Qdrant doesn't expose a
    convenient payload-index listing.  We paginate through all points,
    extract every top-level key, and collect unique values.
    """
    field_values: dict[str, set[str]] = {}
    field_count: dict[str, int] = {}
    field_raw_samples: dict[str, list[object]] = {}

    offset = None
    while True:
        result = qdrant_client.scroll(
            collection_name=collection,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, next_offset = result
        for point in points:
            payload = point.payload or {}
            for key, val in payload.items():
                if key in _INTERNAL_KEYS:
                    continue
                field_count[key] = field_count.get(key, 0) + 1
                if key not in field_raw_samples:
                    field_raw_samples[key] = []
                if len(field_raw_samples[key]) < 5:
                    field_raw_samples[key].append(val)

                # Flatten list values for unique tracking
                if isinstance(val, list):
                    for item in val:
                        if len(field_values.get(key, set())) < MAX_VALUES_PER_FIELD:
                            field_values.setdefault(key, set()).add(str(item))
                elif val is not None and len(field_values.get(key, set())) < MAX_VALUES_PER_FIELD:
                    field_values.setdefault(key, set()).add(str(val))

        if next_offset is None:
            break
        offset = next_offset

    fields: list[FieldInfo] = []
    for name in sorted(field_values.keys()):
        ftype = _classify_field(field_raw_samples.get(name, []))
        unique = sorted(field_values[name])[:MAX_VALUES_PER_FIELD]
        fields.append(
            FieldInfo(
                name=name,
                type=ftype,
                values=unique,
                count=field_count.get(name, 0),
            )
        )
    return fields


def _get_known_metadata_fields() -> list[FieldInfo]:
    """Return the statically known metadata fields from the pipeline schema.

    Used as a fast-path when Qdrant is empty or unreachable — the agent
    still sees which fields *will* be available once documents are ingested.
    """
    return [
        FieldInfo(name="doc_type", type="string", values=[], count=0),
        FieldInfo(name="topics", type="list", values=[], count=0),
        FieldInfo(name="language", type="string", values=[], count=0),
        FieldInfo(name="keywords", type="list", values=[], count=0),
        FieldInfo(name="entities", type="list", values=[], count=0),
        FieldInfo(name="dates", type="list", values=[], count=0),
    ]


# ── Filter normalization ──────────────────────────────────────────────────────


def _parse_filters(raw: dict | str | None) -> dict:
    """Parse filter value from dict or JSON string.

    LLMs often send JSON strings where the schema expects an object.
    Accept both for robustness.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            log.debug("Failed to parse filter string as JSON: %.40s...", raw)
    return {}


# ── Public API ─────────────────────────────────────────────────────────────


async def get_filter_context(
    config: Any,
    query: str | None = None,
    qdrant_client: Any | None = None,
    collection: str = "memex",
    llm_call: Callable[[str], Awaitable[str]] | None = None,
) -> FilterContext:
    """Get available metadata fields, their values, and optionally suggest filters.

    1. Scan Qdrant scroll for all metadata fields
    2. Collect unique values per field (capped at 100)
    3. If query provided and llm_call available, extract suggested filters
    4. Return FilterContext
    """
    if qdrant_client is None:
        log.warning("No Qdrant client provided, returning known schema fields")
        return FilterContext(fields=_get_known_metadata_fields())

    try:
        fields = _discover_fields_from_scroll(qdrant_client, collection)
    except Exception:
        log.warning("Field discovery failed, returning known schema fields", exc_info=True)
        return FilterContext(fields=_get_known_metadata_fields())

    if not fields:
        fields = _get_known_metadata_fields()

    suggested = None
    if query and llm_call is not None:
        try:
            extracted = await extract_filters(query, fields, llm_call=llm_call)
            if extracted.filters:
                suggested = extracted.filters
        except Exception:
            log.debug("Filter suggestion failed for query: %s", query[:60], exc_info=True)

    return FilterContext(
        fields=fields,
        suggested_filters=suggested,
        sample_query=query or "",
    )


# ── LLM filter extraction ──────────────────────────────────────────────────


def _build_extraction_prompt(query: str, available_fields: list[FieldInfo]) -> str:
    """Build the LLM prompt that asks the model to extract metadata filters."""
    field_descriptions: list[str] = []
    for f in available_fields:
        values_str = ", ".join(f.values[:20]) if f.values else "(no values indexed yet)"
        field_descriptions.append(f"- {f.name} ({f.type}): {values_str}")

    fields_block = "\n".join(field_descriptions)

    return (
        "You are a metadata filter extractor for a document search system.\n"
        "Given a user query and the available metadata fields, extract the "
        "metadata filters that should narrow the search.\n\n"
        "## Available Fields\n"
        f"{fields_block}\n\n"
        "## User Query\n"
        f"{query}\n\n"
        "## Instructions\n"
        "- Return ONLY a JSON object with field names as keys.\n"
        "- For string fields: use the exact value (lowercase).\n"
        "- For list fields: use a list of matching values.\n"
        "- If a field has no relevant filter, do not include it.\n"
        "- Do not invent field names that are not listed above.\n"
        "- Return a short explanation of your reasoning after the JSON, "
        "separated by a newline.\n\n"
        "## Response Format\n"
        '```json\n{"field_name": "value"}\n```\n'
        "Explanation: ..."
    )


def _parse_llm_response(response: str) -> tuple[dict, str, float]:
    """Parse the LLM response into (filters, explanation, confidence).

    Attempts to extract a JSON object from the response, falling back to
    an empty dict if parsing fails.  Confidence is a rough heuristic based
    on whether valid JSON was found.
    """
    explanation = ""
    filters: dict = {}

    # Strip markdown code fences if present
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        start = 1
        end = len(lines) - 1
        if lines[-1].strip() == "```":
            end = len(lines) - 1
        text = "\n".join(lines[start:end])

    # Try to split JSON from explanation
    parts = text.split("\n", 1)
    json_part = parts[0].strip()
    explanation = parts[1].strip() if len(parts) > 1 else ""

    # Clean up explanation prefix if present
    if explanation.lower().startswith("explanation:"):
        explanation = explanation[len("explanation:") :].strip()

    try:
        filters = json.loads(json_part)
        if not isinstance(filters, dict):
            filters = {}
            confidence = 0.2
        else:
            confidence = min(0.5 + 0.1 * len(filters), 0.95)
    except (json.JSONDecodeError, ValueError):
        # Try harder — find the first { ... } block
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                filters = json.loads(match.group())
                if isinstance(filters, dict):
                    confidence = min(0.4 + 0.1 * len(filters), 0.85)
                else:
                    filters = {}
                    confidence = 0.15
            except (json.JSONDecodeError, ValueError):
                filters = {}
                confidence = 0.1
                explanation = response.strip()
        else:
            filters = {}
            confidence = 0.1
            explanation = response.strip()

    return filters, explanation, confidence


async def extract_filters(
    query: str,
    available_fields: list[FieldInfo],
    llm_call: Callable[[str], Awaitable[str]] | None = None,
) -> ExtractedFilters:
    """Extract metadata filters from a natural language query.

    Uses LLM to parse query into structured filters given available fields.
    Returns ExtractedFilters with filters dict, explanation, and confidence.
    """
    if llm_call is None:
        return ExtractedFilters(
            filters={},
            explanation="No LLM available for filter extraction",
            confidence=0.0,
        )

    if not available_fields:
        return ExtractedFilters(
            filters={},
            explanation="No metadata fields available for filtering",
            confidence=0.0,
        )

    prompt = _build_extraction_prompt(query, available_fields)
    try:
        response = await llm_call(prompt)
    except Exception:
        log.warning("LLM filter extraction call failed", exc_info=True)
        return ExtractedFilters(
            filters={},
            explanation="LLM call failed",
            confidence=0.0,
        )

    filters, explanation, confidence = _parse_llm_response(response)

    # Validate extracted field names against available fields
    valid_names = {f.name for f in available_fields}
    validated = {k: v for k, v in filters.items() if k in valid_names}

    if len(validated) < len(filters):
        dropped = set(filters.keys()) - valid_names
        log.debug("Dropped unknown fields: %s", dropped)
        confidence *= 0.8

    return ExtractedFilters(
        filters=validated,
        explanation=explanation,
        confidence=round(confidence, 2),
    )


async def auto_extract_filters(
    query: str,
    llm_call: Callable[[str], Awaitable[str]] | None = None,
) -> dict:
    """Auto-extract metadata filters from a query when auto_filter is enabled.

    Discovers available fields, runs LLM extraction, and returns a normalized
    filter dict suitable for search.
    """
    if llm_call is None:
        return {}
    try:
        fields = _get_known_metadata_fields()
        result = await extract_filters(query, fields, llm_call=llm_call)
        if result.confidence > 0.3 and result.filters:
            return _parse_filters(result.filters)
    except Exception:
        log.debug("Auto filter extraction failed for query: %s", query[:60], exc_info=True)
    return {}
