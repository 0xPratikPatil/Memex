"""Citation-based answer generation."""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided documents.\n"
    "\n"
    "IMPORTANT RULES:\n"
    "1. Only answer based on the provided document chunks. If the documents don't contain"
    " enough information to answer the question, respond with exactly: {sentinel}\n"
    "2. Cite your sources using [N] markers where N is the chunk number (1-indexed).\n"
    "3. Every factual claim must have at least one citation.\n"
    "4. Be concise and direct.\n"
    "\n"
    "Document chunks:\n"
    "{context}"
)

REFUSAL_SENTINEL = "INSUFFICIENT_CONTEXT"

TRUNCATION_MARKER = "\n[truncated]"
MIN_CHUNK_CHARS = 200

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_SENTENCE_END = re.compile(r"[.!?]\s+")


def _split_sentences(text: str) -> list[str]:
    result: list[str] = []
    pos = 0
    for m in _SENTENCE_END.finditer(text):
        result.append(text[pos : m.end()].strip())
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        result.append(tail)
    return result


@dataclass
class Citation:
    index: int
    source: str
    chunk_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    rerank_score: float | None = None


@dataclass
class Answer:
    text: str
    refused: bool
    confidence: float
    citations: list[Citation]
    sources: list[str]
    filters_used: dict | None = None

    def formatted(self) -> str:
        """Answer text with numbered source list appended."""
        if not self.citations:
            return self.text
        lines = [self.text, "", "Sources:"]
        seen: list[str] = []
        for c in self.citations:
            if c.source not in seen:
                seen.append(c.source)
                lines.append(f"  [{c.index}] {c.source}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        state = "refused" if self.refused else f"{len(self.citations)} citations"
        return f"<Answer {state}, confidence={self.confidence:.2f}>"

    def __bool__(self) -> bool:
        return not self.refused


# ── Context packing ───────────────────────────────────────────────────────────


def _pack_context(
    chunks: list[dict],
    max_context_chars: int = 8000,
) -> tuple[str, list[dict]]:
    """Pack retrieved chunks into a numbered source block within a character budget.

    Chunks are added in rank order (best first) until the budget is exhausted.
    Oversized chunks are truncated; those under MIN_CHUNK_CHARS after truncation
    are dropped.

    Returns:
        (context_string, chunks_that_were_included)
    """
    blocks: list[str] = []
    used: list[dict] = []
    spent = 0

    for chunk in chunks:
        source = chunk.get("source", "unknown")
        content = chunk.get("content", "")
        header = f"[{len(used) + 1}] Source: {source}\n"
        separator = 2 if used else 0  # "\n\n" length between blocks
        remaining = max_context_chars - spent - len(header) - separator

        if remaining <= 0:
            break

        text = content
        if len(text) > remaining:
            keep = remaining - len(TRUNCATION_MARKER)
            if keep < MIN_CHUNK_CHARS:
                text = text[:MIN_CHUNK_CHARS] + TRUNCATION_MARKER
            else:
                text = text[:keep].rstrip() + TRUNCATION_MARKER

        blocks.append(header + text)
        used.append(chunk)
        spent += separator + len(header) + len(text)

    if len(used) < len(chunks):
        log.info(
            "Context budget of %d chars fit %d of %d retrieved chunks",
            max_context_chars,
            len(used),
            len(chunks),
        )

    return "\n\n".join(blocks), used


# ── Refusal detection ─────────────────────────────────────────────────────────


def _is_refusal(text: str, sentinel: str = REFUSAL_SENTINEL) -> bool:
    """Check whether the model declined to answer.

    Handles the bare sentinel, case-insensitive matches, and the common
    pattern where the model wraps the sentinel in polite prose.
    """
    stripped = text.strip()
    if not stripped:
        return True
    lower = stripped.lower()
    sentinel_lower = sentinel.lower()
    if sentinel_lower in lower:
        return len(stripped) < len(sentinel) + 120
    return False


# ── Citation parsing ──────────────────────────────────────────────────────────


def _parse_citations(
    text: str,
    used_chunks: list[dict],
) -> tuple[str, list[Citation]]:
    """Extract [N] citation markers and resolve them to chunk sources.

    Invalid references (N out of range) are stripped from the text.
    Deduplicates by citation index, preserving first-appearance order.
    """
    citations: list[Citation] = []
    seen: set[int] = set()
    invalid: list[str] = []

    for match in _CITATION_PATTERN.finditer(text):
        number = int(match.group(1))

        if not 1 <= number <= len(used_chunks):
            invalid.append(match.group(0))
            continue

        if number in seen:
            continue
        seen.add(number)

        chunk = used_chunks[number - 1]
        citations.append(
            Citation(
                index=number,
                source=chunk.get("source", "unknown"),
                chunk_text=chunk.get("content", ""),
                metadata=chunk.get("metadata", {}),
                rerank_score=chunk.get("rerank_score"),
            )
        )

    if invalid:
        log.warning(
            "Answer referenced %d invalid source numbers: %s",
            len(invalid),
            ", ".join(sorted(set(invalid))),
        )
        for marker in set(invalid):
            text = text.replace(marker, "")
        text = re.sub(r"[ ]{2,}", " ", text).strip()

    return text, citations


# ── Confidence scoring ────────────────────────────────────────────────────────


def _citation_confidence(text: str) -> float:
    sentences = _split_sentences(text)
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if _CITATION_PATTERN.search(s))
    return cited / len(sentences)


# ── Public API ────────────────────────────────────────────────────────────────


async def generate_answer(
    query: str,
    chunks: list[dict[str, Any]],
    ollama_chat_fn: Callable[[str], Awaitable[str]],
    max_context_chars: int = 8000,
    system_prompt: str | None = None,
    refusal_sentinel: str = REFUSAL_SENTINEL,
) -> Answer:
    """Generate a cited answer from retrieved chunks.

    1. Pack chunks into context budget (max_context_chars)
    2. Build system prompt with citation instructions + refusal sentinel
    3. Call LLM
    4. Parse response: detect refusal, extract citations
    5. Compute confidence = fraction of sentences with citations
    6. Return Answer

    Args:
        query: The user's question.
        chunks: Retrieved chunks (dicts with at least 'content' and 'source').
        ollama_chat_fn: Async callable ``(prompt: str) -> str`` that calls the
            chat LLM and returns the assistant message content.
        max_context_chars: Total character budget for all chunks.
        system_prompt: Override the default system prompt. Use ``{sentinel}``
            and ``{context}`` as placeholders.
        refusal_sentinel: The exact string the model should return when it
            cannot answer from the provided context.

    Returns:
        An Answer. Never raises for unanswerable questions — returns a refusal
        instead.
    """
    if not chunks:
        return Answer(
            text="No documents were retrieved for this question, so there is nothing to answer from.",
            refused=True,
            confidence=0.0,
            citations=[],
            sources=[],
        )

    context, used = _pack_context(chunks, max_context_chars)

    if not used:
        return Answer(
            text="No documents were retrieved for this question, so there is nothing to answer from.",
            refused=True,
            confidence=0.0,
            citations=[],
            sources=[],
        )

    prompt_template = system_prompt or DEFAULT_SYSTEM_PROMPT
    system_msg = prompt_template.replace("{sentinel}", refusal_sentinel).replace("{context}", context)
    full_prompt = f"{system_msg}\n\nQuestion: {query}"

    try:
        response = await ollama_chat_fn(full_prompt)
    except Exception:
        log.exception("LLM call failed during answer generation")
        return Answer(
            text="Answer generation failed due to an LLM error.",
            refused=True,
            confidence=0.0,
            citations=[],
            sources=[],
        )

    if not isinstance(response, str):
        response = str(response)
    response = response.strip()

    if _is_refusal(response, refusal_sentinel):
        log.info("Model refused to answer: %s", query[:60])
        return Answer(
            text=("The retrieved documents do not contain enough information to answer this question."),
            refused=True,
            confidence=0.0,
            citations=[],
            sources=[],
        )

    response, citations = _parse_citations(response, used)
    confidence = _citation_confidence(response)
    sources = list(dict.fromkeys(c.source for c in citations))

    return Answer(
        text=response,
        refused=False,
        confidence=confidence,
        citations=citations,
        sources=sources,
    )
