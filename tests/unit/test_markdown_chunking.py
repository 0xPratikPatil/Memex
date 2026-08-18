"""Unit tests for splitter.py — markdown-aware chunking."""

from __future__ import annotations

from memex.engine.ingestion.splitter import _classify_block, chunk_markdown_aware


class TestClassifyBlock:
    def test_table(self) -> None:
        assert _classify_block("| a | b |\n|---|---|\n| 1 | 2 |") == "table"

    def test_code_block(self) -> None:
        assert _classify_block("```python\nprint('hi')\n```") == "code"

    def test_heading(self) -> None:
        assert _classify_block("## Section Title") == "heading"

    def test_list_dash(self) -> None:
        assert _classify_block("- item one\n- item two") == "list"

    def test_list_star(self) -> None:
        assert _classify_block("* item one\n* item two") == "list"

    def test_list_numbered(self) -> None:
        assert _classify_block("1. first\n2. second") == "list"

    def test_paragraph(self) -> None:
        assert _classify_block("Just some regular text.") == "paragraph"


class TestChunkMarkdownAware:
    def test_empty_input(self) -> None:
        assert chunk_markdown_aware("") == []
        assert chunk_markdown_aware("   ") == []

    def test_single_paragraph(self) -> None:
        result = chunk_markdown_aware("Hello world")
        assert len(result) == 1
        assert result[0]["content"] == "Hello world"
        assert result[0]["chunk_index"] == 0

    def test_table_not_split(self) -> None:
        table = "| Name | Value |\n|------|-------|\n| A | 1 |\n| B | 2 |\n| C | 3 |"
        result = chunk_markdown_aware(table, chunk_size=30)
        # Table should be kept whole even though it exceeds chunk_size
        assert len(result) == 1
        assert "| Name | Value |" in result[0]["content"]

    def test_list_not_split(self) -> None:
        items = "\n".join([f"- Item {i}" for i in range(20)])
        result = chunk_markdown_aware(items, chunk_size=50)
        # Each chunk should contain complete list items
        for chunk in result:
            lines = chunk["content"].split("\n")
            for line in lines:
                assert line.startswith("- Item")

    def test_code_block_not_split(self) -> None:
        code = "```python\ndef foo():\n    return 42\n```"
        result = chunk_markdown_aware(code, chunk_size=20)
        assert len(result) == 1
        assert "```python" in result[0]["content"]

    def test_heading_tracked(self) -> None:
        md = "## Section 1\n\nParagraph one.\n\n## Section 2\n\nParagraph two."
        result = chunk_markdown_aware(md, chunk_size=1000)
        assert len(result) >= 1
        # Last section_header should be from the last heading block
        assert "Section 2" in result[-1]["section_header"]

    def test_multiple_chunks_with_overlap(self) -> None:
        paragraphs = "\n\n".join([f"Paragraph {i} with some text." for i in range(10)])
        result = chunk_markdown_aware(paragraphs, chunk_size=80, overlap=20)
        assert len(result) > 1
        # Each chunk should have content
        for chunk in result:
            assert len(chunk["content"]) > 0

    def test_chunk_index_sequential(self) -> None:
        paragraphs = "\n\n".join([f"Paragraph {i}." for i in range(5)])
        result = chunk_markdown_aware(paragraphs, chunk_size=30)
        for i, chunk in enumerate(result):
            assert chunk["chunk_index"] == i

    def test_mixed_content(self) -> None:
        md = """## Overview

Some introduction text here.

| Feature | Status |
|---------|--------|
| A | Done |
| B | WIP |

- Point one
- Point two

More text after the list."""
        result = chunk_markdown_aware(md, chunk_size=100)
        assert len(result) >= 1
        # All content should be present
        full_text = " ".join(c["content"] for c in result)
        assert "introduction" in full_text
        assert "Feature" in full_text
        assert "Point one" in full_text
