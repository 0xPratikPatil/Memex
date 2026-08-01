"""Unit tests for content-hash deduplication and partial ingest recovery."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from memex.engine.ingestion.hashing import (
    check_partial_ingest,
    clear_source_chunks,
    compute_chunk_hash,
    compute_content_hash,
    dedup_chunks,
    is_already_ingested,
)


class TestComputeContentHash:
    @pytest.mark.parametrize(
        "input_val",
        ["hello", b"hello"],
        ids=["str", "bytes"],
    )
    def test_deterministic_and_bytes_equivalent(self, input_val: str | bytes) -> None:
        assert compute_content_hash(input_val) == compute_content_hash("hello")

    def test_different_inputs_differ(self) -> None:
        assert compute_content_hash("hello") != compute_content_hash("world")

    def test_empty_string_returns_hex(self) -> None:
        h = compute_content_hash("")
        assert len(h) == 64  # SHA256 hex digest length
        assert all(c in "0123456789abcdef" for c in h)


class TestComputeChunkHash:
    def test_deterministic(self) -> None:
        assert compute_chunk_hash("id1", "content") == compute_chunk_hash("id1", "content")

    def test_different_id_changes_hash(self) -> None:
        assert compute_chunk_hash("id1", "content") != compute_chunk_hash("id2", "content")

    def test_different_content_changes_hash(self) -> None:
        assert compute_chunk_hash("id1", "a") != compute_chunk_hash("id1", "b")

    def test_returns_hex_string(self) -> None:
        h = compute_chunk_hash("id", "content")
        assert len(h) == 64


class TestDedupChunks:
    def test_removes_duplicates(self) -> None:
        chunks = [
            {"content": "same content", "section_header": "H1"},
            {"content": "same content", "section_header": "H2"},
            {"content": "different content", "section_header": "H3"},
        ]
        result = dedup_chunks(chunks)
        assert len(result) == 2
        assert result[0]["section_header"] == "H1"
        assert result[1]["section_header"] == "H3"

    def test_keeps_first_occurrence(self) -> None:
        chunks = [
            {"content": "dup", "idx": 0},
            {"content": "dup", "idx": 1},
            {"content": "dup", "idx": 2},
        ]
        result = dedup_chunks(chunks)
        assert len(result) == 1
        assert result[0]["idx"] == 0

    def test_empty_list(self) -> None:
        assert dedup_chunks([]) == []

    def test_no_duplicates(self) -> None:
        chunks = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
        result = dedup_chunks(chunks)
        assert len(result) == 3

    def test_single_chunk(self) -> None:
        chunks = [{"content": "only"}]
        result = dedup_chunks(chunks)
        assert len(result) == 1

    def test_whitespace_content_dedup(self) -> None:
        chunks = [
            {"content": "hello"},
            {"content": "hello"},
        ]
        result = dedup_chunks(chunks)
        assert len(result) == 1


def _make_mock_point(point_id: str, payload: dict | None = None) -> MagicMock:
    """Create a mock Qdrant point."""
    point = MagicMock()
    point.id = point_id
    point.payload = payload or {}
    return point


def _mock_scroll_return(points: list[MagicMock], next_offset=None) -> tuple:
    """Return value for qdrant.scroll()."""
    return points, next_offset


@pytest.mark.asyncio
class TestIsAlreadyIngested:
    async def test_returns_true_when_duplicate(self) -> None:
        mock_client = MagicMock()
        mock_client.scroll.return_value = _mock_scroll_return([_make_mock_point("p1", {"total_chunks": 10})])
        is_dup, count = await is_already_ingested(mock_client, "col", "/doc.pdf", "abc123")
        assert is_dup is True
        assert count == 10

    async def test_returns_false_when_not_found(self) -> None:
        mock_client = MagicMock()
        mock_client.scroll.return_value = _mock_scroll_return([])
        is_dup, count = await is_already_ingested(mock_client, "col", "/doc.pdf", "abc123")
        assert is_dup is False
        assert count == 0

    async def test_uses_correct_filter_fields(self) -> None:
        mock_client = MagicMock()
        mock_client.scroll.return_value = _mock_scroll_return([])
        await is_already_ingested(mock_client, "col", "/doc.pdf", "hash123")
        call_kwargs = mock_client.scroll.call_args[1]
        filter_obj = call_kwargs["scroll_filter"]
        # Should have two must conditions: source + content_hash
        assert len(filter_obj.must) == 2


@pytest.mark.asyncio
class TestCheckPartialIngest:
    async def test_returns_false_when_no_chunks(self) -> None:
        mock_client = MagicMock()
        mock_client.scroll.return_value = _mock_scroll_return([])
        result = await check_partial_ingest(mock_client, "col", "/doc.pdf", 5)
        assert result is False

    async def test_returns_false_when_count_matches(self) -> None:
        mock_client = MagicMock()
        points = [_make_mock_point(f"p{i}", {"total_chunks": 5}) for i in range(5)]
        mock_client.scroll.return_value = _mock_scroll_return(points)
        result = await check_partial_ingest(mock_client, "col", "/doc.pdf", 5)
        assert result is False

    async def test_returns_true_when_count_mismatch(self) -> None:
        mock_client = MagicMock()
        points = [_make_mock_point("p0"), _make_mock_point("p1")]
        mock_client.scroll.return_value = _mock_scroll_return(points)
        result = await check_partial_ingest(mock_client, "col", "/doc.pdf", 5)
        assert result is True


@pytest.mark.asyncio
class TestClearSourceChunks:
    async def test_returns_zero_when_no_chunks(self) -> None:
        mock_client = MagicMock()
        mock_client.scroll.return_value = _mock_scroll_return([])
        count = await clear_source_chunks(mock_client, "col", "/doc.pdf")
        assert count == 0
        mock_client.delete.assert_not_called()

    async def test_deletes_all_matching_chunks(self) -> None:
        mock_client = MagicMock()
        points = [_make_mock_point(f"p{i}") for i in range(3)]
        mock_client.scroll.return_value = _mock_scroll_return(points)
        count = await clear_source_chunks(mock_client, "col", "/doc.pdf")
        assert count == 3
        mock_client.delete.assert_called_once()

    async def test_deletes_in_batches(self) -> None:
        mock_client = MagicMock()
        with patch("memex.engine.ingestion.hashing.config.EMBED_BATCH_SIZE", 2):
            points = [_make_mock_point(f"p{i}") for i in range(5)]
            mock_client.scroll.return_value = _mock_scroll_return(points)
            count = await clear_source_chunks(mock_client, "col", "/doc.pdf")
            assert count == 5
            assert mock_client.delete.call_count == 3

    async def test_paginates_scroll_results(self) -> None:
        mock_client = MagicMock()
        batch1 = [_make_mock_point(f"p{i}") for i in range(2)]
        batch2 = [_make_mock_point(f"p{i}") for i in range(2, 4)]
        mock_client.scroll.side_effect = [
            _mock_scroll_return(batch1, next_offset="offset1"),
            _mock_scroll_return(batch2, next_offset=None),
        ]
        count = await clear_source_chunks(mock_client, "col", "/doc.pdf")
        assert count == 4
