"""Tests for the cross-file embedding batch accumulator."""

from __future__ import annotations

import time

import pytest

from memex.engine.core import config
from memex.engine.ingestion.embed_batcher import _EmbedBatcher, _FLUSH_TIMEOUT_S


class _FakeService:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(texts))] * 3 for _ in texts]


def test_fills_to_batch_size_then_flushes(monkeypatch):
    monkeypatch.setattr(config, "EMBED_BATCH_SIZE", 4)
    svc = _FakeService()
    batcher = _EmbedBatcher(svc.embed)

    vecs = batcher.submit(["a", "b", "c", "d"])
    assert len(vecs) == 4
    assert len(svc.calls) == 1
    assert svc.calls[0] == ["a", "b", "c", "d"]


def test_timeout_flush_small_batch(monkeypatch):
    monkeypatch.setattr(config, "EMBED_BATCH_SIZE", 64)
    svc = _FakeService()
    batcher = _EmbedBatcher(svc.embed)

    start = time.monotonic()
    vecs = batcher.submit(["only-one"])
    elapsed = time.monotonic() - start

    assert len(vecs) == 1
    assert svc.calls and svc.calls[0] == ["only-one"]
    assert elapsed < 2.0  # timeout flush, not a stall


def test_preserves_order_across_calls(monkeypatch):
    monkeypatch.setattr(config, "EMBED_BATCH_SIZE", 2)
    svc = _FakeService()
    batcher = _EmbedBatcher(svc.embed)

    first = batcher.submit(["x", "y"])
    second = batcher.submit(["z"])
    assert [v[0] for v in first] == [2.0, 2.0]
    assert second[0][0] in (2.0, 1.0)
    # ordering preserved: each caller's results correspond to its own texts
    assert len(first) == 2
    assert len(second) == 1


def test_failure_falls_back_per_text(monkeypatch):
    monkeypatch.setattr(config, "EMBED_BATCH_SIZE", 2)
    calls: list[list[str]] = []

    def flaky(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        if len(texts) > 1:
            raise RuntimeError("batch boom")
        return [[1.0, 2.0, 3.0]]

    batcher = _EmbedBatcher(flaky)
    vecs = batcher.submit(["p", "q"])
    assert len(vecs) == 2
    # the failed batch call + two per-text retries
    assert calls[0] == ["p", "q"]
    assert sorted(calls[1:]) == [["p"], ["q"]]


def test_mismatched_length_fills_gaps(monkeypatch):
    monkeypatch.setattr(config, "EMBED_BATCH_SIZE", 4)
    calls: list[list[str]] = []

    def short(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        if len(texts) == 4:
            return [[9.0, 9.0, 9.0]] * 3  # 3 vectors for 4 texts
        return [[7.0, 7.0, 7.0]]

    batcher = _EmbedBatcher(short)
    vecs = batcher.submit(["a", "b", "c", "d"])
    assert len(vecs) == 4
    assert vecs[3] == [7.0, 7.0, 7.0]  # gap filled per-text


def test_empty_submit():
    batcher = _EmbedBatcher(_FakeService().embed)
    assert batcher.submit([]) == []
