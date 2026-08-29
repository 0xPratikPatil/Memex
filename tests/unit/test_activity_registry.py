"""Tests for ActivityRegistry and WAITING_GPU stage."""

from __future__ import annotations

import time

from memex.engine.core.progress import (
    ActivityRegistry,
    PipelineStage,
    activity_registry,
    stage_is_terminal,
)


def test_waiting_gpu_is_non_terminal():
    assert not stage_is_terminal(PipelineStage.WAITING_GPU)
    assert stage_is_terminal(PipelineStage.DONE)


def test_registry_report_and_snapshot():
    reg = ActivityRegistry()
    reg.report("a.md", PipelineStage.CONTEXT)
    reg.report("b.md", PipelineStage.METADATA)

    snap = reg.snapshot()
    assert snap == {"a.md": "Context", "b.md": "Metadata"}


def test_registry_remove():
    reg = ActivityRegistry()
    reg.report("a.md", PipelineStage.EMBEDDING)
    reg.remove("a.md")
    assert reg.snapshot() == {}


def test_registry_prunes_stale():
    reg = ActivityRegistry()
    reg.STALE_S = 0.05
    reg.report("a.md", PipelineStage.CONTEXT)
    time.sleep(0.08)
    assert reg.snapshot() == {}


def test_registry_overwrite_phase():
    reg = ActivityRegistry()
    reg.report("a.md", PipelineStage.CONTEXT)
    reg.report("a.md", PipelineStage.METADATA)
    assert reg.snapshot() == {"a.md": "Metadata"}


def test_module_singleton_registry():
    activity_registry.report("t.md", PipelineStage.EMBEDDING)
    try:
        assert activity_registry.snapshot()["t.md"] == "Embedding"
    finally:
        activity_registry.remove("t.md")
