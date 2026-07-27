#!/usr/bin/env python3
"""End-to-end test for Memex MCP server — tests all 8 tools against live services.

Usage:
    uv run python scripts/test_e2e.py
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Create a test file
TEST_FILE = Path("/tmp/memex_e2e_test.md")
TEST_FILE.write_text("""# Memex E2E Test Document

This is a test document to verify the Memex RAG system works correctly.

## Features

- Direct file reading from filesystem
- Docling document conversion
- Hybrid search with dense + sparse embeddings
- Cross-encoder reranking
- Redis caching

## Test Data

The answer to the ultimate question is 42.
""")


def green(msg: str) -> str:
    return f"\033[32m{msg}\033[0m"


def red(msg: str) -> str:
    return f"\033[31m{msg}\033[0m"


def _mock_ctx() -> MagicMock:
    """Create a mock MCP Context for tool calls that require it."""
    ctx = MagicMock()
    ctx.report_progress = AsyncMock()
    return ctx


async def main():
    results: dict[str, bool] = {}
    mock_ctx = _mock_ctx()

    # ── 0. Service health ──────────────────────────────────────
    print("\n=== 0. Service Health ===")
    try:
        from memex.status import create_service_checker

        checker = create_service_checker()
        statuses = await checker.check_all()
        print(checker.get_status_summary(statuses))
        all_healthy = all(s.healthy for s in statuses.values())
        results["service_health"] = all_healthy
        print(green("PASS") if all_healthy else red("FAIL"))
    except Exception as e:
        print(red(f"FAIL: {e}"))
        results["service_health"] = False

    # ── 1. Unit tests ──────────────────────────────────────────
    print("\n=== 1. Unit Tests ===")
    import subprocess

    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
    )
    print(r.stdout[-200:] if len(r.stdout) > 200 else r.stdout)
    results["unit_tests"] = r.returncode == 0
    print(green("PASS") if r.returncode == 0 else red("FAIL"))

    # ── 2. rag_service_status ──────────────────────────────────
    print("\n=== 2. rag_service_status ===")
    try:
        from memex.server import rag_service_status

        status = await rag_service_status()
        data = json.loads(status)
        for name, info in data.items():
            healthy = ("error" not in info and "strategy" in info) if name == "chunker" else info.get("healthy", False)
            icon = "✓" if healthy else "✗"
            print(f"  {icon} {name}: {info.get('healthy', info.get('strategy', 'unknown'))}")
        all_ok = all(
            ("error" not in v and "strategy" in v) if k == "chunker" else v.get("healthy", False)
            for k, v in data.items()
        )
        results["service_status"] = all_ok
        print(green("PASS") if all_ok else red("FAIL"))
    except Exception as e:
        print(red(f"FAIL: {e}"))
        results["service_status"] = False

    # ── 3. rag_ingest_file ─────────────────────────────────────
    print("\n=== 3. rag_ingest_file ===")
    try:
        from memex.schemas import IngestFileInput
        from memex.server import rag_ingest_file

        r = await rag_ingest_file(IngestFileInput(file_path_or_url=str(TEST_FILE)), mock_ctx)
        print(f"  {r[:120]}...")
        ok = "Successfully ingested" in r or "Already ingested" in r
        results["ingest_file"] = ok
        print(green("PASS") if ok else red("FAIL"))
    except Exception as e:
        print(red(f"FAIL: {e}"))
        results["ingest_file"] = False

    # ── 4. rag_query ───────────────────────────────────────────
    print("\n=== 4. rag_query ===")
    try:
        from memex.schemas import QueryInput
        from memex.server import rag_query

        r = await rag_query(
            QueryInput(
                query="what is the answer to the ultimate question?",
                top_k=3,
                use_reranking=False,
            )
        )
        ok = "42" in r
        print(f"  {'Found 42!' if ok else '42 not found'}")
        results["query"] = ok
        print(green("PASS") if ok else red("FAIL"))
    except Exception as e:
        print(red(f"FAIL: {e}"))
        results["query"] = False

    # ── 5. rag_list_documents ──────────────────────────────────
    print("\n=== 5. rag_list_documents ===")
    try:
        from memex.schemas import ListDocumentsInput
        from memex.server import rag_list_documents

        r = await rag_list_documents(ListDocumentsInput())
        ok = str(TEST_FILE) in r
        print(f"  {'Found test doc' if ok else 'Test doc missing'}")
        results["list_docs"] = ok
        print(green("PASS") if ok else red("FAIL"))
    except Exception as e:
        print(red(f"FAIL: {e}"))
        results["list_docs"] = False

    # ── 6. rag_collection_stats ────────────────────────────────
    print("\n=== 6. rag_collection_stats ===")
    try:
        from memex.server import rag_collection_stats

        r = await rag_collection_stats()
        data = json.loads(r)
        print(f"  Points: {data.get('total_points')}, Status: {data.get('status')}")
        ok = data.get("total_points", 0) > 0
        results["collection_stats"] = ok
        print(green("PASS") if ok else red("FAIL"))
    except Exception as e:
        print(red(f"FAIL: {e}"))
        results["collection_stats"] = False

    # ── 7. rag_delete_document ─────────────────────────────────
    print("\n=== 7. rag_delete_document ===")
    try:
        from memex.schemas import DeleteDocumentInput
        from memex.server import rag_delete_document

        r = await rag_delete_document(DeleteDocumentInput(source_identifier=str(TEST_FILE)))
        print(f"  {r[:120]}")
        ok = "Successfully deleted" in r
        results["delete_doc"] = ok
        print(green("PASS") if ok else red("FAIL"))
    except Exception as e:
        print(red(f"FAIL: {e}"))
        results["delete_doc"] = False

    # ── 8. Lint ────────────────────────────────────────────────
    print("\n=== 8. Lint ===")
    r = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "."],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
    )
    print(r.stdout if r.stdout else "All checks passed!")
    results["lint"] = r.returncode == 0
    print(green("PASS") if r.returncode == 0 else red("FAIL"))

    # ── Clean up ───────────────────────────────────────────────
    TEST_FILE.unlink(missing_ok=True)

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 50)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    for name, ok in results.items():
        print(f"  {green('✓') if ok else red('✗')} {name}")
    print("=" * 50)

    if passed == total:
        print(green("\nALL TESTS PASSED"))
    else:
        print(red(f"\n{total - passed} TESTS FAILED"))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
