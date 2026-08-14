# Production Reliability: Async Docling + Status Tracking + Retry Queue

**Date**: 2026-08-14
**Status**: Draft
**Supersedes**: None (new capability)

## Problem Statement

### Current Issues

1. **Docling sync API blocks and times out on large files**
   - `loader.py` uses `POST /v1/convert/source` (sync) — blocks until done or 504 timeout
   - `splitter.py` uses `POST /v1/chunk/hybrid/source` (sync) — same issue
   - 504 Gateway Timeout on large/complex PDFs (e.g., 100+ pages with tables/images)
   - No visibility into what Docling is doing during the block

2. **No file status tracking**
   - Files processed in fire-and-forget manner
   - No way to know which files are pending, processing, done, or failed
   - No retry mechanism — failed files are just logged and forgotten

3. **CLI display issues in tmux**
   - Rich Live Table wraps badly in narrow tmux panes
   - Table columns expand beyond terminal width

4. **Other timeout issues across the system**
   - `ingestion.py`: 120s parse timeout, 300s total timeout — hard limits, no retry
   - `pipeline.py`: Qdrant operations have 10s timeout — too aggressive for large batches
   - `server.py`: MCP tools use 120s timeout — may be insufficient for complex queries

## Solution: Self-Aware Document Processing Pipeline

### Core Principle

**Every system should know what's happening with files — no silent failures, no lost files, no wasted resources.**

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      CLI / MCP Server                       │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │ memex    │  │ rag_service  │  │ rag_sync            │   │
│  │ status   │  │ _status      │  │ (with retry queue)  │   │
│  └──────────┘  └──────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    StatusTracker (Qdrant)                    │
│  - File status: pending | processing | done | retry | failed│
│  - Retry count, next retry time, last error                 │
│  - Docling task ID for async tracking                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   DoclingAsyncClient                         │
│  - POST /v1/convert/source/async → task_id                  │
│  - GET /v1/status/poll/{task_id} → status + position        │
│  - GET /v1/result/{task_id} → conversion result             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Docling Serve (Docker)                    │
│  - /health → liveness (always 200)                          │
│  - /ready → readiness (503 until models loaded)             │
│  - /v1/convert/source/async → non-blocking submission       │
│  - /v1/status/poll/{task_id} → pending|started|success|fail │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: DoclingAsyncClient

**File**: `memex/engine/ingestion/docling_client.py` (new)

Replace sync HTTP calls with async-aware client:

```python
class DoclingAsyncClient:
    """Async Docling Serve client with status polling."""

    def __init__(self, base_url: str, api_key: str | None = None):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))

    async def health_check(self) -> bool:
        """Check if Docling is reachable and ready."""
        resp = await self._client.get(f"{self._base_url}/health")
        return resp.status_code == 200

    async def ready_check(self) -> bool:
        """Check if Docling models are loaded and ready."""
        try:
            resp = await self._client.get(f"{self._base_url}/ready")
            return resp.status_code == 200
        except httpx.HTTPStatusError:
            return False

    async def submit_conversion(self, payload: dict) -> str:
        """Submit async conversion job, return task_id."""
        resp = await self._client.post(
            f"{self._base_url}/v1/convert/source/async",
            json=payload,
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()["task_id"]

    async def submit_chunking(self, payload: dict) -> str:
        """Submit async chunking job, return task_id."""
        resp = await self._client.post(
            f"{self._base_url}/v1/chunk/hybrid/source/async",
            json=payload,
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()["task_id"]

    async def poll_status(self, task_id: str) -> dict:
        """Poll task status. Returns {task_status, task_position, ...}."""
        resp = await self._client.get(
            f"{self._base_url}/v1/status/poll/{task_id}",
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def get_result(self, task_id: str) -> dict:
        """Fetch completed conversion result."""
        resp = await self._client.get(
            f"{self._base_url}/v1/result/{task_id}",
            headers=self._headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def wait_for_completion(
        self, task_id: str, poll_interval: float = 5.0, max_wait: float = 600.0
    ) -> dict:
        """Poll until task completes or times out."""
        start = time.monotonic()
        while True:
            status = await self.poll_status(task_id)
            if status["task_status"] in ("success", "failure"):
                return status
            if time.monotonic() - start > max_wait:
                raise TimeoutError(f"Task {task_id} exceeded {max_wait}s")
            await asyncio.sleep(poll_interval)
```

### Phase 2: StatusTracker (Qdrant Payload)

**File**: `memex/engine/sources/status_tracker.py` (new)

Store file status in Qdrant payload alongside vectors:

```python
class FileStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    RETRY = "retry"
    FAILED = "failed"

class StatusTracker:
    """Track file processing status in Qdrant payload."""

    def __init__(self, qdrant_client, collection: str):
        self._qdrant = qdrant_client
        self._collection = collection

    def update_status(
        self,
        source_id: str,
        status: str,
        error: str | None = None,
        retry_count: int = 0,
        next_retry_at: str | None = None,
        docling_task_id: str | None = None,
    ) -> None:
        """Update file status in Qdrant payload."""
        # Find all points with this source_id
        results = self._qdrant.scroll(
            collection_name=self._collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
            ),
            limit=1,
        )
        if results[0]:
            point_id = results[0][0].id
            payload_update = {
                "processing_status": status,
                "status_updated_at": datetime.now(UTC).isoformat(),
            }
            if error:
                payload_update["last_error"] = error
            if retry_count > 0:
                payload_update["retry_count"] = retry_count
            if next_retry_at:
                payload_update["next_retry_at"] = next_retry_at
            if docling_task_id:
                payload_update["docling_task_id"] = docling_task_id
            self._qdrant.set_payload(
                collection_name=self._collection,
                payload=payload_update,
                points=[point_id],
            )

    def get_pending_retries(self) -> list[str]:
        """Get files due for retry."""
        results, _ = self._qdrant.scroll(
            collection_name=self._collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="processing_status", match=MatchValue(value="retry")),
                    FieldCondition(
                        key="next_retry_at",
                        range=Range(lte=datetime.now(UTC).isoformat()),
                    ),
                ]
            ),
            limit=100,
        )
        return [p.payload.get("source_id") for p in results]

    def get_status_summary(self) -> dict:
        """Get counts by status."""
        summary = {}
        for status in [FileStatus.PENDING, FileStatus.PROCESSING, FileStatus.DONE,
                       FileStatus.RETRY, FileStatus.FAILED]:
            count, _ = self._qdrant.count(
                collection_name=self._collection,
                count_filter=Filter(
                    must=[FieldCondition(key="processing_status", match=MatchValue(value=status))]
                ),
            )
            summary[status] = count
        return summary
```

### Phase 3: RetryQueue with Exponential Backoff

**File**: `memex/engine/sources/retry_queue.py` (new)

```python
class RetryQueue:
    """Exponential backoff retry queue for failed Docling operations."""

    BACKOFF_SCHEDULE = [
        60,        # 1 minute
        300,       # 5 minutes
        1800,      # 30 minutes
        7200,      # 2 hours
    ]
    MAX_RETRIES = 4

    def __init__(self, status_tracker: StatusTracker, docling_client: DoclingAsyncClient):
        self._tracker = status_tracker
        self._docling = docling_client

    def should_retry(self, error: str, retry_count: int) -> bool:
        """Determine if error is retryable."""
        if retry_count >= self.MAX_RETRIES:
            return False
        # Retry on timeout, 503, 504, connection errors
        retryable = ["timeout", "503", "504", "connection", "gateway"]
        return any(r in error.lower() for r in retryable)

    def schedule_retry(self, source_id: str, error: str, retry_count: int) -> None:
        """Schedule retry with exponential backoff."""
        if not self.should_retry(error, retry_count):
            self._tracker.update_status(source_id, FileStatus.FAILED, error=error)
            return

        backoff = self.BACKOFF_SCHEDULE[min(retry_count, len(self.BACKOFF_SCHEDULE) - 1)]
        next_retry = datetime.now(UTC) + timedelta(seconds=backoff)
        self._tracker.update_status(
            source_id,
            FileStatus.RETRY,
            error=error,
            retry_count=retry_count + 1,
            next_retry_at=next_retry.isoformat(),
        )

    async def process_retries(self) -> int:
        """Process files due for retry. Returns count of retried files."""
        pending = self._tracker.get_pending_retries()
        retried = 0
        for source_id in pending:
            try:
                # Re-submit to Docling
                task_id = await self._docling.submit_conversion(...)
                self._tracker.update_status(source_id, FileStatus.PROCESSING, docling_task_id=task_id)
                retried += 1
            except Exception as exc:
                self.schedule_retry(source_id, str(exc), retry_count + 1)
        return retried
```

### Phase 4: Compact CLI Display

**File**: `memex/cli.py` (modify)

Replace Rich Live Table with compact single-line display:

```python
# Old: Table with columns
table = Table()
table.add_column("File", ratio=3)
table.add_column("Stage", ratio=2)
table.add_column("Chunks", ratio=1)

# New: Compact single-line
def _build_compact_status(active: OrderedDict, completed: int, total: int) -> str:
    """Build compact single-line status for each active file."""
    lines = []
    for path, (stage, chunks, error) in list(active.items())[-4:]:  # Max 4 visible
        fname = os.path.basename(path)
        icon, color, label = _STAGE_STYLE.get(stage, ("?", "dim", stage))
        if stage == "Error":
            lines.append(f"  [{color}]{icon} {fname}: {error}[/{color}]")
        else:
            chunk_str = f" ({chunks})" if chunks > 0 else ""
            lines.append(f"  [{color}]{icon} {fname}{chunk_str}[/{color}]")

    hidden = len(active) - min(len(active), 4)
    if hidden > 0:
        lines.append(f"  [dim]... and {hidden} more[/dim]")

    pct = completed / total * 100 if total > 0 else 0
    lines.append(f"  [progress.percentage]{pct:.1f}%[/progress.percentage] {completed}/{total}")

    return "\n".join(lines)
```

### Phase 5: Status Command

**File**: `memex/cli.py` (add new command)

```python
@app.command()
def status(
    config_path: str = typer.Option("config.yaml", "--config", "-c"),
) -> None:
    """Show processing status for all files."""
    from memex.engine.sources.status_tracker import StatusTracker

    yaml_config = YamlConfig(config_path)
    engine = RAGEngine()
    tracker = StatusTracker(engine._get_qdrant(), config.COLLECTION_NAME)

    summary = tracker.get_status_summary()

    table = Table(title="File Processing Status")
    table.add_column("Status", style="bold")
    table.add_column("Count", justify="right")
    table.add_row("Pending", str(summary.get("pending", 0)))
    table.add_row("Processing", str(summary.get("processing", 0)))
    table.add_row("Done", f"[green]{summary.get('done', 0)}[/green]")
    table.add_row("Retrying", f"[yellow]{summary.get('retry', 0)}[/yellow]")
    table.add_row("Failed", f"[red]{summary.get('failed', 0)}[/red]" if summary.get("failed") else "0")

    console.print(Panel(table))
```

### Phase 6: MCP Status Tool

**File**: `memex/mcp/server.py` (extend existing)

```python
@mcp.tool()
def rag_processing_status() -> dict:
    """Show file processing status from Qdrant."""
    # Query Qdrant for status counts
    # Return {pending: N, processing: N, done: N, retry: N, failed: N}
```

### Phase 7: Update Existing Timeout Handling

**Files to modify**:

| File | Current | New |
|------|---------|-----|
| `loader.py` | Sync POST with 300s timeout | Async submit + poll |
| `splitter.py` | Sync POST with 300s timeout | Async submit + poll |
| `ingestion.py` | 120s parse, 300s total hard limits | Use async Docling, soft limits with retry |
| `pipeline.py` | Qdrant 10s timeout | Increase to 30s for batch ops |
| `server.py` | MCP 120s timeout | Increase to 180s for complex queries |

## Config Changes

**File**: `config.yaml`, `config.example.yaml`, `config.py`

```yaml
converter:
  docling_timeout: 300.0              # Max wait for async result polling (sync timeout removed)
  docling_poll_interval: 5.0          # Seconds between status polls
  docling_max_retries: 4              # Max retry attempts per file
  docling_retry_backoff: [60, 300, 1800, 7200]  # Exponential backoff in seconds

ingestion:
  timeout_parse: 120.0                # Keep as soft limit (log warning, don't hard fail)
  timeout_total: 300.0                # Keep as soft limit

qdrant:
  timeout: 30.0                       # Increased from 10s for batch operations

http:
  timeout: 180.0                      # Increased from 120s for complex MCP queries
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `memex/engine/ingestion/docling_client.py` | Create | Async Docling client with status polling |
| `memex/engine/sources/status_tracker.py` | Create | Qdrant-based file status tracking |
| `memex/engine/sources/retry_queue.py` | Create | Exponential backoff retry queue |
| `memex/cli.py` | Modify | Compact single-line display + status command |
| `memex/mcp/server.py` | Modify | Add rag_processing_status tool |
| `memex/engine/ingestion/loader.py` | Modify | Use DoclingAsyncClient |
| `memex/engine/ingestion/splitter.py` | Modify | Use DoclingAsyncClient |
| `memex/engine/ingestion/ingestion.py` | Modify | Integrate retry queue, soft timeout limits |
| `memex/engine/core/config.py` | Modify | Add new config keys |
| `memex/engine/core/pipeline.py` | Modify | Increase Qdrant timeout |
| `config.yaml` | Modify | Add new config keys |
| `config.example.yaml` | Modify | Add new config keys |

## Success Criteria

1. **Zero 504 timeouts** — async API eliminates blocking
2. **All files tracked** — status visible in CLI and MCP
3. **Automatic retries** — failed files retried with backoff
4. **Compact CLI** — works in narrow tmux panes
5. **Self-aware systems** — every component knows file state

## Out of Scope

- Background worker process (use sync-time retry instead)
- External queue (Redis/RabbitMQ) — Qdrant payload is sufficient
- Real-time WebSocket updates (polling is sufficient for personal RAG)
