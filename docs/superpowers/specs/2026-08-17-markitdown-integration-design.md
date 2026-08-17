# MarkItDown Integration Design

**Date:** 2026-08-17  
**Status:** Approved  
**Approach:** Mirror Marker's Docker architecture (Approach 1)

---

## Goal

Add Microsoft's MarkItDown as an alternative document converter alongside Marker. MarkItDown handles PDF, DOCX, PPTX, XLSX, HTML, EPUB, images, audio, CSV, JSON, XML, and ZIP — converting them to Markdown on CPU with no GPU required. This gives Memex broad format coverage and fast conversion for simple documents, while Marker remains the choice for complex/scanned PDFs.

## Requirements

- `converter.engine` config switch: `marker` (default) or `markitdown`
- MarkItDown deployed as Docker container (consistent with existing architecture)
- Parallel file ingestion — MarkItDown is CPU-only, no GPU contention with Marker
- MarkItDown handles all formats when selected
- No changes to chunking, embedding, context, metadata, or Qdrant layers

## Architecture

### Data Flow

```
CLI ingest / sync
  │
  ├── loader.py: parse_file()
  │     │
  │     ├── engine == "marker"
  │     │     └── marker_client.convert_markdown()
  │     │           GpuLock → Docker marker:5001 → subprocess → Marker models
  │     │
  │     └── engine == "markitdown"
  │           └── markitdown_client.convert_markdown()
  │                 Docker markitdown:5003 → MarkItDown().convert()
  │
  └── pipeline.py: ingest_text()
        chunk → context → metadata → embed → Qdrant (unchanged)
```

### Key Differences from Marker

| Aspect | Marker | MarkItDown |
|--------|--------|------------|
| GPU | Required (4GB+ VRAM) | Not needed (CPU-only) |
| GpuLock | Yes — mutual exclusion with Ollama | None |
| Server isolation | Subprocess per job (crash-proof) | In-process (safe, no GPU state) |
| Concurrency | Semaphore(max_concurrent=2) | Unlimited |
| Speed | 2.9-7.4 pages/sec (GPU) | Sub-second per page (CPU) |
| Image quality | High (ML layout + OCR) | Lower (text stream extraction) |
| Formats | PDF-first | PDF, DOCX, PPTX, XLSX, HTML, EPUB, images, audio, CSV, JSON, XML, ZIP |

## Components

### 1. Docker Container

**`markitdown.Dockerfile`:**

```dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir 'markitdown[all]'
COPY markitdown_server.py /app/
EXPOSE 5003
CMD ["uvicorn", "markitdown_server:app", "--host", "0.0.0.0", "--port", "5003"]
```

~200MB image. No GPU, no models to download.

**`markitdown_server.py`** (~80 lines):

Thin FastAPI server with two endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/convert` | POST | Accept file bytes + filename, return markdown |
| `/health` | GET | Health check |

The `/convert` endpoint:
1. Receives file bytes + filename as multipart form data
2. Instantiates `MarkItDown()` and calls `.convert(file_bytes)`
3. Returns JSON: `{markdown, metadata, format, processing_time}`
4. No subprocess needed — MarkItDown is safe to run in-process (CPU-only, no GPU state, won't crash the server)

**Why no subprocess isolation?** Marker spawns subprocesses because GPU model loading can OOM and kill the process. MarkItDown has no GPU models — a single instance handles all conversions safely. Failures raise exceptions, not crashes.

**docker-compose.yml addition:**

```yaml
markitdown:
  build:
    context: .
    dockerfile: markitdown.Dockerfile
  container_name: memex-markitdown
  ports:
    - "127.0.0.1:5003:5003"
  restart: unless-stopped
  healthcheck:
    test: ["CMD", "python", "-c", "import httpx; httpx.get('http://localhost:5003/health').raise_for_status()"]
    interval: 30s
    timeout: 5s
    start_period: 10s
    retries: 3
  deploy:
    resources:
      limits:
        memory: 1G
```

### 2. Client Module

**`memex/engine/ingestion/markitdown_client.py`** (~60 lines):

Mirrors `marker_client.py` but much simpler — no GpuLock, no polling, no subprocess lifecycle.

```python
@dataclass
class MarkItDownResult:
    markdown: str
    metadata: dict
    format: str           # "pdf", "docx", "pptx", etc.
    processing_time: float

async def convert_markdown(file_bytes: bytes, filename: str) -> MarkItDownResult:
    """Convert a file to Markdown via the MarkItDown Docker service."""
    # 1. POST file bytes to http://localhost:5003/convert
    # 2. Parse JSON response
    # 3. Return MarkItDownResult
```

**Comparison with marker_client.py:**

| Aspect | Marker Client | MarkItDown Client |
|--------|--------------|-------------------|
| GPU lock | Acquires/releases GpuLock | None needed |
| Submit pattern | Async job (submit → poll → fetch) | Synchronous single request |
| Concurrency | Semaphore(max_concurrent=2) | Unlimited |
| Timeout | 300s (configurable) | 30s (sub-second per page) |
| Retry | On OOM/timeout | On connection error |

### 3. Config Changes

**config.yaml:**

```yaml
converter:
  engine: marker              # marker | markitdown
  marker_url: "http://localhost:5001"
  marker_mode: fast
  marker_force_ocr: false
  marker_timeout: 300.0
  max_concurrent: 2
  markitdown_url: "http://localhost:5003"   # NEW
  markitdown_timeout: 30.0                   # NEW (seconds)
```

**`config.py` constants to add:**

```python
MARKITDOWN_URL = "markitdown_url"
MARKITDOWN_TIMEOUT = "markitdown_timeout"
```

**Validation rules:**
- `converter.engine` accepts `marker` or `markitdown`
- `markitdown_url` only validated when `engine == markitdown`
- `markitdown_timeout` defaults to 30s if unset

### 4. Loader Integration

**`loader.py`** routing in `parse_file()`:

```python
async def parse_file(file_path: str = None, file_bytes: bytes = None, ...):
    if engine == "marker":
        result = await marker_client.convert_markdown(file_bytes, filename)
    elif engine == "markitdown":
        result = await markitdown_client.convert_markdown(file_bytes, filename)
    else:
        raise ConfigError(f"Unknown converter engine: {engine}")
    
    return ConversionResult(
        markdown=result.markdown,
        source=source,
        metadata={**base_metadata, **result.metadata},
        content_hash=hashlib.sha256(result.markdown.encode()).hexdigest(),
    )
```

**Uniform interface** — both clients return the same shape, so the rest of the pipeline (chunking, context, metadata, embedding, Qdrant) works unchanged.

**File type handling:**
- MarkItDown supports PDF, DOCX, PPTX, XLSX, HTML, EPUB, images, audio, CSV, JSON, XML, ZIP
- The loader already detects file types by extension — no change needed
- Unknown extensions get rejected before reaching the converter

## Parallel Ingestion

MarkItDown is CPU-only with no GPU contention. Multiple files convert concurrently:

```
File 1 (DOCX) ──→ markitdown_client ──→ Docker markitdown ──→ chunk → embed → Qdrant
File 2 (PDF)  ──→ markitdown_client ──→ Docker markitdown ──→ chunk → embed → Qdrant
File 3 (PPTX) ──→ markitdown_client ──→ Docker markitdown ──→ chunk → embed → Qdrant
File 4 (XLSX) ──→ markitdown_client ──→ Docker markitdown ──→ chunk → embed → Qdrant
```

- No semaphore needed — the FastAPI server handles backpressure
- `loader.py` already runs `parse_file()` concurrently for multiple files
- `embedding.batch_size` (default 64) controls embedding parallelism
- Conversion is not the bottleneck — embedding (Ollama) is

## Error Handling

| Error | Exception | Hint |
|-------|-----------|------|
| Container unreachable | `ServiceUnavailableError` | "docker compose up markitdown" |
| Conversion timeout | `ConversionTimeoutError` | "Increase markitdown_timeout or check file size" |
| Unsupported format | `ConversionError` | "Format X not supported by MarkItDown" |
| Empty output | `CorruptedDocumentError` | "Document may be empty or corrupt" |
| Server 500 | `ConversionError` | Include response body |

**Retry behavior:**
- Connection errors: 3 attempts, exponential backoff
- Conversion errors: no retry (file is corrupt or unsupported)
- Timeout: no retry (file too large or server stuck)

**Logging:**

```python
logger.info("MarkItDown conversion complete", extra={
    "source": filename,
    "stage": "Converting",
    "format": result.format,
    "chars": len(result.markdown),
    "time": f"{result.processing_time:.1f}s",
})
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `markitdown.Dockerfile` | Create | Docker image for MarkItDown service |
| `markitdown_server.py` | Create | FastAPI server (~80 lines) |
| `memex/engine/ingestion/markitdown_client.py` | Create | Client module (~60 lines) |
| `memex/engine/core/config.py` | Modify | Add MARKITDOWN_URL, MARKITDOWN_TIMEOUT constants |
| `memex/engine/ingestion/loader.py` | Modify | Add markitdown routing in parse_file() |
| `config.yaml` | Modify | Add markitdown_url, markitdown_timeout |
| `config.example.yaml` | Modify | Add markitdown config example |
| `docker-compose.yml` | Modify | Add markitdown service |
| `tests/unit/test_markitdown_client.py` | Create | Client unit tests |
| `tests/unit/test_markitdown_server.py` | Create | Server unit tests |

## Testing Strategy

1. **Unit tests**: Mock HTTP client, test result parsing, error handling
2. **Integration tests**: Docker service running, convert a real DOCX/PDF
3. **E2E**: `memex ingest /path/to/file.docx` with `converter.engine: markitdown`
4. **Parallel test**: Ingest multiple files simultaneously, verify all complete
5. **Fallback test**: Verify Marker still works when `converter.engine: marker`

## Out of Scope

- Auto-routing (detect file type and pick converter automatically) — future enhancement
- Dual conversion (run both, pick better result) — not needed now
- Two-pass (fast MarkItDown then upgrade with Marker) — not needed now
- MarkItDown OCR plugin (requires LLM API key) — can add later
- Azure Document Intelligence integration — can add later
