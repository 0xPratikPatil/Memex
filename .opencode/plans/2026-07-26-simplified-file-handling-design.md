# Simplified File Handling Design

## Overview

Remove the file server dependency and implement direct file reading for local usage. This simplifies the architecture, improves performance, and makes the system more portable.

## Current State

### Current Flow
```
MCP Server → HTTP Request → File Server → Volume Mount → Host File
         ↓
      File bytes → Docling API → Parse result
```

### Problems
1. **Complexity**: Extra service to manage (file server)
2. **Performance**: HTTP overhead for file reads
3. **Portability**: Volume mounts tied to local filesystem
4. **Security**: Exposed file server endpoint

## Proposed Architecture

### New Flow
```
MCP Server → pathlib.Path(file_path).read_bytes()
         ↓
      File bytes → Docling API → Parse result
```

### Key Changes

#### 1. Remove File Server
- Delete `src/services/file_server.py`
- Remove fileserver from `docker-compose.yml`
- Remove fileserver build target from `Dockerfile`

#### 2. Update Docling Client
**File: `src/docling_client.py`**

```python
def parse_local_file(file_path: str) -> ConversionResult:
    """Read a local file directly and convert via Docling.
    
    Args:
        file_path: Absolute path on the host (e.g., /mnt/docs/report.pdf)
    """
    from .services.cache import cache_parse_result, get_cached_parse_result
    
    file_hash = hashlib.sha256(file_path.encode()).hexdigest()[:16]
    cached = get_cached_parse_result(file_hash)
    if cached is not None:
        logger.info("Docling cache hit for local file: %s", file_path)
        return ConversionResult(
            markdown=cached["markdown"],
            status=cached.get("status", "success"),
            processing_time=cached.get("processing_time", 0.0),
            errors=cached.get("errors", []),
        )
    
    # Read file directly from filesystem
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_bytes = path.read_bytes()
    filename = path.name
    
    # Convert to Docling format
    b64 = base64.b64encode(file_bytes).decode("ascii")
    
    payload = {
        "options": _build_options(),
        "sources": [
            {
                "kind": "file",
                "base64_content": b64,
                "filename": filename,
            }
        ],
    }
    
    result = _send_to_docling(payload, file_hash)
    cache_parse_result(file_hash, {
        "markdown": result.markdown,
        "status": result.status,
        "processing_time": result.processing_time,
        "errors": result.errors,
    })
    return result
```

#### 3. Update MCP Server Tool
**File: `src/server.py`**

```python
@mcp.tool(
    name="rag_ingest_file",
    title="Ingest File",
    description="Parse and index a local file into the RAG vector database.",
    annotations={"readOnlyHint": False, "openWorldHint": False},
)
async def rag_ingest_file(file_path: str) -> str:
    """Ingest a local file into the RAG knowledge base.
    
    Args:
        file_path: Absolute path to the file (e.g., /mnt/docs/report.pdf)
    
    Returns:
        Confirmation message or error details.
    """
    try:
        from src.docling_client import parse_file
        
        engine = _get_engine()
        
        def _progress(msg: str, pct: int) -> None:
            logger.info("ingest [%d%%] %s", pct, msg)
        
        _progress("Reading file from disk...", 5)
        result = parse_file(file_path)
        
        if not result.ok:
            return f"Error: Docling conversion returned status '{result.status}' with errors: {result.errors}"
        
        _progress("Checking if already ingested...", 10)
        content_hash = engine.compute_file_hash(result.markdown.encode())
        already, chunk_count = engine.is_already_ingested(file_path, content_hash)
        if already:
            return (
                f"Already ingested '{file_path}' "
                f"({chunk_count} chunks, hash: {content_hash[:12]}...). "
                f"File unchanged — skipping."
            )
        
        _progress("Converting with Docling...", 15)
        count = engine.ingest_text(
            result.markdown,
            source_identifier=file_path,
            metadata={
                "content_type": file_path.rsplit(".", 1)[-1] if "." in file_path else "",
                "content_hash": content_hash,
            },
            content_hash=content_hash,
            progress_cb=_progress,
        )
        return (
            f"Successfully ingested '{file_path}'. "
            f"Created {count} chunks. "
            f"(Docling: {result.processing_time:.1f}s, "
            f"{len(result.markdown)} chars, hash: {content_hash[:12]}...)"
        )
    except Exception as exc:
        logger.exception("rag_ingest_file failed")
        return _format_error(exc, f"ingestion of '{file_path}'")
```

#### 4. Simplify Docker Setup
**File: `docker-compose.yml`**

Remove fileserver service:
```yaml
# Remove this section entirely
fileserver:
  build:
    context: .
    dockerfile: Dockerfile
    target: fileserver
  volumes:
    - /mnt:/mnt:ro
    - /home:/home:ro
  networks:
    - backend
  healthcheck:
    test: ["CMD-SHELL", "curl -f http://localhost:9900/health || exit 1"]
    interval: 15s
    timeout: 5s
    retries: 3
    start_period: 5s
  deploy:
    resources:
      limits:
        cpus: "0.5"
        memory: 256M
  restart: unless-stopped
  logging:
    driver: json-file
    options:
      max-size: "10m"
      max-file: "3"
```

Update MCP service to remove fileserver dependency:
```yaml
mcp:
  # ... existing config ...
  # Remove FILE_SERVER_URL from environment
  environment:
    - OLLAMA_HOST=http://ollama:11434
    - QDRANT_URL=http://qdrant:6333
    - DOCLING_URL=http://docling:5001
    # Remove: - FILE_SERVER_URL=http://fileserver:9900
```

**File: `Dockerfile`**

Remove fileserver build target:
```dockerfile
# Remove this entire section
FROM python:3.12-slim AS fileserver
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uvicorn httpx fastapi
COPY services/file_server.py /app/file_server.py
WORKDIR /app
EXPOSE 9900
HEALTHCHECK --interval=15s --timeout=5s --retries=3 --start-period=5s CMD curl -f http://localhost:9900/health || exit 1
CMD ["python", "file_server.py"]
```

#### 5. Update Configuration
**File: `src/config.py`**

Remove file server configuration:
```python
# Remove this setting
FILE_SERVER_URL: str = Field(
    default="http://localhost:9900",
    description="URL of the local file server (for reading files from disk)",
)
```

#### 6. Update Documentation
**Files to update:**
- `README.md` - Remove file server section, update architecture diagram
- `CHANGELOG.md` - Add entry for simplified file handling
- `CONTRIBUTING.md` - Update development setup instructions

## Implementation Steps

### Phase 1: Core Changes
1. **Update `src/docling_client.py`**
   - Modify `parse_local_file()` to read files directly
   - Remove `fetch_file_from_server()` function
   - Add proper error handling for file not found

2. **Update `src/server.py`**
   - Update `rag_ingest_file()` tool documentation
   - Remove file server references

3. **Update `src/config.py`**
   - Remove `FILE_SERVER_URL` setting

### Phase 2: Docker Simplification
4. **Update `Dockerfile`**
   - Remove fileserver build target

5. **Update `docker-compose.yml`**
   - Remove fileserver service
   - Update MCP service environment variables

### Phase 3: Documentation
6. **Update `README.md`**
   - Simplify architecture diagram
   - Update quick start guide
   - Remove file server section

7. **Update `CHANGELOG.md`**
   - Add entry for simplified file handling

8. **Update `CONTRIBUTING.md`**
   - Simplify development setup

## Testing

### Unit Tests
- Test `parse_local_file()` with valid file
- Test `parse_local_file()` with missing file
- Test `parse_local_file()` with permissions error

### Integration Tests
- Test MCP tool `rag_ingest_file()` with local file
- Test end-to-end ingestion flow

### Manual Testing
- Start services: `docker compose up -d`
- Ingest a test file: Use MCP tool with local file path
- Verify chunks are created in Qdrant

## Migration Guide

### For Existing Users
1. **Stop services**: `docker compose down`
2. **Pull latest changes**: `git pull`
3. **Rebuild images**: `docker compose build`
4. **Start services**: `docker compose up -d`
5. **Update MCP configuration**: Remove `FILE_SERVER_URL` if set

### Breaking Changes
- `FILE_SERVER_URL` environment variable no longer used
- File server service removed
- Volume mounts for `/mnt:/mnt:ro` and `/home:/home:ro` no longer needed

## Benefits

1. **Simplicity**: One less service to manage
2. **Performance**: No HTTP overhead for file reads
3. **Portability**: Works on any system with local file access
4. **Security**: No exposed file server endpoint
5. **Easier deployment**: Fewer components to configure

## Risks and Mitigations

### Risk: File Path Validation
- **Mitigation**: Add proper path validation and sanitization
- **Mitigation**: Check file exists before reading

### Risk: Permission Errors
- **Mitigation**: Clear error messages for permission issues
- **Mitigation**: Document required permissions

### Risk: Large Files
- **Mitigation**: Add file size limits
- **Mitigation**: Stream large files if needed

## Success Criteria

1. ✅ File server removed from Docker setup
2. ✅ MCP reads files directly from filesystem
3. ✅ All tests pass
4. ✅ Documentation updated
5. ✅ Migration guide provided

## Timeline

- **Phase 1 (Core Changes)**: 1-2 hours
- **Phase 2 (Docker)**: 30 minutes
- **Phase 3 (Documentation)**: 30 minutes
- **Testing**: 1 hour
- **Total**: ~3 hours

## Future Considerations

### Remote File Support
If remote file support is needed in the future:
1. Add base64 content parameter to MCP tool
2. Implement hybrid approach (local + remote)
3. Keep file server as optional component

### File Watching
For automatic ingestion of new files:
1. Add file watcher service
2. Monitor directories for changes
3. Auto-ingest new files
