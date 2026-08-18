# Conditional Docker Services + Lightweight OCR Fallback

**Date**: 2026-08-18
**Status**: Approved
**Scope**: Two subsystems — (1) setup.sh conditional service startup, (2) OCR fallback service

---

## 1. Problem Statement

### 1.1 setup.sh starts all Docker services unconditionally

`setup.sh` line 539 runs `docker compose up -d --build --remove-orphans` which starts all 6 services (qdrant, ollama, marker, markitdown, redis, ml-services) regardless of which `converter.engine` is configured. When using `markitdown`, the marker and ml-services containers waste ~2GB VRAM and CPU. Health checks also always probe marker even when it's not needed.

**Impact**: Unnecessary resource usage, slower setup, confusing health check failures for services that shouldn't be running.

### 1.2 Scanned PDFs OOM on 8GB GPU

Marker uses Surya OCR which needs >8GB VRAM for scanned PDFs. On an 8GB RTX 3070 (shared with desktop), documents like `Deed-of-Family-Trust.pdf` and `Wealth_How_the_Worlds_High-Net-Worth_Grow.pdf` fail with OOM. No fallback mechanism exists — the user must manually retry on a cloud GPU or accept failure.

**Impact**: ~5 scanned PDFs permanently failed in the collection, no way to process them on local hardware.

---

## 2. Design: Conditional Docker Services

### 2.1 Approach

Read `converter.engine` from `config.yaml` via the existing `_read_config_model()` helper. Build a dynamic service list that includes only the services required for the configured engine.

### 2.2 Service Matrix

| Service | Port | marker | markitdown | Always |
|---------|------|--------|------------|--------|
| qdrant | 6333 | yes | yes | yes |
| ollama | 11434 | yes | yes | yes |
| redis | 6379 | yes | yes | yes |
| marker | 5001 | yes | no | no |
| ml-services | 5002 | yes | no | no |
| markitdown | 5003 | no | yes | no |

### 2.3 Implementation

Replace the hardcoded `BOOT_SERVICES` array in `setup.sh`:

```bash
CONVERTER=$(_read_config_model "converter.engine" "CONVERTER_ENGINE" "marker")

# Base services — always needed
SERVICES=(qdrant ollama redis)

# Converter-specific services
case "$CONVERTER" in
    marker)
        SERVICES+=(marker ml-services)
        ;;
    markitdown)
        SERVICES+=(markitdown)
        ;;
    docling)
        # Legacy: still needs marker for conversion
        SERVICES+=(marker ml-services)
        ;;
    *)
        info "unknown converter engine '$CONVERTER' — starting all services"
        SERVICES+=(marker ml-services markitdown)
        ;;
esac
```

Health check loop iterates over `$SERVICES` instead of the hardcoded array. The final `docker compose up` passes only the needed service names:

```bash
docker compose up -d --build --remove-orphans "${SERVICES[@]}"
```

### 2.4 Files Modified

- `setup.sh` — replace hardcoded `BOOT_SERVICES` with dynamic list, update health checks

---

## 3. Design: Lightweight OCR Fallback Service

### 3.1 Architecture

New standalone Docker service following the Marker job-based pattern:

```
MCP Server (host)
  │
  ├─ HTTP ──► Docker: Marker (:5001)
  │             └─ If OOM on scanned PDF:
  │                 └─ HTTP ──► Docker: OCR Service (:5004)
  │                               └─ PP-OCRv6 small / LightOnOCR-2-1B
  │
  ├─ HTTP ──► Docker: MarkItDown (:5003)  [if configured]
  │
  └─ ... other services
```

### 3.2 OCR Service

**Container**: `ocr.Dockerfile` based on `python:3.12-slim` with `onnxruntime-gpu` + `transformers`.

**Server**: `ocr_server.py` — FastAPI with three endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/convert` | POST | Accept PDF pages as images, return markdown |
| `/health` | GET | Service health + current model + VRAM usage |
| `/model/swap` | POST | Hot-swap the active OCR model |

**Model loading**:
- Default: `PP-OCRv6 small` (34.5M params, ~500MB VRAM, ONNX)
- Optional: `LightOnOCR-2-1B` (1B params, ~2GB VRAM, PyTorch)
- Models loaded on startup, cached in Docker volume `ocr_models`
- Hot-swap via `/model/swap` — unloads current, loads new

**API contract**:

```
POST /convert
  Request:  multipart/form-data (files: list of page images)
  Response: {
    "markdown": "extracted text...",
    "pages": [{"page": 1, "text": "...", "confidence": 0.95}],
    "model": "pp-ocrv6-small",
    "processing_time": 1.2
  }

GET /health
  Response: {
    "status": "ok",
    "model": "pp-ocrv6-small",
    "vram_mb": 512,
    "uptime_s": 3600
  }

POST /model/swap
  Request:  {"model": "lightonocr-2-1b"}
  Response: {"status": "ok", "previous": "pp-ocrv6-small", "current": "lightonocr-2-1b"}
```

### 3.3 Fallback Flow

In `marker_client.py`, the `convert_markdown()` function gains OOM detection:

```python
def convert_markdown(file_bytes: bytes, filename: str) -> MarkerResult:
    try:
        # ... existing Marker submission logic ...
        return _poll_and_fetch(job_id, filename)
    except ConversionError as exc:
        if not _is_oom_error(exc):
            raise
        if not config.OCR_FALLBACK_ENABLED:
            raise
        logger.warning("Marker OOM, falling back to OCR service",
                       extra={"source": filename, "stage": "Converting"})
        return _ocr_fallback(file_bytes, filename)
```

`_is_oom_error()` checks the error message for CUDA OOM patterns (`memory allocation failed`, `CUDACachingAllocator`).

`_ocr_fallback()` sends the PDF to the OCR service via `ocr_client.py`, which follows the same HTTP client pattern as `marker_client.py` (tenacity retry, structured errors).

### 3.4 Config Additions

```yaml
converter:
  # ... existing keys ...
  ocr_fallback: true                    # enable OCR fallback on Marker OOM
  ocr_url: "http://localhost:5004"      # OCR service URL
  ocr_model: "pp-ocrv6-small"           # default OCR model (pp-ocrv6-small | lightonocr-2-1b)
  ocr_timeout: 120.0                    # seconds per conversion
```

### 3.5 VRAM Budget (8GB GPU)

| Component | VRAM | Notes |
|-----------|------|-------|
| Ollama (embed + chat) | ~4.5GB | qwen3-embedding:0.6b + qwen2.5:1.5b |
| OCR service (PP-OCRv6 small) | ~0.5GB | ONNX, stays loaded |
| Marker (fast mode, no OCR) | ~3GB | Layout + VLM only, no Surya |
| **Total** | **~8GB** | Fits within 8GB with ~0.5GB headroom |

When Marker is actively converting, GpuLock evicts Ollama (existing behavior). The OCR service's 0.5GB footprint is small enough to coexist with Marker's layout models.

### 3.6 Files Created

- `ocr.Dockerfile` — container definition
- `ocr_server.py` — FastAPI server
- `memex/engine/ingestion/ocr_client.py` — HTTP client

### 3.7 Files Modified

- `marker_client.py` — add OOM detection + fallback call
- `docker-compose.yml` — add `ocr` service block
- `config.yaml` — add `ocr_fallback`, `ocr_url`, `ocr_model`, `ocr_timeout`
- `config.example.yaml` — add OCR config example
- `setup.sh` — conditional service startup (section 2)
- `AGENTS.md` — document OCR fallback feature

---

## 4. OCR Model Research Summary

Models evaluated for 8GB GPU fallback:

| Model | Params | VRAM | Accuracy | Speed | License |
|-------|--------|------|----------|-------|---------|
| **PP-OCRv6 small** | 34.5M | ~500MB | 81.3% Hmean | 80+ pg/min | Apache 2.0 |
| PP-OCRv6 tiny | 1.1M | ~200MB | 73.5% Hmean | 100+ pg/min | Apache 2.0 |
| **LightOnOCR-2-1B** | 1B | ~2GB | SOTA | 5.7 pg/s (H100) | Apache 2.0 |
| GRM-OCR | 300M | ~1GB | Strong | Fast | Apache 2.0 |
| XCurOS-OCR | 0.9B | CPU-only | Good | Moderate | Apache 2.0 |
| HunyuanOCR-1.5 | 1.5B | ~2GB | Excellent | Fast (DFlash) | Tencent |

**Selected**: PP-OCRv6 small (default, fits alongside Ollama) + LightOnOCR-2-1B (on-demand for highest accuracy).

**Rationale**: PP-OCRv6 small has the best accuracy/VRAM ratio. At 500MB VRAM, it coexists with Ollama on 8GB. LightOnOCR-2-1B is available for cases where accuracy matters more than VRAM (e.g., when Ollama is temporarily evicted).

---

## 5. Testing Strategy

### 5.1 Unit Tests

- `test_ocr_client.py` — mock HTTP responses, verify retry/error handling
- `test_ocr_server.py` — TestClient for /convert, /health, /model/swap endpoints
- `test_marker_client.py` — add tests for OOM detection + fallback path
- `test_convert_one.py` — add tests for `build_converter_args` with OCR config

### 5.2 Integration Tests

- Start OCR Docker service, verify /health returns ok
- Send a test PDF page image to /convert, verify markdown output
- Test model swap from PP-OCRv6 small to LightOnOCR-2-1B

### 5.3 E2E Test

- Ingest a scanned PDF that would OOM on Marker
- Verify fallback triggers and file is ingested via OCR service
- Query the ingested content to verify quality

---

## 6. Implementation Order

1. **setup.sh conditional services** — small, self-contained, immediate value
2. **OCR Docker service** — ocr.Dockerfile + ocr_server.py + health check
3. **OCR client** — memex/engine/ingestion/ocr_client.py
4. **Marker fallback integration** — OOM detection + fallback in marker_client.py
5. **Config + docker-compose** — add OCR config keys + service block
6. **Tests** — unit + integration + E2E
7. **Docs** — update AGENTS.md + config.example.yaml
