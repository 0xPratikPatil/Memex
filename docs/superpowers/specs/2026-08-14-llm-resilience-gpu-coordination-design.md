# LLM Resilience + GPU Coordination + setup.sh Hardening

**Date**: 2026-08-14
**Status**: Approved
**Supersedes**: None (follows the Marker replacement)

## Problem Statement

### Symptom

`memex ingest README.md` produced a `httpx.ReadTimeout` on the Ollama chat call in
`context.py:_batch_context_from_summary` — the call burned the full `http.timeout`
(180s), dumped a 500-line traceback, then degraded the whole document to empty
context prefixes. The same stall previously hit Docling conversions (504s,
"server disconnected") and is a recurring class of failure.

### Root cause

Investigation (all isolated reproductions fast, 0.3-8.5s):

| Test | Result |
|------|--------|
| Direct Ollama chat (single prompt) | 0.3s |
| Realistic 5k-char batch prompt | 1.1s |
| 4 sequential batch calls | 1.3-2.2s each |
| Full README path (parse→summary→31 chunks context) | 8.5s total |
| Thread + event-loop client reuse | no hang |

**The stall is transient resource contention, not a code-speed bug.** On an 8GB
GPU, Marker (3.4GB VRAM during a job) + Ollama (2.3GB: qwen2.5:1.5b chat +
bge-m3 embed) + ml-services (1.4GB) cannot all be resident — when marker grabs
VRAM, Ollama's next call stalls or forces a model reload, eating the entire
180s timeout.

### Real code weaknesses found (independent of the transient)

1. **`num_predict` silently dropped** — `context.py:_chat()` accepts `num_predict`
   but never forwards it to the LLM; Ollama can generate unbounded output.
2. **No retry on LLM calls** — one transient stall → whole document loses
   context (context.py:216 catches and gives up immediately).
3. **Single-blob timeout** — `httpx.Timeout(total, connect=10)` lets one read
   hang for the entire 180s before failing.
4. **500-line traceback on a handled degrade path** — noise, not signal.
5. **No GPU coordination** — marker and Ollama race for VRAM with no policy.
6. **setup.sh is docling-era** — no GPU detection, no auto-config for unknown
   machines, no pre-flight verification of both GPU services.

## Scope

**LLM resilience + GPU mutual exclusion + setup.sh hardening.** (Option A from
brainstorming; user chose this over full job-based LLM architecture.)

## Design

### Section 1 — LLM Resilience Layer

**1.1 `ollama.py` layered timeouts**

Replace the single-blob timeout with phase-split timeouts:

```python
httpx.Timeout(
    connect=10.0,          # TCP connect
    read=LLM_READ_TIMEOUT, # waiting for response body (config, default 60s)
    write=30.0,            # sending request
    pool=30.0,             # waiting for a pooled connection
)
```

Config keys: `llm.timeout` (total budget, default 180s) and `llm.read_timeout`
(default 60s — a single read should not need 180s).

**1.2 Honor `num_predict`**

- `ollama.py` request body adds `"options": {"temperature": 0, "num_predict": num_predict}`.
- `context.py:_chat()` actually forwards its `num_predict` argument (currently
  dropped at the call site).

**1.3 Retry with fast backoff**

- Wrap the chat call in tenacity: **2 retries, 2s/4s backoff**, only on
  `httpx.TimeoutException` / `httpx.TransportError` (transient).
- After retries exhausted, degrade per caller:
  - `context.py`: empty contexts + **one concise warning** (no `exc_info` at
    WARNING; full traceback only at DEBUG), including attempt count.
  - Metadata extractor / answer generator: same retry policy (2 retries, 2s/4s
    backoff) applied at their LLM call sites, then their existing degrade
    paths (skip enrichment / no answer), with the same concise-warning style.

**1.4 Context batch cap**

New config `contextual_retrieval.max_batches` (default 8). Documents producing
more batches than the cap get section-header fallback for the remainder —
prevents pathological docs from triggering dozens of sequential LLM calls.

### Section 2 — GPU Mutual Exclusion Coordinator

**New module: `memex/engine/utils/gpu_lock.py`**

```
GpuLock
  VRAM threshold (config gpu.vram_threshold_mb, default 6000)
  acquire(owner):
    if VRAM used > threshold:
      unload Ollama models (POST /api/generate {keep_alive: 0})
      poll nvidia-smi until VRAM free, bounded by gpu.max_wait_s (default 120)
      on timeout: log warning + proceed (never block ingestion)
  release(owner):
    Ollama reloads on demand (keep_alive=24h container default)
```

**Wiring:**

| Call site | Action |
|-----------|--------|
| `marker_client.convert_markdown()` | `acquire("marker")` before submit; `release` after result/timeout |
| `pipeline._ingest_chunks` LLM stages | `acquire("llm")` around context+metadata LLM calls; release before embedding |
| `embedding.py` batch embed | `acquire("embed")`; release after |

**Behavior:**
- Large cloud GPU (≥16GB): VRAM never exceeds threshold → lock is a no-op
  pass-through, both services run concurrently.
- Small GPU (8GB): first acquirer wins; the other waits (bounded), then
  proceeds with a warning.
- Ollama unload uses the official API (`keep_alive: 0`), models reload on the
  next call.
- Failure-safe: nvidia-smi or Ollama API errors → log + proceed.

Config keys: `gpu.enabled` (true), `gpu.vram_threshold_mb` (6000),
`gpu.max_wait_s` (120).

### Section 3 — setup.sh Hardening

New-machine flow (unknown GPU):

```
1. GPU detection: nvidia-smi memory.total
   ≥16GB  → marker_mode=balanced, gpu.enabled=false
   8-15GB → marker_mode=fast,      gpu.enabled=true
   no GPU → marker_mode=fast, TORCH_DEVICE=cpu, warn
2. Auto-write config.yaml from detected GPU if missing
3. Ollama model pulls (existing, verify)
4. Marker build + health gate (models pre-cached at build → ~15s startup)
5. Pre-flight: ollama chat < 5s AND marker /health 200
```

Failure modes covered: broken driver → CPU fallback + warning; missing Docker →
existing early abort; long first build → progress message + MARKER_MEM_LIMIT.

### Section 4 — Error Handling & Testing

**Error handling:** every stall degrades, never blocks. All handled degrade
paths log one concise `logger.warning("... (attempts=N)")` with `exc_info` only
at DEBUG. GpuLock is best-effort.

**Testing:**

| Test file | Covers |
|-----------|--------|
| `tests/unit/test_gpu_lock.py` *(new)* | acquire/release; below-threshold no-op; unload called over threshold; nvidia-smi failure → proceeds; max_wait → warns+proceeds |
| `tests/unit/test_llm_providers.py` | num_predict sent; layered timeout shape |
| `tests/unit/test_contextual_retrieval.py` | retry 2x then degrade; max_batches cap; concise warning |
| `tests/unit/test_marker_client.py` | convert_markdown acquires+releases lock |
| `tests/unit/test_pipeline_chunking.py` | LLM stages acquire/release |
| `tests/unit/test_config.py` | new key defaults |

Validation: `make lint`, `make typecheck`, `make test` green; `./setup.sh`
dry-run on this machine (8GB → fast + gpu.enabled); manual E2E ingest README →
context completes <10s, no traceback on forced stall.

## Files to Create / Modify

**New:**
- `memex/engine/utils/gpu_lock.py`
- `tests/unit/test_gpu_lock.py`

**Modify:**
- `memex/engine/llm/ollama.py` — layered timeouts, num_predict
- `memex/engine/llm/base.py` — retry wrapper for chat_sync
- `memex/engine/ingestion/context.py` — forward num_predict, concise warnings, max_batches
- `memex/engine/ingestion/embedding.py` — GpuLock around embed
- `memex/engine/ingestion/marker_client.py` — GpuLock around convert
- `memex/engine/core/pipeline.py` — GpuLock around LLM stages
- `memex/engine/core/config.py` — new keys (llm.read_timeout, gpu.*, max_batches)
- `config.yaml` / `config.example.yaml` — new keys
- `setup.sh` — GPU detection + auto-config + pre-flight
- tests listed above

## Success Criteria

1. `memex ingest README.md` completes context generation in <10s (no 180s stall).
2. A forced Ollama stall degrades with ONE concise warning + retry count, no traceback storm.
3. On an 8GB GPU, marker and Ollama never contend (mutual exclusion works).
4. On a ≥16GB cloud GPU, both run concurrently (lock is a no-op).
5. `./setup.sh` on a fresh machine auto-detects GPU and verifies both GPU services.
6. All tests pass; lint + typecheck clean.
