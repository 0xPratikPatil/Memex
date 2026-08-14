# Implementation Plan — LLM Resilience + GPU Coordination + setup.sh Hardening

**Spec:** `docs/superpowers/specs/2026-08-14-llm-resilience-gpu-coordination-design.md`
**Date:** 2026-08-14

## Phase 1: Config keys + LLM resilience layer

### Task 1.1: Add config keys
- Edit `memex/engine/core/config.py`:
  - `LLM_READ_TIMEOUT: float = _cfg_float("llm.read_timeout", 60.0)`
  - `CONTEXT_MAX_BATCHES: int = _cfg_int("contextual_retrieval.max_batches", 8)`
  - `GPU_ENABLED: bool = _cfg_bool("gpu.enabled", True)`
  - `GPU_VRAM_THRESHOLD_MB: int = _cfg_int("gpu.vram_threshold_mb", 6000)`
  - `GPU_MAX_WAIT_S: float = _cfg_float("gpu.max_wait_s", 120.0)`
- Update `config.yaml` + `config.example.yaml` with the same keys + comments

### Task 1.2: Ollama layered timeouts + num_predict
- Edit `memex/engine/llm/ollama.py`:
  - `_get_client()`: replace `httpx.Timeout(self._timeout, connect=10.0)` with
    `httpx.Timeout(connect=10.0, read=config.LLM_READ_TIMEOUT, write=30.0, pool=30.0)`
  - `chat()`: accept `num_predict: int | None = None`, add
    `"num_predict": num_predict` to `options` when provided
- Edit `memex/engine/llm/base.py`:
  - `chat()` abstract signature gains `num_predict: int | None = None`
  - `chat_sync()` forwards it
- Edit `memex/engine/ingestion/context.py`:
  - `_chat(prompt, num_predict=200)` → forward `num_predict=self._llm.chat_sync(prompt, model=model, num_predict=num_predict)`
- Update `tests/unit/test_llm_providers.py`: assert num_predict in request body; assert layered timeout shape

**Checkpoint:** `make lint && make typecheck && make test` — all pass

---

## Phase 2: LLM retry with fast backoff + concise degradation

### Task 2.1: Retry wrapper in base.py
- Edit `memex/engine/llm/base.py`:
  - Add tenacity retry helper `_retry_llm_call(fn, prompt, model, num_predict)`:
    - `retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError))`
    - `stop=stop_after_attempt(3)` (1 initial + 2 retries)
    - `wait=wait_fixed(2)` then 4s — use `wait_exponential(multiplier=2, min=2, max=4)`
    - returns `(result, attempts)` tuple so callers can report attempt count
  - `chat_sync()` uses the wrapper; on final failure re-raises typed error
- Import `httpx` in base.py

### Task 2.2: Concise degradation in context.py
- Edit `memex/engine/ingestion/context.py`:
  - `_batch_context_from_summary` / `_batch_context_from_surrounding`:
    - replace `logger.warning(..., exc_info=True)` with
      `logger.warning("LLM batch context failed after %d attempts — empty contexts", attempts)`
    - full traceback via `logger.debug(..., exc_info=True)`
  - Add `max_batches` cap in `enrich_chunks`: if `len(all_batches) > config.CONTEXT_MAX_BATCHES`,
    process the first `max_batches` with LLM, rest with `_context_from_header` fallback
- Update `tests/unit/test_contextual_retrieval.py`:
  - mock chat to raise TimeoutException twice then succeed → assert retries + success
  - mock chat to always raise → assert ONE warning (no exc_info), empty contexts
  - 12 batches with max_batches=4 → assert 4 LLM calls, 8 header fallbacks

### Task 2.3: Metadata + answer generators same policy
- Edit `memex/engine/metadata/extractor.py`: LLM call wrapped with same retry; degrade path logs concise warning (no exc_info at WARNING)
- Edit `memex/engine/generation/answers.py`: same retry on chat call; refusal/degrade path unchanged but concise warning

**Checkpoint:** `make lint && make typecheck && make test` — all pass

---

## Phase 3: GPU mutual exclusion coordinator

### Task 3.1: Create gpu_lock.py
- Create `memex/engine/utils/gpu_lock.py`:
  - `class GpuLock`:
    - `acquire(owner: str) -> None`:
      - if not `config.GPU_ENABLED` → no-op
      - read VRAM used via `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`
      - if used < `GPU_VRAM_THRESHOLD_MB` → no-op (pass-through)
      - else: unload Ollama models (`POST {OLLAMA_BASE}/api/generate {"model": "<loaded>", "keep_alive": 0}` for each loaded model via `ollama ps`), poll VRAM until below threshold or `GPU_MAX_WAIT_S` elapses
      - on timeout / any exception → `logger.warning(...)` + proceed
    - `release(owner: str) -> None` → no-op (Ollama reloads on demand)
  - module-level `gpu_lock = GpuLock()` singleton
- Create `tests/unit/test_gpu_lock.py`:
  - below threshold → no nvidia-smi call (no-op)
  - over threshold → unload called, poll loop
  - nvidia-smi failure → proceeds with warning
  - max_wait exceeded → warns + proceeds
  - gpu.enabled=false → no-op

### Task 3.2: Wire into marker client
- Edit `memex/engine/ingestion/marker_client.py`:
  - `convert_markdown()`: `gpu_lock.acquire("marker")` before `_submit`, `gpu_lock.release("marker")` in finally
- Update `tests/unit/test_marker_client.py`: assert acquire/release called

### Task 3.3: Wire into pipeline LLM stages + embedding
- Edit `memex/engine/core/pipeline.py` `_ingest_chunks`:
  - wrap context+metadata LLM stage with `gpu_lock.acquire("llm")` / release
- Edit `memex/engine/ingestion/embedding.py` `embed_batch`:
  - `gpu_lock.acquire("embed")` / release
- Update `tests/unit/test_pipeline_chunking.py` + any embedding test: assert lock calls

**Checkpoint:** `make lint && make typecheck && make test` — all pass

---

## Phase 4: setup.sh hardening

### Task 4.1: GPU detection + auto-config
- Edit `setup.sh`:
  - New function `detect_gpu()`: `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits`
    - ≥16384 → write `converter.marker_mode: balanced`, `gpu.enabled: false` into config if missing
    - 8192-16383 → `marker_mode: fast`, `gpu.enabled: true`
    - no GPU → `marker_mode: fast`, `TORCH_DEVICE=cpu` in compose override, warn
  - Only writes config.yaml when the file does not exist (never clobber user config)
  - Show detected GPU + chosen mode in output

### Task 4.2: Pre-flight GPU check
- Edit `setup.sh` after marker health gate:
  - `check_http "marker" "http://localhost:5001/health"` (existing)
  - New: ollama chat smoke test `curl -s -m 5 POST /api/chat` → assert response < 5s
  - New: print GPU summary (VRAM total/used, marker mode, gpu.enabled)
- Verify `setup.sh` runs to completion on this machine (8GB → fast + enabled)

**Checkpoint:** `./setup.sh` completes; `make lint && make test` green

---

## Phase 5: End-to-end verification

### Task 5.1: Forced-stall E2E
- Test script: monkeypatch `OllamaLLM.chat` to sleep 5s then raise `httpx.ReadTimeout` twice, then succeed
  → run `memex ingest README.md` (or direct context gen) → assert: completes <15s, ONE concise warning with attempts, context prefixes present
- Manual: `uv run memex ingest README.md` → context stage completes <10s, no traceback

### Task 5.2: Full validation
- `make lint`, `make typecheck`, `make test` — all green
- `git add -A && git commit` with summary

## Files Modified (summary)

**New:** `memex/engine/utils/gpu_lock.py`, `tests/unit/test_gpu_lock.py`

**Modified:** `config.py`, `config.yaml`, `config.example.yaml`, `llm/ollama.py`, `llm/base.py`, `ingestion/context.py`, `ingestion/embedding.py`, `ingestion/marker_client.py`, `core/pipeline.py`, `metadata/extractor.py`, `generation/answers.py`, `setup.sh`, tests (llm_providers, contextual_retrieval, marker_client, pipeline_chunking, config)

## Estimated Effort

- Phase 1: small (config + timeout plumbing + tests)
- Phase 2: medium (retry wrapper + degradation + tests)
- Phase 3: medium (gpu_lock + wiring + tests)
- Phase 4: medium (setup.sh bash)
- Phase 5: small (E2E + commit)
