#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Memex Bootstrap — one command to ready everything.
#
#   ./setup.sh                              # use defaults
#   EMBED_MODEL=llama3.2:1b ./setup.sh      # custom embedding model
#   CHAT_MODEL=qwen3.5:0.8b ./setup.sh      # custom chat model
#
# What it does:
#   1. Creates .env from .env.example if missing
#   2. Installs Python deps (uv sync) into project .venv
#   3. Checks Docker is running
#   4. Builds and starts all backend services
#   5. Waits for health checks
#   6. Pulls Ollama models (skips if already present)
#   7. Verifies models + features respond
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")"

# ── Load .env if it exists (does not override existing env vars) ────────────
# Only export vars the shell script needs — Python reads .env via python-dotenv.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Unset vars that confuse uv (uv chokes on non-integer env vars like HTTP_TIMEOUT=60.0)
unset HTTP_TIMEOUT DOCLING_TIMEOUT QDRANT_TIMEOUT 2>/dev/null || true

# ── Models (env var > .env > default) ───────────────────────────────────────
EMBED="${EMBED_MODEL:-qwen3-embedding:0.6b}"
CHAT="${CHAT_MODEL:-qwen3.5:0.8b}"
RERANK="${RERANK_MODEL:-Qwen/Qwen3-Reranker-0.6B}"
SPARSE="${SPARSE_MODEL:-Qdrant/bm25}"

BOOT_SERVICES=(qdrant ollama docling ml-services redis)

# ── Helpers ──────────────────────────────────────────────────────────────────
ok()   { echo "  ✓ $1"; }
fail() { echo "  ✗ $1"; exit 1; }
info() { echo "  → $1"; }

# ── Banner ──────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║            Memex Bootstrap              ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  embed   ${EMBED}"
echo "  chat    ${CHAT}"
echo "  rerank  ${RERANK}"
echo "  sparse  ${SPARSE}"
echo ""

# ── 0. Create .env if missing ────────────────────────────────────────────────
if [ ! -f .env ]; then
    cp .env.example .env
    info "created .env from .env.example"
fi

# ── 1. Python environment ──────────────────────────────────────────────────
echo "[1/8] Python environment"
if uv sync; then
    ok "deps installed"
else
    info "Python deps failed — run 'uv sync' manually later"
fi

# ── 2. Docker ───────────────────────────────────────────────────────────────
echo "[2/8] Docker"
docker info >/dev/null 2>&1 || fail "Docker not running"
ok "running"

# ── 3. Start services ───────────────────────────────────────────────────────
echo "[3/8] Services"
docker compose up -d --build --remove-orphans
ok "started"

# ── 4. Health checks ────────────────────────────────────────────────────────
echo "[4/8] Health checks"

check_http() {
    local name="$1" url="$2"
    while ! curl -sf --max-time 3 "$url" >/dev/null 2>&1; do sleep 2; done
    ok "$name"
}

for svc in "${BOOT_SERVICES[@]}"; do
    if ! docker compose ps -q "$svc" &>/dev/null; then
        info "${svc}: not in compose, skipping"
        continue
    fi
    while ! docker compose ps "$svc" 2>/dev/null | tail -n+2 | grep -q "healthy"; do
        sleep 2
    done
done

check_http "qdrant"      "http://localhost:6333/"
check_http "ollama"      "http://localhost:11434/api/tags"
check_http "docling"     "http://localhost:5001/health"
check_http "ml-services" "http://localhost:5002/health"
ok "redis         (Docker healthcheck)"

# ── 5. Pull models ──────────────────────────────────────────────────────────
echo "[5/8] Models"
pull() {
    local m="$1"
    if docker compose exec -T ollama ollama list 2>/dev/null | grep -q "$m"; then
        info "$m (cached)"
    else
        info "$m (downloading…)"
        docker compose exec -T ollama ollama pull "$m"
    fi
}
pull "$EMBED"
pull "$CHAT"
ok "ready"

# ── 6. Verify models loaded ─────────────────────────────────────────────────
echo "[6/8] Verify models"
curl -sf -X POST http://localhost:11434/api/embeddings \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${EMBED}\",\"prompt\":\"test\"}" >/dev/null \
    && ok "${EMBED}" || fail "${EMBED} not responding"
curl -sf -X POST http://localhost:11434/api/chat \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${CHAT}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}" >/dev/null \
    && ok "${CHAT}" || fail "${CHAT} not responding"

# ── 7. Verify features ─────────────────────────────────────────────────────
echo "[7/8] Features"
# Redis cache ping
if docker compose exec -T redis redis-cli ping | grep -q PONG 2>/dev/null; then
    echo "  ✓ redis cache"
else
    echo "  ✗ redis cache (non-fatal)"
fi
# Hybrid chunker availability
if uv run python -c "
from rag.chunking import is_hybrid_chunker_available
ok = is_hybrid_chunker_available()
assert ok, 'HybridChunker not available — check docling install'
" 2>&1; then
    echo "  ✓ hybrid chunker"
else
    echo "  ✗ hybrid chunker (non-fatal) — run: uv sync"
fi

# ── Done ────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║           Ready — run the MCP            ║"
echo "╚══════════════════════════════════════════╝"
echo ""
docker compose ps --format "table {{.Service}}\t{{.Status}}" 2>/dev/null
echo ""
echo "  uv run memex"
echo ""
