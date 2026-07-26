#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Memex Bootstrap — one command to ready everything.
#
#   ./setup.sh                      # use defaults
#   EMBED_MODEL=llama3.2:1b ./setup.sh   # custom embedding model
#   CHAT_MODEL=qwen2.5:0.5b ./setup.sh     # custom chat model
#
# What it does:
#   1. Checks Docker is running
#   2. Builds and starts all backend services
#   3. Waits for health checks
#   4. Pulls Ollama models (skips if already present)
#   5. Verifies ML services respond
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")"

# ── Load .env if it exists (does not override existing env vars) ────────────
if [ -f .env ]; then set -a; source .env; set +a; fi

# ── Models (env var > .env > default) ───────────────────────────────────────
EMBED="${EMBED_MODEL:-bge-m3}"
CHAT="${CHAT_MODEL:-qwen2.5:0.5b}"
RERANK="${RERANK_MODEL:-BAAI/bge-reranker-base}"
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

# ── 1. Docker ───────────────────────────────────────────────────────────────
echo "[1/7] Docker"
docker info >/dev/null 2>&1 || fail "Docker not running"
ok "running"

# ── 2. Start services ───────────────────────────────────────────────────────
echo "[2/7] Services"
docker compose up -d --build --remove-orphans
ok "started"

# ── 3. Health checks ────────────────────────────────────────────────────────
echo "[3/7] Health checks"

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
    # Wait for Docker health check first
    while ! docker compose ps "$svc" 2>/dev/null | tail -n+2 | grep -q "healthy"; do
        sleep 2
    done
done

# Verify actual endpoints respond
check_http "qdrant"     "http://localhost:6333/"
check_http "ollama"     "http://localhost:11434/api/tags"
check_http "docling"    "http://localhost:5001/health"
check_http "ml-services" "http://localhost:5002/health"
ok "redis         (Docker healthcheck)"

# ── 4. Pull models ──────────────────────────────────────────────────────────
echo "[4/7] Models"
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

# ── 5. Verify models loaded ─────────────────────────────────────────────────
echo "[5/7] Verify models"
# Test embedding model responds
curl -sf -X POST http://localhost:11434/api/embeddings \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${EMBED}\",\"prompt\":\"test\"}" >/dev/null \
    && ok "${EMBED}" || fail "${EMBED} not responding"
# Test chat model responds
curl -sf -X POST http://localhost:11434/api/chat \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${CHAT}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}" >/dev/null \
    && ok "${CHAT}" || fail "${CHAT} not responding"

# ── 6. Verify advanced features ───────────────────────────────────────────────
echo "[6/7] Advanced features"
# Redis cache ping (informational only — won't stop bootstrap)
if docker compose exec -T redis redis-cli ping | grep -q PONG 2>/dev/null; then
    echo "  ✓ redis cache ping"
else
    echo "  ✗ redis cache ping (non-fatal)"
fi
# Hybrid chunker availability (informational only — won't stop bootstrap)
if uv sync 2>/dev/null && uv run python -c "from rag.chunking import is_hybrid_chunker_available; assert is_hybrid_chunker_available(), 'not available'" 2>/dev/null; then
    echo "  ✓ hybrid chunker"
else
    echo "  ✗ hybrid chunker (non-fatal, run: uv sync)"
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

## ── Advanced Features ─────────────────────────────────────────────────
## Advanced features (query expansion, contextual retrieval, metadata
##   extraction) are enabled by default. Their env vars are already set
##   to true in .env.example — no action needed unless you want to
##   disable them. Run `make dev` to install all dependencies."
