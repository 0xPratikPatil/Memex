#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Memex Bootstrap — one command to ready everything.
#
#   ./setup.sh                      # use defaults
#   EMBED_MODEL=llama3.2:1b ./setup.sh   # custom embedding model
#   CHAT_MODEL=qwen2:0.5b ./setup.sh     # custom chat model
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
CHAT="${CHAT_MODEL:-qwen2:0.5b}"
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
echo "[1/6] Docker"
docker info >/dev/null 2>&1 || fail "Docker not running"
ok "running"

# ── 2. Start services ───────────────────────────────────────────────────────
echo "[2/6] Services"
docker compose up -d --build --remove-orphans
ok "started"

# ── 3. Health checks ────────────────────────────────────────────────────────
echo "[3/6] Health checks"
for svc in "${BOOT_SERVICES[@]}"; do
    if ! docker compose ps -q "$svc" &>/dev/null; then
        info "${svc}: not in compose, skipping"
        continue
    fi
    while ! docker compose ps "$svc" 2>/dev/null | tail -n+2 | grep -q "healthy"; do
        sleep 2
    done
    ok "${svc}"
done

# ── 4. Pull models ──────────────────────────────────────────────────────────
echo "[4/6] Models"
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

# ── 5. Verify ────────────────────────────────────────────────────────────────
echo "[5/6] Verify"
curl -sf http://localhost:5002/health >/dev/null && ok "ml-services" || fail "ml-services unreachable"

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
## These need the chat model (pulled above):
##   ENABLE_QUERY_EXPANSION=true  ENABLE_HYDE=true
##   ENABLE_CONTEXTUAL_RETRIEVAL=true
##   ENABLE_METADATA_EXTRACTION=true
## See .env.example for all options."
