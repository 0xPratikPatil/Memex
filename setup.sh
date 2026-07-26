#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Memex Bootstrap — one command to ready everything
#
# Reads models from environment, falls back to defaults.
#   EMBED_MODEL=bge-m3 ./setup.sh     # custom embedding model
#   CHAT_MODEL=llama3.2:3b ./setup.sh # custom chat model
#   ./setup.sh                        # use defaults
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")"

# ── Config (env vars take priority) ────────────────────────────────────────
source_env() {  # load .env if it exists (doesn't override existing env vars)
    if [ -f ".env" ]; then
        set -a; source .env; set +a
    fi
}
source_env

# Models — env var > .env > default
EMBED_MODEL="${EMBED_MODEL:-bge-m3}"
CHAT_MODEL="${CHAT_MODEL:-qwen2:0.5b}"
RERANK_MODEL="${RERANK_MODEL:-BAAI/bge-reranker-base}"
SPARSE_MODEL="${SPARSE_MODEL:-Qdrant/bm25}"

# Services expected to be healthy
SERVICES=(qdrant ollama docling ml-services redis)

# ── Banner ─────────────────────────────────────────────────────────────────
echo "╔════════════════════════════════╗"
echo "║       Memex Bootstrap         ║"
echo "╚════════════════════════════════╝"
echo ""
echo "  Embed:  ${EMBED_MODEL}"
echo "  Chat:   ${CHAT_MODEL}"
echo "  Rerank: ${RERANK_MODEL} (in ml-services)"
echo "  Sparse: ${SPARSE_MODEL} (in ml-services)"
echo ""

# ── Step 1: Check Docker ──────────────────────────────────────────────────
echo "[1/5] Checking Docker..."
if ! docker info >/dev/null 2>&1; then
    echo "  ERROR: Docker is not running. Start Docker first."
    exit 1
fi
echo "  ✓ OK"

# ── Step 2: Start services ────────────────────────────────────────────────
echo "[2/5] Starting services..."
docker compose up -d --build --remove-orphans
echo "  ✓ Started"

# ── Step 3: Wait for healthy ──────────────────────────────────────────────
echo "[3/5] Waiting for services..."
for svc in "${SERVICES[@]}"; do
    printf "  %-15s " "${svc}:"
    if ! docker compose ps -q "$svc" &>/dev/null; then
        echo "SKIP (not in compose)"
        continue
    fi
    until docker compose ps "$svc" 2>/dev/null | tail -n +2 | grep -q "healthy"; do
        sleep 2
    done
    echo "✓"
done
echo "  ✓ All services healthy"

# ── Step 4: Pull models into Ollama ───────────────────────────────────────
echo "[4/5] Pulling models..."
pull_if_needed() {
    local model="$1"
    if docker compose exec -T ollama ollama list 2>/dev/null | grep -q "$model"; then
        echo "  → ${model} (already present)"
    else
        echo "  → ${model} (pulling...)"
        docker compose exec -T ollama ollama pull "$model"
        echo "  → ${model} ✓"
    fi
}
pull_if_needed "$EMBED_MODEL"
pull_if_needed "$CHAT_MODEL"
echo "  ✓ Models ready"

# ── Step 5: Verify ML services ────────────────────────────────────────────
echo "[5/5] Verifying ML services..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:5002/health 2>/dev/null | grep -q "200"; then
    echo "  ✓ ML services responded"
else
    echo "  ⚠ ML services still loading (check: docker logs rag-ml-services-1)"
fi

# ── Done ──────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════╗"
echo "║     Bootstrap Complete              ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "Services:"
docker compose ps --format "table {{.Service}}\t{{.Status}}" 2>/dev/null
echo ""
echo "  Run MCP:  uv run memex"
echo ""
