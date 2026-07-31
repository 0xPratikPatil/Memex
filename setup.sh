#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Memex Bootstrap — one command to ready everything.
#
#   ./setup.sh                              # use config.yaml defaults
#   EMBED_MODEL=llama3.2:1b ./setup.sh      # override embedding model
#   CHAT_MODEL=qwen2.5:1.5b ./setup.sh      # override chat model
#
# What it does:
#   1. Creates .env from .env.example if missing (secrets only)
#   2. Creates config.yaml from config.example.yaml if missing
#   3. Installs Python deps (uv sync) into project .venv
#   4. Checks Docker is running
#   5. Builds and starts all backend services
#   6. Waits for health checks
#   7. Pulls Ollama models (reads from config.yaml, env var overrides work)
#   8. Verifies models + features respond
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

# ── Models (env var > config.yaml > default) ────────────────────────────────
# config.yaml is single source of truth; env vars allow ad-hoc overrides.
_read_config_model() {
    local yaml_path="$1" env_var="$2" default="$3"
    if [ -n "${!env_var:-}" ]; then
        echo "${!env_var}"
        return
    fi
    if [ -f config.yaml ] && command -v python3 &>/dev/null; then
        local val
        val=$(python3 -c "
import yaml
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
keys = '$yaml_path'.split('.')
node = cfg
try:
    for k in keys:
        node = node[k]
    print(node)
except (KeyError, TypeError):
    print('')
" 2>/dev/null)
        if [ -n "$val" ] && [ "$val" != "None" ] && [ "$val" != "null" ]; then
            echo "$val"
            return
        fi
    fi
    echo "$default"
}

EMBED=$(_read_config_model "embedding.model" "EMBED_MODEL" "qwen3-embedding:0.6b")
CHAT=$(_read_config_model "llm.model" "CHAT_MODEL" "qwen2.5:1.5b")
RERANK=$(_read_config_model "reranker.model" "RERANK_MODEL" "Qwen/Qwen3-Reranker-0.6B")
SPARSE=$(_read_config_model "sparse.model" "SPARSE_MODEL" "Qdrant/bm25")

BOOT_SERVICES=(qdrant ollama docling)

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

# ── 0. Create config files if missing ──────────────────────────────────────────
if [ ! -f .env ]; then
    cp .env.example .env
    info "created .env from .env.example"
fi
if [ ! -f config.yaml ]; then
    cp config.example.yaml config.yaml 2>/dev/null || true
    if [ -f config.yaml ]; then
        info "created config.yaml from config.example.yaml"
    fi
fi

# ── 1. Python environment ──────────────────────────────────────────────────
echo "[1/8] Python environment"
if uv sync --extra local; then
    ok "deps installed (with in-process ML)"
else
    info "Python deps failed — run 'uv sync --extra local' manually later"
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
