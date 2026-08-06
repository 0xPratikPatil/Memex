#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Memex Bootstrap — one command to ready everything.
#
#   ./setup.sh                              # use config.yaml defaults
#   EMBED_MODEL=llama3.2:1b ./setup.sh      # override embedding model
#   CHAT_MODEL=qwen2.5:1.5b ./setup.sh      # override chat model
#   ./setup.sh --skip-prereqs               # skip system tool install (repeat runs)
#
# What it does:
#   0. Auto-installs missing system prerequisites (Ubuntu/Debian): curl, git,
#      python3, make, jq, uv, Docker Engine + Compose, NVIDIA driver + Toolkit
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

# ── CLI flags ────────────────────────────────────────────────────────────────
SKIP_PREREQS=false
for arg in "$@"; do
    case "$arg" in
        --skip-prereqs) SKIP_PREREQS=true ;;
        *) echo "  ✗ Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

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

BOOT_SERVICES=(qdrant ollama docling ml-services)

# ── Helpers ──────────────────────────────────────────────────────────────────
ok()   { echo "  ✓ $1"; }
fail() { echo "  ✗ $1"; exit 1; }
info() { echo "  → $1"; }

# ── Prerequisite helpers (Step 0) ───────────────────────────────────────────
need_sudo() {
    if [ "$(id -u)" -eq 0 ]; then
        return 0
    fi
    if ! command -v sudo &>/dev/null; then
        echo "  ✗ 'sudo' not found — re-run as root or install sudo." >&2
        exit 1
    fi
    if ! sudo -n true 2>/dev/null; then
        echo "  → sudo requires a password — entering interactive mode." >&2
    fi
    return 0
}

apt_install() {
    local pkg="$1"
    if dpkg -s "$pkg" &>/dev/null; then
        ok "$pkg (present)"
        return 0
    fi
    info "installing $pkg"
    if sudo apt-get install -y "$pkg" &>/dev/null; then
        ok "$pkg"
    else
        echo "  ✗ apt-get install $pkg failed." >&2
        echo "    Try manually: sudo apt-get install -y $pkg" >&2
    fi
}

install_uv() {
    if command -v uv &>/dev/null; then
        ok "uv (present)"
        return 0
    fi
    info "installing uv"
    if curl -LsSf https://astral.sh/uv/install.sh | sh &>/dev/null; then
        # astral installer puts uv in ~/.local/bin; ensure it's on PATH for this run
        export PATH="$HOME/.local/bin:$PATH"
        ok "uv"
    else
        echo "  ✗ uv install failed." >&2
        echo "    Try manually: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    fi
}

install_docker() {
    if command -v docker &>/dev/null && docker compose version &>/dev/null; then
        ok "docker + compose (present)"
        return 0
    fi
    info "installing Docker Engine + Compose via official apt repo"
    if sudo apt-get update &>/dev/null \
        && sudo apt-get install -y ca-certificates curl gnupg &>/dev/null \
        && sudo install -m 0755 -d /etc/apt/keyrings &>/dev/null \
        && curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
            | sudo gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg &>/dev/null \
        && sudo chmod a+r /etc/apt/keyrings/docker.gpg &>/dev/null; then
        # shellcheck disable=SC1091
        . /etc/os-release
        echo \
            "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu ${VERSION_CODENAME:-stable} stable" \
            | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
        if sudo apt-get update &>/dev/null \
            && sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
                docker-buildx-plugin docker-compose-plugin &>/dev/null; then
            ok "docker + compose"
        else
            echo "  ✗ Docker apt install failed." >&2
        fi
    else
        echo "  ✗ Docker repo setup failed." >&2
    fi
}

install_nvidia() {
    if command -v nvidia-smi &>/dev/null; then
        ok "NVIDIA driver (present)"
    else
        info "installing NVIDIA driver"
        if sudo apt-get install -y ubuntu-drivers-common &>/dev/null \
            && sudo ubuntu-drivers install &>/dev/null; then
            info "NVIDIA driver installed — reboot may be required"
        else
            echo "  ✗ NVIDIA driver install failed — run 'sudo ubuntu-drivers install' manually." >&2
        fi
    fi

    # Container Toolkit: required for GPU passthrough to Docker containers.
    if [ -x /usr/bin/nvidia-ctk ] || [ -x /usr/local/bin/nvidia-ctk ]; then
        ok "NVIDIA Container Toolkit (present)"
        return 0
    fi
    info "installing NVIDIA Container Toolkit"
    if curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
            | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg &>/dev/null \
        && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
            | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
            | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null \
        && sudo apt-get update &>/dev/null \
        && sudo apt-get install -y nvidia-container-toolkit &>/dev/null; then
        ok "NVIDIA Container Toolkit"
        sudo nvidia-ctk runtime configure --runtime=docker &>/dev/null \
            && info "configured docker runtime" \
            || echo "  → run manually: sudo nvidia-ctk runtime configure --runtime=docker" >&2
    else
        echo "  ✗ NVIDIA Container Toolkit install failed." >&2
    fi
}

install_prereqs() {
    [ "$SKIP_PREREQS" = true ] && { info "skipping system prerequisites (--skip-prereqs)"; return; }

    echo "[0/8] System prerequisites"
    need_sudo

    # Ubuntu/Debian only
    if ! command -v apt-get &>/dev/null; then
        echo "  ✗ apt-get not found — this script auto-installs only on Ubuntu/Debian." >&2
        echo "    Install curl, git, python3, make, jq, uv, Docker manually, then re-run." >&2
        return
    fi

    info "updating apt index"
    sudo apt-get update &>/dev/null || echo "  → apt-get update had issues, continuing" >&2

    for pkg in curl ca-certificates git python3 python3-venv python3-pip make jq; do
        apt_install "$pkg"
    done

    install_uv
    install_docker

    # GPU-aware: only attempt NVIDIA if a GPU is present
    if lspci 2>/dev/null | grep -qi nvidia || ls /dev/nvidiactl &>/dev/null; then
        install_nvidia
    else
        info "no NVIDIA GPU detected — ollama/docling will need CPU config or manual GPU setup"
    fi
}

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

# ── 0. System prerequisites ──────────────────────────────────────────────────
install_prereqs

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
# Use uv to install the pinned Python version (from .python-version) —
# independent of whatever the system apt python3 happens to be.
if command -v uv &>/dev/null; then
    PY_VER=$(cat .python-version 2>/dev/null | head -n1 || true)
    if [ -n "$PY_VER" ] && ! uv python find "$PY_VER" &>/dev/null; then
        info "installing Python ${PY_VER} via uv"
        uv python install "$PY_VER" &>/dev/null || info "uv python install failed — continuing with system python"
    fi
    if uv sync --extra local; then
        ok "deps installed (with in-process ML)"
    else
        info "Python deps failed — run 'uv sync --extra local' manually later"
    fi
else
    info "uv not found — run 'uv sync --extra local' manually later"
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
from memex.engine.ingestion.splitter import is_hybrid_chunker_available
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
