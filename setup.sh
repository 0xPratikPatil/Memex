#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Memex Bootstrap — one command to ready everything.
#
#   ./setup.sh                              # use config.yaml defaults
#   EMBED_MODEL=llama3.2:1b ./setup.sh      # override embedding model
#   CHAT_MODEL=qwen2.5:1.5b ./setup.sh      # override chat model
#   ./setup.sh --skip-prereqs               # skip system tool install (repeat runs)
#   ./setup.sh --no-hardening               # skip server hardening (swap/cgroup/GPU)
#
# What it does:
#   0. Auto-installs missing system prerequisites (Ubuntu/Debian): curl, git,
#      python3, make, jq, uv, Docker Engine + Compose, NVIDIA driver + Toolkit
#   1. Server hardening (idempotent): kernel swap accounting (GRUB), Docker
#      daemon.json (cgroupfs + nvidia default runtime), GPU passthrough check.
#      May require a reboot — re-run ./setup.sh after rebooting to continue.
#   2. Creates .env from .env.example if missing (secrets only)
#   3. Creates config.yaml from config.example.yaml if missing
#   4. Installs Python deps (uv sync) into project .venv
#   5. Checks Docker is running
#   6. Builds and starts all backend services
#   7. Waits for health checks
#   8. Pulls Ollama models (reads from config.yaml, env var overrides work)
#   9. Verifies models + features respond
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")"

# ── CLI flags ────────────────────────────────────────────────────────────────
SKIP_PREREQS=false
NO_HARDENING=false
for arg in "$@"; do
    case "$arg" in
        --skip-prereqs) SKIP_PREREQS=true ;;
        --no-hardening) NO_HARDENING=true ;;
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
unset HTTP_TIMEOUT QDRANT_TIMEOUT MARKER_TIMEOUT 2>/dev/null || true

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

# ── Dynamic service list (reads converter.engine from config.yaml) ──────────
CONVERTER=$(_read_config_model "converter.engine" "CONVERTER_ENGINE" "marker")

# Base services — always needed
BOOT_SERVICES=(qdrant ollama redis)

# Converter-specific services
case "$CONVERTER" in
    marker)
        BOOT_SERVICES+=(marker ml-services ocr)
        ;;
    markitdown)
        BOOT_SERVICES+=(markitdown ocr)
        ;;
    docling)
        # Legacy: still needs marker for conversion
        BOOT_SERVICES+=(marker ml-services)
        ;;
    *)
        info "unknown converter engine '$CONVERTER' — starting all services"
        BOOT_SERVICES+=(marker ml-services markitdown ocr)
        ;;
esac

# ── Helpers ──────────────────────────────────────────────────────────────────
ok()   { echo "  ✓ $1"; }
fail() { echo "  ✗ $1"; exit 1; }
info() { echo "  → $1"; }

# ── GPU detection (new-machine auto-config) ───────────────────────────────────
# Detects GPU VRAM and sets marker mode / gpu coordination accordingly.
# Writes config.yaml ONLY if it does not exist (never clobbers user config).
detect_gpu() {
    local vram_mb=""
    local mode="fast"
    local gpu_enabled="true"

    if command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits &>/dev/null; then
        vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
        info "GPU detected: ${vram_mb}MB VRAM"
        if [ -n "$vram_mb" ] && [ "$vram_mb" -ge 16384 ] 2>/dev/null; then
            mode="balanced"       # big GPU: high quality, no contention
            gpu_enabled="false"
            info "  → large GPU: marker_mode=balanced, gpu.enabled=false (concurrent)"
        elif [ -n "$vram_mb" ] && [ "$vram_mb" -ge 8192 ] 2>/dev/null; then
            mode="fast"           # small GPU: fast mode + mutual exclusion
            gpu_enabled="true"
            info "  → small GPU: marker_mode=fast, gpu.enabled=true (mutual exclusion)"
        else
            gpu_enabled="true"
            info "  → very small GPU: marker_mode=fast, gpu.enabled=true"
        fi
    else
        info "no NVIDIA GPU detected — marker will run on CPU (slower, but works)"
        mode="fast"
        gpu_enabled="false"
    fi

    # Auto-write config.yaml only when absent.
    if [ ! -f config.yaml ]; then
        if [ -f config.example.yaml ]; then
            cp config.example.yaml config.yaml
            info "created config.yaml from example"
        fi
    fi
    if [ -f config.yaml ]; then
        # Update marker_mode + gpu settings in place (idempotent).
        if grep -q "marker_mode:" config.yaml; then
            sed -i "s/^  marker_mode:.*/  marker_mode: $mode  # auto-detected by setup.sh/" config.yaml
        fi
        if grep -q "gpu:" config.yaml; then
            sed -i "s/^  enabled: true.*# enforce marker.*/  enabled: $gpu_enabled  # auto-detected by setup.sh/" config.yaml
            sed -i "s/^  enabled: false.*# enforce marker.*/  enabled: $gpu_enabled  # auto-detected by setup.sh/" config.yaml
        fi
        info "config.yaml: marker_mode=$mode gpu.enabled=$gpu_enabled"
    fi
}

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

install_git_lfs() {
    if command -v git-lfs &>/dev/null; then
        ok "git-lfs (present)"
        return 0
    fi
    info "installing git-lfs"
    if sudo apt-get install -y git-lfs &>/dev/null; then
        git lfs install &>/dev/null || true
        ok "git-lfs"
    else
        echo "  ✗ git-lfs install failed." >&2
        echo "    Try manually: sudo apt-get install -y git-lfs && git lfs install" >&2
    fi
}

install_prereqs() {
    [ "$SKIP_PREREQS" = true ] && { info "skipping system prerequisites (--skip-prereqs)"; return; }

    echo "[0/9] System prerequisites"
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
    install_git_lfs
    install_docker

    # GPU-aware: only attempt NVIDIA if a GPU is present
    if lspci 2>/dev/null | grep -qi nvidia || ls /dev/nvidiactl &>/dev/null; then
        install_nvidia
    else
        info "no NVIDIA GPU detected — ollama/marker will need CPU config or manual GPU setup"
    fi
}

# ── Server hardening (Step 1) ────────────────────────────────────────────────
# Idempotent, non-destructive server prep. Never crashes setup on a hardware/OS
# quirk — non-critical steps warn and continue. Hard stops are reserved for a
# required reboot (clean exit) and an invalid daemon.json.
server_hardening() {
    [ "$NO_HARDENING" = true ] && { info "skipping server hardening (--no-hardening)"; return; }

    echo "[1/9] Server hardening"

    # Ubuntu/Debian with GRUB only — otherwise skip gracefully.
    if ! command -v apt-get &>/dev/null; then
        info "not Ubuntu/Debian — skipping server hardening"
        return
    fi

    fix_swap_kernel_params
    write_daemon_json
    verify_gpu_passthrough

    ok "hardening complete"
}

# ── Swap / cgroup kernel fix (GRUB) ─────────────────────────────────────────
# Adds memory+swap accounting to the kernel cmdline. Fixes Docker's
# "no swap limit" warning. Requires a reboot to take effect.
fix_swap_kernel_params() {
    if grep -q 'swapaccount=1' /proc/cmdline 2>/dev/null; then
        ok "swap accounting (active)"
        return
    fi

    if [ ! -f /etc/default/grub ] || ! command -v update-grub &>/dev/null; then
        echo "  → swap accounting not enabled. To enable manually:" >&2
        echo "    sudo sed -i 's/^GRUB_CMDLINE_LINUX=\"\"/GRUB_CMDLINE_LINUX=\"cgroup_enable=memory swapaccount=1\"/' /etc/default/grub" >&2
        echo "    sudo update-grub && sudo reboot" >&2
        echo "    (continuing — this only silences Docker's swap warning)" >&2
        return
    fi

    info "enabling kernel swap accounting (GRUB)"
    local grub_file="/etc/default/grub"

    # Use python3 (guaranteed prereq) for a robust, idempotent edit — safer
    # than sed string surgery against arbitrary existing GRUB values.
    if ! python3 - "$grub_file" <<'EOF'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()

params = {"cgroup_enable=memory", "swapaccount=1"}
existing = set()
m = re.search(r'^GRUB_CMDLINE_LINUX="(.*)"\s*$', text, re.M)
if m:
    existing = set(m.group(1).split())
missing = params - existing
if not missing:
    print("already configured")
    sys.exit(0)

to_add = " ".join(sorted(missing))
if m:
    new_val = (m.group(1) + " " + to_add).strip()
    text = text.replace(m.group(0), f'GRUB_CMDLINE_LINUX="{new_val}"')
else:
    text += '\nGRUB_CMDLINE_LINUX="%s"\n' % to_add

path.write_text(text)
print("updated")
EOF
    then
        echo "  ✗ failed to edit /etc/default/grub — edit manually and add:" >&2
        echo '    GRUB_CMDLINE_LINUX="cgroup_enable=memory swapaccount=1"' >&2
        return
    fi

    if sudo update-grub; then
        ok "GRUB updated (swap accounting)"
    else
        echo "  ✗ update-grub failed — run 'sudo update-grub' manually." >&2
        return
    fi

    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Reboot required — swap accounting is active only after      ║"
    echo "║  reboot. Re-run ./setup.sh after rebooting to continue.      ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    exit 0
}

# ── Docker daemon.json ──────────────────────────────────────────────────────
# Merge-safe write of /etc/docker/daemon.json:
#   - cgroupfs cgroup driver (NVIDIA mitigation for GPU loss on daemon-reload)
#   - nvidia as the default runtime
# Preserves existing keys. Validates JSON before writing; backs up original.
write_daemon_json() {
    local daemon="/etc/docker/daemon.json"
    local target
    target='{"exec-opts":["native.cgroupdriver=cgroupfs"],"default-runtime":"nvidia","runtimes":{"nvidia":{"path":"nvidia-container-runtime","runtimeArgs":[]}}}'

    # Read existing config (or empty object).
    local existing="{}"
    if [ -f "$daemon" ]; then
        existing=$(cat "$daemon")
    fi

    # Merge using jq if available, else python3.
    local merged=""
    if command -v jq &>/dev/null; then
        merged=$(printf '%s' "$existing" | jq -c ". * $target" 2>/dev/null) || merged=""
    elif command -v python3 &>/dev/null; then
        merged=$(python3 -c "
import json, sys
existing = json.loads(sys.argv[1]) if sys.argv[1].strip() else {}
target = json.loads(sys.argv[2])
existing.update(target)
print(json.dumps(existing))
" "$existing" "$target" 2>/dev/null) || merged=""
    fi

    if [ -z "$merged" ]; then
        echo "  ✗ failed to merge daemon.json (invalid existing config?)" >&2
        echo "    Fix /etc/docker/daemon.json manually, then re-run." >&2
        return
    fi

    # If nothing to change, skip write + restart (idempotent).
    if [ "$merged" = "$existing" ]; then
        ok "daemon.json (already configured)"
        return
    fi

    # Validate JSON before writing.
    if ! printf '%s' "$merged" | python3 -m json.tool >/dev/null 2>&1; then
        echo "  ✗ merged daemon.json is invalid JSON — aborting write." >&2
        echo "    No changes were made to /etc/docker/daemon.json." >&2
        exit 1
    fi

    # Backup original, write merged, restart docker.
    if [ -f "$daemon" ]; then
        sudo cp "$daemon" "$daemon.bak" 2>/dev/null || true
    fi
    printf '%s\n' "$merged" | sudo tee "$daemon" >/dev/null
    ok "daemon.json written (cgroupfs + nvidia default runtime)"

    if sudo systemctl restart docker; then
        ok "docker daemon restarted"
    else
        echo "  → docker restart failed — run 'sudo systemctl restart docker' manually." >&2
    fi
}

# ── GPU passthrough verification ────────────────────────────────────────────
# Confirms containers can actually see the GPU. Non-blocking on failure.
verify_gpu_passthrough() {
    if ! lspci 2>/dev/null | grep -qi nvidia && [ ! -e /dev/nvidiactl ]; then
        info "no NVIDIA GPU detected — skipping GPU passthrough check"
        return
    fi

    info "verifying GPU passthrough (docker run --gpus all nvidia-smi)"
    if timeout 180 docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
        ok "GPU passthrough verified"
    else
        echo "  → GPU passthrough check failed (image pull or GPU access)." >&2
        echo "    Fallback checks:" >&2
        docker info 2>/dev/null | grep -q "nvidia" \
            && echo "    ✓ docker info shows nvidia runtime" \
            || echo "    ✗ nvidia runtime not found in docker info — run: sudo nvidia-ctk runtime configure --runtime=docker" >&2
        echo "    (continuing — GPU is for acceleration, not required for correctness)" >&2
    fi
}

# ── Banner ──────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║            Memex Bootstrap              ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  embed      ${EMBED}"
echo "  chat       ${CHAT}"
echo "  rerank     ${RERANK}"
echo "  sparse     ${SPARSE}"
echo "  converter  ${CONVERTER}"
echo "  services   ${BOOT_SERVICES[*]}"
echo ""

# ── 0. System prerequisites ──────────────────────────────────────────────────
install_prereqs

# ── 1. Server hardening ─────────────────────────────────────────────────────
server_hardening

# ── 2. Create config files if missing ─────────────────────────────────────────
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

# Detect GPU and auto-tune marker mode + GPU coordination for this machine.
detect_gpu

# ── 3. Python environment ──────────────────────────────────────────────────
echo "[3/9] Python environment"
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

# ── 4. Docker ───────────────────────────────────────────────────────────────
echo "[4/9] Docker"
docker info >/dev/null 2>&1 || fail "Docker not running"
ok "running"

# ── 5. Start services ───────────────────────────────────────────────────────
echo "[5/9] Services"
docker compose up -d --build --remove-orphans "${BOOT_SERVICES[@]}"
ok "started"

# ── 6. Health checks ────────────────────────────────────────────────────────
echo "[6/9] Health checks"

check_http() {
    local name="$1" url="$2" timeout="${3:-60}"
    local elapsed=0
    while ! curl -sf --max-time 3 "$url" >/dev/null 2>&1; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ "$elapsed" -ge "$timeout" ]; then
            echo "  ✗ $name: health check timed out after ${timeout}s"
            return 1
        fi
    done
    ok "$name"
}

# Health check URLs per service
declare -A HEALTH_URLS=(
    [qdrant]="http://localhost:6333/"
    [ollama]="http://localhost:11434/api/tags"
    [redis]="redis-cli"
    [marker]="http://localhost:5001/health"
    [ml-services]="http://localhost:5002/health"
    [markitdown]="http://localhost:5003/health"
    [ocr]="http://localhost:5004/health"
)

for svc in "${BOOT_SERVICES[@]}"; do
    if ! docker compose ps -q "$svc" &>/dev/null; then
        info "${svc}: not in compose, skipping"
        continue
    fi
    elapsed=0
    while ! docker compose ps "$svc" 2>/dev/null | tail -n+2 | grep -q "healthy"; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ "$elapsed" -ge 60 ]; then
            echo "  ✗ $svc: never became healthy after 60s"
            break
        fi
    done
    url="${HEALTH_URLS[$svc]:-}"
    if [ "$url" = "redis-cli" ]; then
        if docker compose exec -T redis redis-cli ping 2>/dev/null | grep -q "PONG"; then
            ok "$svc"
        else
            echo "  ✗ $svc: redis-cli ping failed"
        fi
    elif [ -n "$url" ]; then
        check_http "$svc" "$url" 60
    else
        ok "$svc (no health URL configured)"
    fi
done

# ── 7. Pull models ──────────────────────────────────────────────────────────
echo "[7/9] Models"
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

# ── 8. Verify models loaded ─────────────────────────────────────────────────
echo "[8/9] Verify models"
curl -sf -X POST http://localhost:11434/api/embeddings \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${EMBED}\",\"prompt\":\"test\"}" >/dev/null \
    && ok "${EMBED}" || fail "${EMBED} not responding"
curl -sf -X POST http://localhost:11434/api/chat \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${CHAT}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}" >/dev/null \
    && ok "${CHAT}" || fail "${CHAT} not responding"

# Pre-flight: Ollama chat latency (catches GPU contention at setup time)
echo "  → measuring Ollama chat latency (${CHAT})…"
_CHAT_START=$(date +%s%N)
curl -sf -X POST http://localhost:11434/api/chat \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${CHAT}\",\"messages\":[{\"role\":\"user\",\"content\":\"say ok\"}],\"stream\":false}" >/dev/null \
    || { echo "  ✗ chat latency check failed — GPU may be contended" >&2; }
_CHAT_END=$(date +%s%N)
_CHAT_MS=$(( (_CHAT_END - _CHAT_START) / 1000000 ))
if [ "$_CHAT_MS" -lt 5000 ] 2>/dev/null; then
    ok "Ollama chat responds in ${_CHAT_MS}ms"
else
    echo "  → Ollama chat took ${_CHAT_MS}ms — GPU may be busy or models cold (non-fatal)" >&2
fi

# ── 9. Verify features ─────────────────────────────────────────────────────
echo "[9/9] Features"
# Converter availability
case "$CONVERTER" in
    marker)
        if uv run python -c "
from memex.engine.ingestion.marker_client import is_marker_available
ok = is_marker_available()
assert ok, 'Marker not available — check marker service'
" 2>&1; then
            echo "  ✓ marker converter"
        else
            echo "  ✗ marker converter (non-fatal) — run: docker compose up -d marker"
        fi
        ;;
    markitdown)
        if curl -sf http://localhost:5003/health >/dev/null 2>&1; then
            echo "  ✓ markitdown converter"
        else
            echo "  ✗ markitdown converter (non-fatal) — run: docker compose up -d markitdown"
        fi
        ;;
    *)
        echo "  → converter '$CONVERTER' — skipping feature check"
        ;;
esac

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
