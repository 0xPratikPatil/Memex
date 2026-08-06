# Design: Auto-Install System Prerequisites in setup.sh

**Date**: 2026-08-06
**Status**: Draft
**Scope**: Add Step 0 to `setup.sh` that auto-installs all missing system prerequisites on Ubuntu/Debian so a fresh system can run the project with one command.

---

## Problem Statement

The current `setup.sh` assumes `uv`, `docker`, `docker compose`, `curl`, `python3`, and an NVIDIA Container Toolkit are already installed. On a fresh system it fails immediately (`fail "Docker not running"`). A new machine needs a one-command path from bare OS to running MCP server.

---

## Design

Insert a **Step 0 — System prerequisites** section into the existing `setup.sh`, before the current Step 1. It auto-installs only what's missing, is idempotent, GPU-aware, and non-fatal per component.

### Tools installed (only if missing)

| Tool | Why | Install method |
|------|-----|----------------|
| `curl`, `ca-certificates` | health checks, uv installer | `apt-get` |
| `git` | repo management | `apt-get` |
| `python3`, `python3-venv`, `python3-pip` | uv bootstrap, `requires-python>=3.12` | `apt-get` |
| `make` | `make test/lint/typecheck` | `apt-get` |
| `jq` | JSON verify in docs | `apt-get` |
| `uv` | Python env + deps | `curl https://astral.sh/uv/install.sh | sh` |
| Docker Engine + CLI | containers | official Docker apt repo |
| Docker Compose plugin | `docker compose` | official Docker apt repo |
| NVIDIA driver + Container Toolkit | GPU passthrough (ollama/docling) | `ubuntu-drivers` + NVIDIA apt repo — **only if GPU detected** |

### Behavior rules

1. **Idempotent**: each tool checked via `command -v`; installed only when absent. Re-runs are fast and safe.
2. **Sudo**: system packages installed via `sudo`. If `sudo` unavailable or non-interactive, print clear error + manual instructions and exit.
3. **GPU-aware**: detect NVIDIA GPU via `lspci | grep -i nvidia` or `nvidia-smi`. Only then install driver + Container Toolkit. If no GPU, warn that ollama/docling will run CPU-only or need manual config, then continue.
4. **Graceful failure**: each component wrapped in a helper that prints a clear message and manual commands on failure, rather than silently aborting the whole script.
5. **`--skip-prereqs`**: skip system tool install for repeat runs; only run app/bootstrap steps.
6. **Reboot warning**: after NVIDIA driver install, if `nvidia-smi` is not live, print "reboot may be required" and continue (Docker services still start; GPU requests may fail until reboot).

### Argument parsing

Add `--skip-prereqs` flag parse at the top of the script. All other args remain unchanged.

---

## Files Modified

| File | Change |
|------|--------|
| `setup.sh` | Add Step 0 prerequisites section, `--skip-prereqs` flag, helper functions, renumber step labels to include 0 |

---

## Testing

1. `bash -n setup.sh` — syntax check.
2. ShellCheck if available — `shellcheck setup.sh`.
3. Manual dry-run on a fresh Ubuntu system (or VM): run `./setup.sh --skip-prereqs` to verify app steps unaffected.
4. Run `./setup.sh` on a system missing only `uv` to verify it installs uv then proceeds.
5. Run twice — second run reports all tools "present" and skips installs.

---

## Risks

- **NVIDIA driver install** may require a reboot (kernel module). Non-fatal: script continues and warns.
- **`sudo` interaction**: on a headless fresh install the apt steps may prompt for a password. Documented as needing a sudo-capable user.
- **Docker apt repo** requires `apt-get update` before install; handled in the same block.
