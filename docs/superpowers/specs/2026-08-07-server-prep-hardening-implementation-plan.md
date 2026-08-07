# Implementation Plan — Server Prep Hardening

**Spec:** `docs/superpowers/specs/2026-08-07-server-prep-hardening-design.md`
**Date:** 2026-08-07
**Target:** Ubuntu 22.04 / 24.04 cloud server

## Phase 1: Hardening phase skeleton + `--no-hardening` flag

### Task 1.1: Add `--no-hardening` CLI flag
- Edit `setup.sh`:
  - Add `NO_HARDENING=false`; parse `--no-hardening` in the existing arg loop (alongside `--skip-prereqs`)
  - Reject unknown args (existing behavior)
- Update header comment to document `--no-hardening`

### Task 1.2: Add hardening phase function + invocation point
- Add `server_hardening()` function after `install_prereqs`
- Call it between `install_prereqs` and the config-file creation block (before Docker services start)
- Guards:
  - `[ "$NO_HARDENING" = true ] && return` (skip entirely)
  - `command -v apt-get` required; else print note and return
  - `need_sudo` already handled by prereqs; if `sudo -n true` fails, proceed anyway (functions degrade gracefully)

### Task 1.3: Guard the step numbering
- Current phases use `[N/8]`. Add hardening as `[0.5/8] Server hardening` before the existing `[0/8]`? No — renumber instead:
  - Change labels to a single source of truth: keep `[0/8] System prerequisites`, insert `[1/9] Server hardening`, and bump the remaining 8 steps to `2/9`..`9/9`
- Keep it simple: insert `[1/9]` and renumber subsequent echo labels (`[1/8]`→`[2/9]`, etc.). Update all 8 existing labels.

**Checkpoint:** `bash -n setup.sh`; `shellcheck setup.sh` passes

---

## Phase 2: Swap / cgroup kernel fix (GRUB)

### Task 2.1: Implement `fix_swap_kernel_params()`
- Logic:
  1. If `grep -q swapaccount=1 /proc/cmdline` → `ok "swap accounting (active)"` and return
  2. Else if `/etc/default/grub` exists AND `command -v update-grub`:
     - Ensure `GRUB_CMDLINE_LINUX` line exists (append if missing)
     - Append `cgroup_enable=memory swapaccount=1` to the value, deduping tokens already present (use a small sed/awk edit; preserve quotes and other tokens)
     - `sudo update-grub`
     - Print prominent reboot banner (see Task 2.2) and `exit 0`
  3. Else: print manual instructions (`GRUB_CMDLINE_LINUX="cgroup_enable=memory swapaccount=1"`, `sudo update-grub`, reboot), then return (continue setup)

### Task 2.2: Reboot gate banner
- After successful GRUB update, print:
  ```
  ╔══════════════════════════════════════════╗
  ║   Reboot required — re-run ./setup.sh    ║
  ║   after reboot to continue.              ║
  ╚══════════════════════════════════════════╝
  ```
- `exit 0` (clean stop — nothing else started yet)

### Task 2.3: Idempotency test
- Manual: run with `swapaccount=1` already in `/proc/cmdline` (simulate by mocking) → skips edit, no duplicate params

**Checkpoint:** `bash -n setup.sh`; `shellcheck setup.sh` passes

---

## Phase 3: Docker daemon.json merge + cgroup driver

### Task 3.1: Implement `write_daemon_json()`
- Target keys:
  ```json
  {
    "exec-opts": ["native.cgroupdriver=cgroupfs"],
    "default-runtime": "nvidia",
    "runtimes": { "nvidia": { "path": "nvidia-container-runtime", "runtimeArgs": [] } }
  }
  ```
- Merge procedure:
  1. Read existing `/etc/docker/daemon.json` or start with `{}`
  2. If `command -v jq` → `jq -s '.[0] * .[1]' existing target`; else use `python3` merge
  3. Validate merged JSON with `python3 -m json.tool` before writing
  4. If already equal to existing → skip write + restart
  5. Backup to `/etc/docker/daemon.json.bak`, write merged file, `sudo systemctl restart docker`

### Task 3.2: Error handling
- Invalid merged JSON → restore `.bak`, print clear error, `exit 1`
- `systemctl restart docker` fails → print error, continue (nvidia runtime may already be configured)

**Checkpoint:** `bash -n setup.sh`; `shellcheck setup.sh` passes; manual merge test with pre-seeded `data-root` key

---

## Phase 4: GPU passthrough verification

### Task 4.1: Implement `verify_gpu_passthrough()`
- GPU presence check (same as existing `install_prereqs`): `lspci | grep -qi nvidia` or `ls /dev/nvidiactl`
- If no GPU → print note, return
- Else:
  - `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` with a 180s pull timeout (wrap in `timeout`)
  - On success → `ok`
  - On failure → fallback: `docker info` shows `nvidia` runtime + CDI devices; print actionable message; continue

### Task 4.2: Wire into hardening phase
- Call `verify_gpu_passthrough` as the last step of `server_hardening()`

**Checkpoint:** `bash -n setup.sh`; `shellcheck setup.sh` passes; manual run on GPU host verifies success path; CPU-only host skips cleanly

---

## Phase 5: Docs + final verification

### Task 5.1: Update README.md
- Under setup: note the hardening phase, the reboot gate (`re-run ./setup.sh`), and `--no-hardening`

### Task 5.2: Update DOCKER.md
- Document daemon.json rationale (cgroupfs mitigation for NVIDIA GPU-loss issue + nvidia default runtime) and the swap-account reboot requirement

### Task 5.3: Full verification
- `bash -n setup.sh`
- `shellcheck setup.sh`
- Manual smoke on Ubuntu 22.04/24.04: fresh run → reboot gate; re-run after reboot → swap warning gone, `docker info` shows nvidia default runtime, `docker run --rm --gpus all nvidia-smi` succeeds
- Idempotency: second run makes no GRUB/daemon.json changes, no Docker restart

**Checkpoint:** all static checks pass; manual server verification complete
