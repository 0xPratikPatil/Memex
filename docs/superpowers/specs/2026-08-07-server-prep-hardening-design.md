# Server Prep Hardening — Design

**Status:** Approved (pending spec review)
**Date:** 2026-08-07
**Target:** Ubuntu 22.04 / 24.04 cloud server (headless, GPU)

## Problem

Two recurring Docker warnings/errors appear on fresh Ubuntu cloud servers, and the NVIDIA GPU setup is not fully verified before services start:

1. `WARNING: Your kernel does not support swap limit capabilities` — root cause is kernel boot params missing memory+swap accounting.
2. GPU containers intermittently failing with `Failed to initialize NVML: Unknown Error` after `systemctl daemon-reload` — the known NVIDIA + systemd cgroup driver issue.
3. No runtime verification that GPU passthrough actually works inside containers before the stack is brought up.

`setup.sh` already installs Docker + NVIDIA Container Toolkit but does not touch kernel/cgroup config and does not verify GPU passthrough.

## Goal

Extend `setup.sh` with an idempotent, non-destructive "server hardening" phase that, on Ubuntu 22+:

1. Fixes the swap-limit kernel warning (GRUB `swapaccount=1`), gated behind a manual reboot.
2. Writes a merge-safe `/etc/docker/daemon.json` that sets `cgroupfs` cgroup driver (NVIDIA mitigation) and `nvidia` as the default runtime.
3. Verifies GPU passthrough with a real `docker run --gpus all nvidia-smi` check.
4. Never aborts setup on a hardware/OS quirk — non-critical steps warn and continue; only a missing `sudo`, a required reboot, or an invalid daemon.json are hard stops.

## Research Basis (official docs)

| Issue | Official source | Fix |
|-------|----------------|-----|
| Swap limit warning | Docker "Kernel cgroup swap limit capabilities" troubleshooting | `GRUB_CMDLINE_LINUX="cgroup_enable=memory swapaccount=1"` → `update-grub` → reboot |
| GPU loss on `systemctl daemon-reload` | NVIDIA Container Toolkit troubleshooting | `{"exec-opts": ["native.cgroupdriver=cgroupfs"]}` in `/etc/docker/daemon.json` |
| NVIDIA toolkit setup | NVIDIA install guide | driver → toolkit → `nvidia-ctk runtime configure --runtime=docker` → restart docker |
| GPU verification | NVIDIA sample workload | `docker run --rm --gpus all nvidia-smi` |

## Design

### New CLI flag

`./setup.sh --no-hardening` — skips the server-hardening phase entirely (safety valve for exotic hosts).

### New phase: Server hardening (Step 0.5, runs before services start)

Runs only when:
- `sudo` is available (via the existing `need_sudo` helper), and
- the host is Ubuntu/Debian with `apt-get`, and
- `--no-hardening` was not passed.

Ordering matters: the phase runs **before** `docker compose up -d`, so a Docker daemon restart cannot kill live services.

#### A. Swap / cgroup kernel fix (root cause of "no swap limit")

1. If `/proc/cmdline` already contains `swapaccount=1` → skip (idempotent, already fixed).
2. Else if `/etc/default/grub` exists **and** `update-grub` is installed:
   - Append `cgroup_enable=memory swapaccount=1` to `GRUB_CMDLINE_LINUX` in `/etc/default/grub`, deduping params already present. Never touch other options.
   - Run `sudo update-grub`.
   - Print a prominent banner: **"Reboot required — re-run ./setup.sh after reboot to continue."**
   - `exit 0` (setup stops cleanly; nothing else has been started yet).
3. Else (no GRUB, or minimal cloud image): print manual instructions, **continue** — this is a warning fix, not a blocker.

#### B. Docker daemon.json (cgroup driver + nvidia default runtime)

Target merged config (preserves any existing keys such as `data-root`):

```json
{
  "exec-opts": ["native.cgroupdriver=cgroupfs"],
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": { "path": "nvidia-container-runtime", "runtimeArgs": [] }
  }
}
```

Merge procedure (never blind-overwrite):
1. If `/etc/docker/daemon.json` exists, read it; else start with `{}`.
2. Merge the three keys above using `jq` (already a prereq); fall back to `python3` if jq is missing.
3. Validate the merged JSON with `python3 -m json.tool` **before** writing. If invalid → restore the original file and abort with a clear message.
4. Back up the original to `/etc/docker/daemon.json.bak`, then write the merged result.
5. `sudo systemctl restart docker`. If restart fails → print the error but continue (the nvidia runtime may already be configured).

Idempotency: if the three target keys already match, no write and no restart occur.

#### C. GPU passthrough verification

1. If no NVIDIA GPU detected on host (`lspci | grep -qi nvidia` or `/dev/nvidiactl` exists — same check as existing code) → skip, print note, continue.
2. Else run `docker run --rm --gpus all <nvidia/cuda:base image> nvidia-smi` with a generous pull timeout (e.g. 180s). Image pinned to a known tag (no `:latest`).
3. On success → `ok`.
4. On failure → fall back to daemon-level checks (`docker info` shows the `nvidia` runtime and CDI devices). If those also fail, print an actionable message and continue — GPU is needed for acceleration, but setup must not crash.

### Error-handling contract

| Failure | Behavior |
|---------|----------|
| No `sudo` | Hard stop (existing `need_sudo` behavior) |
| Reboot required | Hard stop with clear re-run instructions |
| Invalid daemon.json after merge | Restore backup, hard stop with clear message |
| GRUB missing / `update-grub` absent | Warn + manual instructions, continue |
| Docker restart fails | Warn, continue |
| GPU verify fails (no GPU) | Skip, continue |
| GPU verify fails (GPU present) | Warn with fallback checks, continue |

### Scope guards (YAGNI)

- No changes to `docker-compose.yml` (kernel/cgroup config is not settable from compose; per-service `deploy.resources.devices` already handles GPU).
- No auto-reboot — manual reboot gate only.
- No new runtime tools beyond the existing prereqs (`jq`, `python3`, `curl`, `sudo`).

## Testing

1. **Syntax/static:** `bash -n setup.sh`; `shellcheck setup.sh` (repo uses shellcheck).
2. **Fresh Ubuntu 22.04/24.04:** run `./setup.sh` → swap warning present before, gone after reboot + re-run; `docker info` shows `nvidia` default runtime; `docker run --rm --gpus all nvidia-smi` succeeds.
3. **Idempotency:** run `./setup.sh` twice → no duplicate GRUB params, daemon.json byte-identical on second run, no Docker restart triggered.
4. **No-GPU path:** on a CPU-only box, setup completes with NVIDIA steps skipped and no errors.
5. **`--no-hardening`:** phase skipped entirely.
6. **Existing daemon.json merge:** pre-seed a config with extra keys (`data-root`) → keys preserved after merge.

## Files touched

- `setup.sh` — new hardening phase, new `--no-hardening` flag, updated header comment.
- `README.md` — brief note under setup about the hardening phase + reboot gate.
- `DOCKER.md` — document the daemon.json rationale and the reboot requirement.
